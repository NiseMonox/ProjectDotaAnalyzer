"""backend.app.osutil: the primitives that differ between POSIX and Windows.

Every helper is tested twice where that means something: for real on the
system the suite runs on (Linux and macOS during development, Windows in CI),
and with the other system's branch switched on by ``IS_WINDOWS`` and stand-ins
for the Windows-only modules, so a mistake in the Windows code shows on a
development machine that has no Windows.
"""
from __future__ import annotations

import contextlib
import errno
import io
import os
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from backend.app import osutil

REPO = Path(__file__).resolve().parents[2]
# A child process that does nothing for a minute, on every system.
SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]


def _wait_until(condition, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


@pytest.fixture
def as_windows(monkeypatch):
    """Run the Windows branches of osutil on whatever system this is."""
    monkeypatch.setattr(osutil, "IS_WINDOWS", True)


# ---------------------------------------------------------------------------
# atomic_replace
# ---------------------------------------------------------------------------

def test_atomic_replace_replaces_the_destination(tmp_path):
    src, dst = tmp_path / "new.tmp", tmp_path / "data.json"
    dst.write_bytes(b"old")
    src.write_bytes(b"new")
    osutil.atomic_replace(src, dst)
    assert dst.read_bytes() == b"new" and not src.exists()
    src.write_bytes(b"newer")
    osutil.atomic_replace(str(src), str(dst))           # str paths too
    assert dst.read_bytes() == b"newer"
    with pytest.raises(FileNotFoundError):
        osutil.atomic_replace(tmp_path / "missing.tmp", dst)


def test_atomic_replace_gets_through_while_a_reader_has_the_destination_open(tmp_path):
    """The Windows case for real: a file that is open cannot be replaced there, so the replace
    has to wait for the reader. On POSIX the same code passes at the first attempt."""
    src, dst = tmp_path / "new.tmp", tmp_path / "data.json"
    dst.write_bytes(b"old")
    src.write_bytes(b"new")
    opened, seen = threading.Event(), []

    def reader():
        with open(dst, "rb") as handle:
            opened.set()
            time.sleep(0.3)
            seen.append(handle.read())

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        assert opened.wait(5)
        osutil.atomic_replace(src, dst)
    finally:
        thread.join()
    assert dst.read_bytes() == b"new" and seen == [b"old"]


def test_atomic_replace_does_not_retry_on_posix(tmp_path, monkeypatch):
    monkeypatch.setattr(osutil, "IS_WINDOWS", False)
    calls, naps = [], []

    def denied(src, dst):
        calls.append((src, dst))
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(osutil.os, "replace", denied)
    monkeypatch.setattr(osutil.time, "sleep", naps.append)
    with pytest.raises(PermissionError):
        osutil.atomic_replace("a", "b")
    assert len(calls) == 1 and naps == []       # a directory that may not be written: reported at once


def test_atomic_replace_retries_a_destination_in_use_on_windows(tmp_path, monkeypatch, as_windows):
    src, dst = tmp_path / "new.tmp", tmp_path / "data.json"
    src.write_bytes(b"new")
    real_replace, attempts, naps = os.replace, [], []

    def busy_twice(a, b):
        attempts.append(1)
        if len(attempts) <= 2:
            raise PermissionError(errno.EACCES, "The process cannot access the file")
        real_replace(a, b)

    monkeypatch.setattr(osutil.os, "replace", busy_twice)
    monkeypatch.setattr(osutil.time, "sleep", naps.append)
    osutil.atomic_replace(src, dst, delay=0.01)
    assert len(attempts) == 3 and naps == [0.01, 0.01]
    assert dst.read_bytes() == b"new"


def test_atomic_replace_gives_up_after_its_retries_on_windows(tmp_path, monkeypatch, as_windows):
    src, dst = tmp_path / "new.tmp", tmp_path / "data.json"
    src.write_bytes(b"new")
    attempts = []

    def always_busy(a, b):
        attempts.append(1)
        raise PermissionError(errno.EACCES, "The process cannot access the file")

    monkeypatch.setattr(osutil.os, "replace", always_busy)
    monkeypatch.setattr(osutil.time, "sleep", lambda _s: None)
    with pytest.raises(PermissionError):
        osutil.atomic_replace(src, dst, retries=4)
    assert len(attempts) == 5                           # the first attempt and four more
    assert src.read_bytes() == b"new"                   # left for the caller's clean-up
    attempts.clear()
    with pytest.raises(PermissionError):
        osutil.atomic_replace(src, dst, retries=0)
    assert len(attempts) == 1

    def missing(a, b):                                  # any other error is not waited for
        attempts.append(1)
        raise FileNotFoundError(errno.ENOENT, "No such file")

    attempts.clear()
    monkeypatch.setattr(osutil.os, "replace", missing)
    with pytest.raises(FileNotFoundError):
        osutil.atomic_replace(src, dst)
    assert len(attempts) == 1


# ---------------------------------------------------------------------------
# read_bytes
# ---------------------------------------------------------------------------

def test_read_bytes_reads_and_reports_a_missing_file_at_once(tmp_path, monkeypatch, as_windows):
    path = tmp_path / "data.bin"
    path.write_bytes(b"\x00\xffcontent\r\n")
    assert osutil.read_bytes(path) == b"\x00\xffcontent\r\n"
    assert osutil.read_bytes(str(path)) == b"\x00\xffcontent\r\n"
    naps = []
    monkeypatch.setattr(osutil.time, "sleep", naps.append)
    with pytest.raises(FileNotFoundError):                  # "not cached": the callers' normal case, never waited for
        osutil.read_bytes(tmp_path / "missing.bin")
    assert naps == []


def test_read_bytes_waits_for_a_file_shut_by_another_handle_on_windows(tmp_path, monkeypatch, as_windows):
    """A replace in progress, or a scanner that shares nothing: a reader gets "permission denied"."""
    import builtins

    path = tmp_path / "data.bin"
    path.write_bytes(b"content")
    real_open, refusals, naps = builtins.open, [2], []

    def shut_out(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file) == path and refusals[0] > 0:
            refusals[0] -= 1
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", shut_out)
    monkeypatch.setattr(osutil.time, "sleep", naps.append)
    assert osutil.read_bytes(path, delay=0.01) == b"content"
    assert naps == [0.01, 0.01]
    refusals[0] = 10**6                                     # really not allowed: the error, after the retries
    with pytest.raises(PermissionError):
        osutil.read_bytes(path, retries=3)
    assert len(naps) == 5
    monkeypatch.setattr(osutil, "IS_WINDOWS", False)        # ... and at once on POSIX
    with pytest.raises(PermissionError):
        osutil.read_bytes(path)
    assert len(naps) == 5


def test_read_bytes_gets_through_while_other_threads_replace_the_file(tmp_path):
    """The Windows case for real (on POSIX nothing ever refuses the read or the replace):
    a reader of a cache file next to requests that mark it as used and replace it."""
    path = tmp_path / "cached.json"
    path.write_bytes(b"0")
    stop, errors = threading.Event(), []

    def busy():
        try:
            n = 0
            while not stop.is_set():
                n += 1
                with contextlib.suppress(OSError):          # as storage.touch(): a file being replaced
                    os.utime(path)
                fresh = tmp_path / f"cached.{threading.get_ident()}.tmp"
                fresh.write_bytes(str(n).encode("ascii"))
                osutil.atomic_replace(fresh, path)
                time.sleep(0.002)
        except BaseException as exc:
            errors.append(exc)

    workers = [threading.Thread(target=busy) for _ in range(2)]
    for worker in workers:
        worker.start()
    try:
        deadline = time.monotonic() + 0.5
        reads = 0
        while time.monotonic() < deadline:
            assert osutil.read_bytes(path).isdigit()        # always one complete version of the file
            reads += 1
            time.sleep(0.002)
    finally:
        stop.set()
        for worker in workers:
            worker.join()
    assert errors == [] and reads > 0


# ---------------------------------------------------------------------------
# remove_file
# ---------------------------------------------------------------------------

def test_remove_file_removes_and_ignores_a_missing_file(tmp_path):
    path = tmp_path / "old.json"
    path.write_bytes(b"x")
    osutil.remove_file(path)
    assert not path.exists()
    osutil.remove_file(path)                                # already gone
    osutil.remove_file(str(tmp_path / "never-there.json"))


def test_remove_file_gets_through_while_a_reader_has_the_file_open(tmp_path):
    """The Windows case for real: an open file cannot be deleted there, the removal waits for
    the reader. On POSIX the same code passes at the first attempt."""
    path = tmp_path / "old.json"
    path.write_bytes(b"content")
    opened, seen = threading.Event(), []

    def reader():
        with open(path, "rb") as handle:
            opened.set()
            time.sleep(0.3)
            seen.append(handle.read())

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        assert opened.wait(5)
        osutil.remove_file(path)
    finally:
        thread.join()
    assert not path.exists() and seen == [b"content"]


def test_remove_file_retries_only_on_windows(tmp_path, monkeypatch):
    path = tmp_path / "old.json"
    path.write_bytes(b"x")
    real_unlink, refusals, naps = Path.unlink, [2], []

    def unlink(self, *args, **kwargs):
        if refusals[0] > 0:
            refusals[0] -= 1
            raise PermissionError(errno.EACCES, "The process cannot access the file")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(osutil.time, "sleep", naps.append)
    monkeypatch.setattr(osutil, "IS_WINDOWS", False)
    with pytest.raises(PermissionError):                    # POSIX: a directory that may not be written
        osutil.remove_file(path)
    assert naps == [] and path.exists()
    monkeypatch.setattr(osutil, "IS_WINDOWS", True)
    refusals[0] = 2
    osutil.remove_file(path, delay=0.01)
    assert naps == [0.01, 0.01] and not path.exists()
    path.write_bytes(b"x")
    refusals[0] = 10**6                                     # in use for good
    with pytest.raises(PermissionError):
        osutil.remove_file(path, retries=3)
    assert len(naps) == 5 and path.exists()


# ---------------------------------------------------------------------------
# FileLock
# ---------------------------------------------------------------------------

def test_file_lock_excludes_a_second_holder(tmp_path):
    path = tmp_path / "job.lock"
    first, second = osutil.FileLock(path), osutil.FileLock(str(path))
    assert not first.locked
    assert first.acquire(blocking=False) is True
    assert first.locked and path.is_file()
    assert first.acquire(blocking=False) is True        # already held: nothing new is taken
    assert second.acquire(blocking=False) is False      # the lock belongs to the handle, also in one process
    assert not second.locked
    first.release()
    assert not first.locked
    assert second.acquire(blocking=False) is True
    second.release()
    second.release()                                    # releasing twice is harmless
    assert path.is_file() and path.read_bytes() == b""  # kept, and nothing was written to it


def test_file_lock_as_a_context_manager_waits_for_the_holder(tmp_path):
    path = tmp_path / "job.lock"
    holder = osutil.FileLock(path)
    assert holder.acquire()
    released_at = []

    def release_later():
        time.sleep(0.4)
        released_at.append(time.monotonic())
        holder.release()

    thread = threading.Thread(target=release_later)
    thread.start()
    try:
        with osutil.FileLock(path) as lock:             # blocks until the holder lets go
            entered = time.monotonic()
            assert lock.locked
        assert not lock.locked
    finally:
        thread.join()
    assert released_at and entered >= released_at[0]


def test_file_lock_reports_a_lock_file_that_cannot_be_opened(tmp_path):
    lock = osutil.FileLock(tmp_path / "no-such-directory" / "job.lock")
    with pytest.raises(OSError):
        lock.acquire(blocking=False)
    assert not lock.locked
    lock.release()                                      # nothing to release, no error


_HOLD_LOCK = """
import sys, time
from backend.app import osutil
lock = osutil.FileLock(sys.argv[1])
assert lock.acquire(blocking=False)
open(sys.argv[2], "w", encoding="utf-8").close()
time.sleep(60)
"""


def test_file_lock_is_held_across_processes_and_released_when_the_holder_dies(tmp_path):
    path, ready = tmp_path / "job.lock", tmp_path / "ready"
    holder = subprocess.Popen([sys.executable, "-c", _HOLD_LOCK, str(path), str(ready)], cwd=REPO)
    try:
        assert _wait_until(ready.exists), "the other process never took the lock"
        mine = osutil.FileLock(path)
        assert mine.acquire(blocking=False) is False
        holder.kill()
        holder.wait(timeout=15)
        # No stale lock: the system released it with the process.
        assert _wait_until(lambda: mine.acquire(blocking=False))
        mine.release()
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=15)


class _FakeMsvcrt:
    """msvcrt.locking as Windows does it: one holder per file, EACCES for the others."""

    LK_UNLCK, LK_LOCK, LK_NBLCK = 0, 1, 2

    def __init__(self) -> None:
        self.held: dict[int, int] = {}      # inode -> the descriptor that locked it
        self.calls: list[tuple[int, int, int]] = []
        self.fail_with: OSError | None = None

    def locking(self, fd: int, mode: int, nbytes: int) -> None:
        assert nbytes == 1 and os.lseek(fd, 0, os.SEEK_CUR) == 0     # always the first byte
        self.calls.append((fd, mode, nbytes))
        if self.fail_with is not None:
            raise self.fail_with
        inode = os.fstat(fd).st_ino
        if mode == self.LK_UNLCK:
            assert self.held.pop(inode) == fd
        elif inode in self.held:
            raise PermissionError(errno.EACCES, "Permission denied")
        else:
            self.held[inode] = fd


@pytest.fixture
def fake_msvcrt(monkeypatch, as_windows):
    fake = _FakeMsvcrt()
    monkeypatch.setattr(osutil, "msvcrt", fake)
    monkeypatch.setattr(osutil, "fcntl", None)          # as on Windows
    return fake


def test_file_lock_windows_branch_locks_one_byte_without_blocking(tmp_path, fake_msvcrt):
    path = tmp_path / "job.lock"
    first, second = osutil.FileLock(path), osutil.FileLock(path)
    assert first.acquire(blocking=False) is True
    assert second.acquire(blocking=False) is False and not second.locked
    assert [mode for _fd, mode, _n in fake_msvcrt.calls] == [fake_msvcrt.LK_NBLCK, fake_msvcrt.LK_NBLCK]
    first.release()
    assert fake_msvcrt.calls[-1][1] == fake_msvcrt.LK_UNLCK and not fake_msvcrt.held
    assert second.acquire(blocking=False) is True
    second.release()
    assert path.read_bytes() == b""


def test_file_lock_windows_branch_polls_to_wait(tmp_path, fake_msvcrt, monkeypatch):
    """msvcrt has no call that waits (LK_LOCK gives up after ten seconds): acquire() polls."""
    path = tmp_path / "job.lock"
    holder, waiter = osutil.FileLock(path), osutil.FileLock(path)
    assert holder.acquire()
    naps = []

    def nap(seconds):
        naps.append(seconds)
        if len(naps) == 3:
            holder.release()

    monkeypatch.setattr(osutil.time, "sleep", nap)
    assert waiter.acquire() is True
    assert naps == [osutil.FileLock.POLL_S] * 3
    assert all(mode != fake_msvcrt.LK_LOCK for _fd, mode, _n in fake_msvcrt.calls)
    waiter.release()


def test_file_lock_windows_branch_reports_other_errors_and_leaks_no_handle(tmp_path, fake_msvcrt, monkeypatch):
    lock = osutil.FileLock(tmp_path / "job.lock")
    opened, closed = [], []
    real_open, real_close = os.open, os.close

    def tracking_open(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        if Path(path) == lock.path:                     # not what another thread may open meanwhile
            opened.append(fd)
        return fd

    def tracking_close(fd):
        if fd in opened:
            closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(osutil.os, "open", tracking_open)
    monkeypatch.setattr(osutil.os, "close", tracking_close)
    fake_msvcrt.fail_with = OSError(errno.EINVAL, "Invalid argument")   # e.g. a share without locks
    with pytest.raises(OSError):
        lock.acquire()
    assert not lock.locked and len(opened) == 1 and closed == opened
    # ... and a failing unlock does not keep release() from closing the handle
    fake_msvcrt.fail_with = None
    assert lock.acquire(blocking=False)
    fake_msvcrt.fail_with = OSError(errno.EACCES, "Permission denied")
    lock.release()
    assert not lock.locked and len(opened) == 2 and closed == opened


def test_file_lock_without_any_lock_module_is_an_oserror(tmp_path, monkeypatch):
    # The POSIX branch opens the lock file with calls Windows does not have: only there.
    for windows in ((True,) if osutil.IS_WINDOWS else (False, True)):
        monkeypatch.setattr(osutil, "IS_WINDOWS", windows)
        monkeypatch.setattr(osutil, "fcntl", None)
        monkeypatch.setattr(osutil, "msvcrt", None)
        lock = osutil.FileLock(tmp_path / "job.lock")
        with pytest.raises(OSError):
            lock.acquire(blocking=False)
        assert not lock.locked


# ---------------------------------------------------------------------------
# popen_group_kwargs / kill_process_tree
# ---------------------------------------------------------------------------

def test_popen_group_kwargs_per_system(monkeypatch):
    monkeypatch.setattr(osutil, "IS_WINDOWS", False)
    assert osutil.popen_group_kwargs() == {"start_new_session": True}
    monkeypatch.setattr(osutil, "IS_WINDOWS", True)
    assert osutil.popen_group_kwargs() == {"creationflags": 0x00000200}     # CREATE_NEW_PROCESS_GROUP
    if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        assert osutil.popen_group_kwargs()["creationflags"] == subprocess.CREATE_NEW_PROCESS_GROUP


# A process with a child of its own. The child holds a file lock for as long as it lives
# (the system releases it when the process ends): that is how the test sees whether it is
# gone. Its process id would not do: Windows hands the id of a dead process to the next one.
_PARENT_WITH_CHILD = """
import subprocess, sys, time
from backend.app import osutil
child = subprocess.Popen([sys.executable, "-c",
                          "import sys, time; from backend.app import osutil; "
                          "lock = osutil.FileLock(sys.argv[1]); lock.acquire(); time.sleep(60)", sys.argv[1]])
probe = osutil.FileLock(sys.argv[1])
while probe.acquire(blocking=False):    # free: the child has not got that far yet
    probe.release()
    time.sleep(0.02)
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    handle.write(str(child.pid))
time.sleep(60)
"""


# The interpreter itself. In a virtual environment on Windows sys.executable is a launcher
# that starts it as a child process, at once: too early for a job object put around the
# launcher after its start (see osutil.adopt_child). The scripts above need no package.
_INTERPRETER = getattr(sys, "_base_executable", None) or sys.executable


@pytest.mark.parametrize("adopted", [False, True])
def test_kill_process_tree_kills_a_group_leader_with_its_children(tmp_path, adopted):
    """Both ways on Windows: the tree taskkill finds for a process started elsewhere, and the
    job object of a child given to adopt_child(). On POSIX the process group either way."""
    child_lock, ready = tmp_path / "child.lock", tmp_path / "ready"
    parent = subprocess.Popen([_INTERPRETER if adopted else sys.executable, "-c", _PARENT_WITH_CHILD,
                               str(child_lock), str(ready)], cwd=REPO, **osutil.popen_group_kwargs())
    if adopted:
        assert osutil.adopt_child(parent.pid) is osutil.IS_WINDOWS
    probe = osutil.FileLock(child_lock)
    try:
        assert _wait_until(lambda: ready.exists() and ready.read_text(encoding="utf-8").strip(), timeout=30)
        assert probe.acquire(blocking=False) is False       # the child is running
        assert osutil.process_exists(parent.pid)
        started = time.monotonic()
        osutil.kill_process_tree(parent.pid)
        parent.wait(timeout=15)                             # the caller reaps
        assert time.monotonic() - started < 15
        assert _wait_until(lambda: probe.acquire(blocking=False)), "the child of the killed process is still running"
        osutil.kill_process_tree(parent.pid)                # gone already: not an error
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=15)
        osutil.release_child(parent.pid)                    # not adopted: nothing to release
        if not probe.locked and ready.exists() and not probe.acquire(blocking=False):
            osutil.kill_process_tree(int(ready.read_text(encoding="utf-8")))    # do not leave it behind
        probe.release()
    assert parent.pid not in osutil._child_jobs


def test_an_adopted_child_does_not_outlive_its_release(tmp_path):
    """Windows ends what is in a job object when the job's last handle is closed: release_child()
    after the child was waited for, or the end of this process, whatever ends it. That is how a
    JVM goes with a server that was ended in the Task Manager. POSIX has nothing like it: there
    the process group is killed, by kill_process_tree() alone."""
    child_lock, ready = tmp_path / "child.lock", tmp_path / "ready"
    parent = subprocess.Popen([_INTERPRETER, "-c", _PARENT_WITH_CHILD, str(child_lock), str(ready)],
                              cwd=REPO, **osutil.popen_group_kwargs())
    assert osutil.adopt_child(parent.pid) is osutil.IS_WINDOWS
    probe = osutil.FileLock(child_lock)
    try:
        assert _wait_until(lambda: ready.exists() and ready.read_text(encoding="utf-8").strip(), timeout=30)
        assert probe.acquire(blocking=False) is False       # the child is running
        osutil.release_child(parent.pid)
        if osutil.IS_WINDOWS:
            parent.wait(timeout=15)
            assert _wait_until(lambda: probe.acquire(blocking=False)), "the child outlived the job object"
        else:
            time.sleep(0.2)
            assert parent.poll() is None and probe.acquire(blocking=False) is False
    finally:
        osutil.kill_process_tree(parent.pid)
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=15)
        if not probe.locked and ready.exists() and not probe.acquire(blocking=False):
            osutil.kill_process_tree(int(ready.read_text(encoding="utf-8")))    # do not leave it behind
        probe.release()


def test_kill_process_tree_of_a_child_in_our_own_group_kills_only_that_child():
    """Not started as a group leader: the group it is in is this test run's own."""
    child = subprocess.Popen(SLEEPER)
    try:
        osutil.kill_process_tree(child.pid)
        child.wait(timeout=15)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=15)
    assert child.returncode != 0                        # ... and we are still here to check it


def test_kill_process_tree_never_raises_and_never_aims_at_whole_groups(monkeypatch):
    killed = []
    monkeypatch.setattr(osutil.os, "kill", lambda pid, sig: killed.append(("kill", pid)))
    if hasattr(os, "killpg"):
        monkeypatch.setattr(osutil.os, "killpg", lambda pid, sig: killed.append(("killpg", pid)))
    monkeypatch.setattr(osutil.subprocess, "run", lambda *a, **k: killed.append(("taskkill", a)))
    for windows in (False, True):
        monkeypatch.setattr(osutil, "IS_WINDOWS", windows)
        for bad in (0, -1, -12345, os.getpid(), "not a pid", None):
            osutil.kill_process_tree(bad)               # 0 / -1 would mean "every process of the group / the user"
    assert killed == []


@pytest.mark.skipif(not hasattr(os, "killpg"),
                    reason="process groups are a POSIX concept; Windows kills by process tree")
def test_kill_process_tree_posix_branch_picks_group_or_process(monkeypatch):
    monkeypatch.setattr(osutil, "IS_WINDOWS", False)
    sent = []
    monkeypatch.setattr(osutil.os, "kill", lambda pid, sig: sent.append(("kill", pid, sig)))
    monkeypatch.setattr(osutil.os, "killpg", lambda pid, sig: sent.append(("killpg", pid, sig)))
    monkeypatch.setattr(osutil.os, "getpgrp", lambda: 500)

    monkeypatch.setattr(osutil.os, "getpgid", lambda pid: pid)          # leads its own group
    osutil.kill_process_tree(4242)
    monkeypatch.setattr(osutil.os, "getpgid", lambda pid: 500)          # a member of our group
    osutil.kill_process_tree(4243)
    monkeypatch.setattr(osutil.os, "getpgid", lambda pid: pid)          # ... and never our group itself
    osutil.kill_process_tree(500)

    def exited(pid):
        raise ProcessLookupError(errno.ESRCH, "No such process")

    monkeypatch.setattr(osutil.os, "getpgid", exited)                   # leader gone: what is left of its group
    osutil.kill_process_tree(4244)
    assert sent == [("killpg", 4242, signal.SIGKILL), ("kill", 4243, signal.SIGKILL),
                    ("kill", 500, signal.SIGKILL), ("killpg", 4244, signal.SIGKILL)]

    def refuse(pid, sig):
        raise PermissionError(errno.EPERM, "Operation not permitted")   # macOS: a group of zombies only

    monkeypatch.setattr(osutil.os, "killpg", refuse)
    osutil.kill_process_tree(4245)                                      # swallowed


def test_kill_process_tree_windows_branch_runs_taskkill_on_the_tree(monkeypatch, as_windows):
    runs, terminated = [], []
    outcome = {"returncode": 0, "error": None}

    def fake_run(command, **kwargs):
        runs.append((command, kwargs))
        if outcome["error"] is not None:
            raise outcome["error"]
        return types.SimpleNamespace(returncode=outcome["returncode"])

    monkeypatch.setattr(osutil.subprocess, "run", fake_run)
    monkeypatch.setattr(osutil.os, "kill", lambda pid, sig: terminated.append((pid, sig)))
    monkeypatch.setenv("SystemRoot", r"C:\Windows")

    osutil.kill_process_tree(4242)
    command, kwargs = runs[-1]
    assert command[1:] == ["/PID", "4242", "/T", "/F"]                  # /T: with its children, /F: at once
    assert "taskkill" in command[0].lower()
    assert kwargs["creationflags"] == 0x08000000                        # CREATE_NO_WINDOW: no console flashes up
    assert kwargs["stdin"] == kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL
    assert kwargs["timeout"] > 0
    assert terminated == []

    outcome["returncode"] = 128                                         # "process not found": nothing left to do
    osutil.kill_process_tree(4243)
    assert terminated == []

    outcome["returncode"] = 1                                           # taskkill failed: the process itself at least
    osutil.kill_process_tree(4244)
    assert terminated == [(4244, signal.SIGTERM)]                       # os.kill is TerminateProcess on Windows

    for error in (FileNotFoundError(errno.ENOENT, "taskkill"), subprocess.TimeoutExpired("taskkill", 15)):
        outcome["error"] = error
        osutil.kill_process_tree(4245)
    assert terminated[1:] == [(4245, signal.SIGTERM)] * 2

    def gone(pid, sig):
        raise OSError(errno.EINVAL, "The parameter is incorrect")      # no such process

    monkeypatch.setattr(osutil.os, "kill", gone)
    osutil.kill_process_tree(4246)                                      # swallowed


class _FakeJobs:
    """Job objects as Windows keeps them: a job knows its processes; terminating it or closing
    its last handle ends them."""

    def __init__(self) -> None:
        self.jobs: dict[int, list[int]] = {}        # open handle -> the process ids assigned to it
        self.ended: list[int] = []                  # process ids ended through their job
        self.fail: dict[str, OSError] = {}          # call name -> the error it raises
        self.calls: list[tuple] = []

    def _call(self, name: str, *args) -> None:
        self.calls.append((name, *args))
        if name in self.fail:
            raise self.fail[name]

    def create(self) -> int:
        self._call("create")
        handle = 1000 + len([call for call in self.calls if call[0] == "create"])
        self.jobs[handle] = []
        return handle

    def assign(self, job: int, pid: int) -> None:
        self._call("assign", job, pid)
        self.jobs[job].append(pid)

    def terminate(self, job: int) -> None:
        self._call("terminate", job)
        self.ended += self.jobs[job]

    def close(self, job: int) -> None:
        self._call("close", job)
        self.ended += self.jobs.pop(job)            # kill on close


@pytest.fixture
def fake_jobs(monkeypatch, as_windows):
    fake = _FakeJobs()
    monkeypatch.setattr(osutil, "_job_api", lambda: fake)
    monkeypatch.setattr(osutil, "_child_jobs", {})
    taskkill = []
    monkeypatch.setattr(osutil.subprocess, "run",
                        lambda command, **kwargs: taskkill.append(command) or types.SimpleNamespace(returncode=0))
    monkeypatch.setattr(osutil.os, "kill", lambda pid, sig: pytest.fail("os.kill terminates on Windows"))
    fake.taskkill = taskkill
    return fake


def test_adopt_child_is_nothing_on_posix(monkeypatch):
    monkeypatch.setattr(osutil, "IS_WINDOWS", False)
    monkeypatch.setattr(osutil, "_job_api", lambda: pytest.fail("no job objects on POSIX"))
    assert osutil.adopt_child(4242) is False
    osutil.release_child(4242)
    assert osutil._child_jobs == {}


def test_kill_process_tree_windows_branch_ends_the_job_of_an_adopted_child(fake_jobs):
    """taskkill /T takes for the children of a process whatever names its id as the parent, and
    Windows hands the id of a parent that has exited to a new process. The job object of an
    adopted child holds the processes that child started, and nothing else."""
    assert osutil.adopt_child(4242) is True
    job = osutil._child_jobs[4242]
    assert fake_jobs.calls == [("create",), ("assign", job, 4242)]

    osutil.kill_process_tree(4242)
    assert fake_jobs.calls[-1] == ("terminate", job) and fake_jobs.ended == [4242]
    assert fake_jobs.taskkill == []                                     # no walk of the process list
    osutil.kill_process_tree(4242)                                      # again (the parser's atexit hook): harmless
    assert fake_jobs.taskkill == []

    osutil.release_child(4242)
    assert fake_jobs.calls[-1] == ("close", job) and fake_jobs.jobs == {} and osutil._child_jobs == {}
    osutil.release_child(4242)                                          # twice, or never adopted: nothing
    assert fake_jobs.calls[-1] == ("close", job)

    osutil.kill_process_tree(4243)                                      # a process started elsewhere
    assert [command[1:] for command in fake_jobs.taskkill] == [["/PID", "4243", "/T", "/F"]]


def test_adopt_child_windows_branch_without_job_objects_leaves_taskkill(fake_jobs, caplog):
    refused = OSError(errno.EACCES, "Access is denied")
    fake_jobs.fail["create"] = refused
    assert osutil.adopt_child(4242) is False and osutil._child_jobs == {}
    del fake_jobs.fail["create"]
    fake_jobs.fail["assign"] = refused                                  # e.g. the process has exited already
    assert osutil.adopt_child(4242) is False and osutil._child_jobs == {}
    assert fake_jobs.jobs == {}                                         # the job made for it is not left open
    del fake_jobs.fail["assign"]
    osutil.kill_process_tree(4242)
    assert len(fake_jobs.taskkill) == 1 and not any(call[0] == "terminate" for call in fake_jobs.calls)

    assert osutil.adopt_child(4242) is True                             # terminating fails: taskkill after all
    fake_jobs.fail["terminate"] = refused
    osutil.kill_process_tree(4242)
    assert len(fake_jobs.taskkill) == 2
    fake_jobs.fail["close"] = refused
    osutil.release_child(4242)                                          # never raises
    assert osutil._child_jobs == {}
    for bad in ("not a pid", None):
        assert osutil.adopt_child(bad) is False
        osutil.release_child(bad)


def test_adopt_child_windows_branch_replaces_the_job_of_an_earlier_process_with_that_id(fake_jobs):
    """A child that was never released, and its process id given to the next one by Windows:
    the kill must reach the new process, not an empty job."""
    assert osutil.adopt_child(4242) and osutil.adopt_child(4242)
    first, second = [call[1] for call in fake_jobs.calls if call[0] == "assign"]
    assert osutil._child_jobs == {4242: second} and list(fake_jobs.jobs) == [second]
    osutil.kill_process_tree(4242)
    assert fake_jobs.calls[-1] == ("terminate", second)


def test_job_object_structure_has_the_win32_layout():
    """JOBOBJECT_EXTENDED_LIMIT_INFORMATION, checked here because the structure is the same on
    every 64-bit (and every 32-bit) system; SetInformationJobObject refuses any other size."""
    import ctypes

    limits_type = osutil._job_limits_type()
    wide = ctypes.sizeof(ctypes.c_void_p) == 8
    if not wide and os.name != "nt":
        return      # 32-bit POSIX aligns 64-bit integers its own way, and never passes the structure on
    assert ctypes.sizeof(limits_type) == (144 if wide else 112)
    limits = limits_type()
    basic = type(limits.BasicLimitInformation)
    assert basic.LimitFlags.offset == 16 and basic.LimitFlags.size == 4
    assert (limits_type.IoInfo.offset, limits_type.IoInfo.size) == ((64, 48) if wide else (48, 48))
    limits.BasicLimitInformation.LimitFlags = osutil._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    assert bytes(limits)[16:20] == (0x2000).to_bytes(4, sys.byteorder)
    assert osutil._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION == 9


class _FakeKernel32:
    """kernel32 as osutil._JobApi sees it: functions that take a return and argument types and
    are called; every call is recorded and answered from ``results``."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.results = {"CreateJobObjectW": 0x1_0000_0A10, "SetInformationJobObject": 1,
                        "AssignProcessToJobObject": 1, "TerminateJobObject": 1,
                        "OpenProcess": 0x1_0000_0B20, "CloseHandle": 1}
        self.functions: dict[str, types.SimpleNamespace] = {}

    def __getattr__(self, name: str):
        if name not in self.results:
            raise AttributeError(name)                  # as a DLL without that export
        if name not in self.functions:
            def call(*args, _name=name):
                self.calls.append((_name, *args))
                return self.results[_name]

            function = types.SimpleNamespace(restype="int", argtypes=None)
            function.call = call
            self.functions[name] = function
        return _Callable(self.functions[name])


class _Callable:
    """A foreign function: callable, and its ``restype`` / ``argtypes`` can be set."""

    def __init__(self, function: types.SimpleNamespace) -> None:
        object.__setattr__(self, "_function", function)

    def __call__(self, *args):
        return self._function.call(*args)

    def __getattr__(self, name: str):
        return getattr(self._function, name)

    def __setattr__(self, name: str, value) -> None:
        setattr(self._function, name, value)


def test_job_api_calls_kernel32_the_way_win32_documents_it(monkeypatch):
    """The binding itself, with a stand-in for the DLL: which functions, which arguments, which
    types. A handle is as wide as a pointer; ctypes' default return type, a C int, would cut a
    64-bit one in half and every later call would name another object."""
    import ctypes
    from ctypes import wintypes

    dll, loaded = _FakeKernel32(), []
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, **kwargs: loaded.append((name, kwargs)) or dll, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
    monkeypatch.setattr(ctypes, "WinError",
                        lambda code=None: PermissionError(errno.EACCES, f"Windows error {code}"), raising=False)
    api = osutil._JobApi()
    assert loaded == [("kernel32", {"use_last_error": True})]
    assert {name: function.restype for name, function in dll.functions.items()} == {
        "CreateJobObjectW": wintypes.HANDLE, "OpenProcess": wintypes.HANDLE,
        "SetInformationJobObject": wintypes.BOOL, "AssignProcessToJobObject": wintypes.BOOL,
        "TerminateJobObject": wintypes.BOOL, "CloseHandle": wintypes.BOOL,
    }
    assert all(function.argtypes is not None for function in dll.functions.values())
    assert dll.functions["AssignProcessToJobObject"].argtypes == (wintypes.HANDLE, wintypes.HANDLE)
    assert ctypes.sizeof(wintypes.HANDLE) == ctypes.sizeof(ctypes.c_void_p)

    job = api.create()
    assert job == 0x1_0000_0A10
    created, limited = dll.calls
    assert created == ("CreateJobObjectW", None, None)                  # unnamed, default security: not inherited
    name, handle, info_class, limits, size = limited
    assert (name, handle, info_class) == ("SetInformationJobObject", job, 9)
    assert size == ctypes.sizeof(osutil._job_limits_type())
    assert limits._obj.BasicLimitInformation.LimitFlags == 0x2000       # kill on job close, and nothing else
    assert limits._obj.BasicLimitInformation.ActiveProcessLimit == 0

    del dll.calls[:]
    api.assign(job, 4242)
    assert dll.calls == [("OpenProcess", 0x0101, False, 4242),          # PROCESS_SET_QUOTA | PROCESS_TERMINATE
                         ("AssignProcessToJobObject", job, 0x1_0000_0B20),
                         ("CloseHandle", 0x1_0000_0B20)]
    del dll.calls[:]
    api.terminate(job)
    api.close(job)
    assert dll.calls == [("TerminateJobObject", job, 1), ("CloseHandle", job)]

    # Every refusal is an OSError, and no handle is left open on the way out.
    del dll.calls[:]
    dll.results["AssignProcessToJobObject"] = 0
    with pytest.raises(OSError, match="Windows error 5"):
        api.assign(job, 4242)
    assert dll.calls[-1] == ("CloseHandle", 0x1_0000_0B20)
    del dll.calls[:]
    dll.results["OpenProcess"] = None                                   # the process is gone
    with pytest.raises(OSError):
        api.assign(job, 4242)
    assert [call[0] for call in dll.calls] == ["OpenProcess"]
    dll.results["TerminateJobObject"] = 0
    with pytest.raises(OSError):
        api.terminate(job)
    del dll.calls[:]
    dll.results["SetInformationJobObject"] = 0
    with pytest.raises(OSError):
        api.create()
    assert dll.calls[-1] == ("CloseHandle", 0x1_0000_0A10)
    dll.results["CreateJobObjectW"] = None
    with pytest.raises(OSError):
        api.create()


def test_popen_group_kwargs_loads_the_job_binding_before_the_child_starts(monkeypatch, as_windows):
    loaded = []
    monkeypatch.setattr(osutil, "_job_api", lambda: loaded.append(1))
    assert osutil.popen_group_kwargs() == {"creationflags": 0x00000200}
    assert loaded == [1]
    monkeypatch.setattr(osutil, "_job_api", lambda: (_ for _ in ()).throw(OSError("no kernel32")))
    assert osutil.popen_group_kwargs() == {"creationflags": 0x00000200}     # adopt_child() says what is wrong


# ---------------------------------------------------------------------------
# process_exists
# ---------------------------------------------------------------------------

def test_process_exists_for_real_processes():
    assert osutil.process_exists(os.getpid()) is True
    assert osutil.process_exists(0) is False and osutil.process_exists(-1) is False
    child = subprocess.Popen(SLEEPER)
    try:
        assert osutil.process_exists(child.pid) is True     # asking must not hurt it ...
        time.sleep(0.2)
        assert child.poll() is None                         # ... as os.kill(pid, 0) would on Windows
    finally:
        child.kill()
        child.wait(timeout=15)
    assert osutil.process_exists(child.pid) is False        # reaped: the id is free


def test_process_exists_windows_branch(monkeypatch, as_windows):
    state = {"open_error": None, "exit_code": 259, "closed": []}

    def open_process(access, inherit, pid):
        assert access == 0x1000 and inherit is False        # PROCESS_QUERY_LIMITED_INFORMATION
        if state["open_error"] is not None:
            raise state["open_error"]
        return 77

    fake = types.SimpleNamespace(
        OpenProcess=open_process,
        GetExitCodeProcess=lambda handle: state["exit_code"],
        CloseHandle=state["closed"].append,
        STILL_ACTIVE=259,
    )
    monkeypatch.setitem(sys.modules, "_winapi", fake)
    killed = []
    monkeypatch.setattr(osutil.os, "kill", lambda pid, sig: killed.append(pid))

    assert osutil.process_exists(4242) is True
    state["exit_code"] = 1                                  # ended; someone still holds a handle to it
    assert osutil.process_exists(4242) is False
    assert state["closed"] == [77, 77]                      # the handle is given back each time
    state["open_error"] = PermissionError(errno.EACCES, "Access is denied")
    assert osutil.process_exists(4) is True                 # a system process exists, we just may not look
    state["open_error"] = OSError(errno.EINVAL, "The parameter is incorrect")
    assert osutil.process_exists(4242) is False
    assert killed == []                                     # never os.kill on Windows: it terminates


# ---------------------------------------------------------------------------
# harden_stdio
# ---------------------------------------------------------------------------

def _stream(encoding: str, errors: str = "strict") -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors=errors, newline="\n")


def test_harden_stdio_makes_a_code_page_stream_unable_to_fail(monkeypatch):
    out, err = _stream("cp932"), _stream("cp1252", errors="backslashreplace")
    with pytest.raises(UnicodeEncodeError):
        _stream("cp932").write("张三 picked 斧王\n")          # PYTHONIOENCODING=cp932 gives strict
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    osutil.harden_stdio()
    print("张三 picked 斧王")
    out.flush()
    written = out.buffer.getvalue()
    assert written == "\\u5f20三 picked 斧王\n".encode("cp932")   # what the code page has stays readable
    assert out.encoding == "cp932"                               # the encoding itself is not changed
    assert err.errors == "backslashreplace"


def test_harden_stdio_covers_what_windows_gives_redirected_output(monkeypatch):
    """Python on Windows opens a pipe or file with the code page and surrogateescape."""
    out = _stream("cp932", errors="surrogateescape")
    with pytest.raises(UnicodeEncodeError):
        _stream("cp932", errors="surrogateescape").write("张三\n")   # surrogateescape fails here too
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    osutil.harden_stdio()
    print("张三")
    out.flush()
    assert out.buffer.getvalue() == "\\u5f20三\n".encode("cp932")
    assert out.errors == "backslashreplace"


def test_harden_stdio_leaves_utf8_and_replaced_streams_alone(monkeypatch):
    out = _stream("utf-8")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", io.StringIO())       # a capture object without reconfigure()
    osutil.harden_stdio()
    assert out.errors == "strict"
    monkeypatch.setattr(sys, "stdout", None)                # pythonw.exe: no console at all
    osutil.harden_stdio()
