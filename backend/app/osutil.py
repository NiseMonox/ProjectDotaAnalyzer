"""Operating-system primitives that differ between POSIX and Windows.

The one place where the application asks which system it runs on. Each
helper keeps the POSIX behaviour the project was written against (the
production server is Linux) and adds the Windows counterpart:

- :func:`atomic_replace`, :func:`read_bytes` and :func:`remove_file`:
  ``os.replace``, reading a whole file and ``unlink``; on Windows they wait a
  moment when the file is in use elsewhere.
- :class:`FileLock`: an exclusive lock on a lock file, across processes
  (``fcntl.flock`` / ``msvcrt.locking``).
- :func:`popen_group_kwargs` and :func:`kill_process_tree`: start a child as
  the leader of its own process group, and kill it with its children.
- :func:`adopt_child` and :func:`release_child`: on Windows, a job object
  around a child, so that the kill ends exactly the processes it started
  and the child never outlives this process.
- :func:`process_exists`: whether a process id is in use, without touching
  the process.
- :func:`harden_stdio`: ``print()`` never fails on a character the console's
  code page lacks.

Imports nothing from this project, so every module may use it.

What Windows does differently (the reasons for this module):

- A file that is open cannot be replaced or deleted: ``os.replace`` onto it
  and ``os.unlink`` raise ``PermissionError``. Python opens files without the
  sharing mode that would allow it, and so do virus scanners and the search
  indexer, which look at a new file for a moment. Some of those share
  nothing at all: while they look, the file cannot even be opened for reading.
- There is no ``fcntl``, no process group to signal and no ``SIGKILL``:
  ``os.kill(pid, sig)`` is ``TerminateProcess`` for every ``sig`` except the
  two console events. That includes signal 0: the POSIX idiom for "does this
  process exist" kills the process there.
- A process keeps the id of its parent for life, also after the parent has
  exited and its id was given to a new process: who is whose child cannot be
  read from the process list (what ``taskkill /T`` does). A job object knows
  its processes for certain.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]
try:
    import msvcrt
except ImportError:  # POSIX
    msvcrt = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

IS_WINDOWS: bool = os.name == "nt"

# subprocess only defines these on Windows; the values are part of the
# Win32 API (process creation flags).
_CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
# OpenProcess access right that is granted for nearly every process.
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
# taskkill exit codes that mean there is nothing left to do: killed / no such process.
_TASKKILL_DONE = (0, 128)
_TASKKILL_TIMEOUT_S = 15.0
# Job objects (Win32 API): the limit flag that ends every process of a job when
# its last handle is closed, the information class the flag is set through,
# and the access rights AssignProcessToJobObject asks of a process handle.
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def atomic_replace(src: str | Path, dst: str | Path, *, retries: int = 20, delay: float = 0.05) -> None:
    """``os.replace(src, dst)`` that copes with a destination in use on Windows.

    On Windows the replace fails with ``PermissionError`` while another
    thread or process has ``dst`` open (a reader of the same cache file, a
    virus scanner looking at the file just written). That lasts
    milliseconds, so the call is repeated up to ``retries`` times,
    ``delay`` seconds apart (one second with the defaults), before the
    error is raised. ``src`` is left in place then; removing it is the
    caller's clean-up, as with ``os.replace``.

    On POSIX a replace never fails for that reason: there this is exactly
    ``os.replace``, and a ``PermissionError`` (a directory that may not be
    written) is raised at once.
    """
    _retry_while_in_use(lambda: os.replace(src, dst), retries, delay)


def read_bytes(path: str | Path, *, retries: int = 20, delay: float = 0.05) -> bytes:
    """The content of ``path``; waits on Windows while another handle keeps the file to itself.

    For files that other threads or processes replace at any moment (the
    caches). On Windows a replace in progress, or a program that opens the
    file without sharing it (a virus scanner, a backup), shuts a reader out
    for an instant and the read fails with ``PermissionError``: it is
    repeated like :func:`atomic_replace` does. A file that is not there
    raises ``FileNotFoundError`` at once, on every system; on POSIX this is
    ``Path.read_bytes()``.
    """
    def read() -> bytes:
        with open(path, "rb") as handle:
            return handle.read()

    return _retry_while_in_use(read, retries, delay)


def remove_file(path: str | Path, *, retries: int = 20, delay: float = 0.05) -> None:
    """Delete ``path`` (no error when it is not there); waits on Windows while it is in use.

    Windows does not delete a file that is open: ``unlink`` fails with
    ``PermissionError`` while a reader, or a virus scanner looking at a
    file that was just written, has it. Repeated like
    :func:`atomic_replace`; the error is raised when the file is still in
    use after that. On POSIX this is ``Path.unlink(missing_ok=True)``.
    """
    _retry_while_in_use(lambda: Path(path).unlink(missing_ok=True), retries, delay)


def _retry_while_in_use(action, retries: int, delay: float):
    """``action()``; on Windows a ``PermissionError`` is retried ``retries`` times, ``delay`` s apart."""
    attempts = max(0, retries) if IS_WINDOWS else 0
    while True:
        try:
            return action()
        except PermissionError:
            if attempts <= 0:
                raise
            attempts -= 1
            time.sleep(delay)


class FileLock:
    """Exclusive advisory lock on a lock file, held across processes.

    ::

        lock = FileLock(directory / "_learn.lock")
        if not lock.acquire(blocking=False):
            return          # another process holds it
        try:
            ...
        finally:
            lock.release()

        with FileLock(path):    # waits for the lock
            ...

    The operating system releases the lock when its holder dies, so it
    cannot go stale. The lock file is created when missing and never
    deleted (deleting it would race with the next holder); nothing is
    written to it.

    ``fcntl.flock`` on POSIX: the file is opened read-only, so a lock file
    that belongs to another user (a cron run as root next to the service)
    can still be locked. ``msvcrt.locking`` on the first byte on Windows.
    Either way the lock belongs to this object's file handle: a second
    ``FileLock`` on the same path waits for the first one, also inside one
    process.

    One object is for one owner: it is not re-entrant (``acquire`` on a
    lock that is already held returns True and takes nothing new) and not
    meant to be shared between threads.
    """

    #: Seconds between attempts of a blocking ``acquire`` on Windows, where
    #: the system offers no call that waits.
    POLL_S = 0.1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    @property
    def locked(self) -> bool:
        """Whether this object holds the lock."""
        return self._fd is not None

    def acquire(self, blocking: bool = True) -> bool:
        """Take the lock. False when ``blocking`` is false and someone else holds it.

        Raises ``OSError`` when the lock file cannot be opened or the file
        system (or this Python) has no file locks: the caller decides
        whether to go on unguarded.
        """
        if self._fd is not None:
            return True
        fd = self._open()
        try:
            taken = self._lock_windows(fd, blocking) if IS_WINDOWS else self._lock_posix(fd, blocking)
        except BaseException:
            os.close(fd)
            raise
        if not taken:
            os.close(fd)
            return False
        self._fd = fd
        return True

    def release(self) -> None:
        """Give the lock back. Does nothing when it is not held; never raises."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            if IS_WINDOWS and msvcrt is not None:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass  # closing the handle releases the lock too
        finally:
            with suppress(OSError):
                os.close(fd)  # POSIX: closing releases the flock

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    def _open(self) -> int:
        if IS_WINDOWS:
            try:
                return os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
            except PermissionError:  # exists with the read-only attribute: read access can lock too
                return os.open(self.path, os.O_RDONLY)
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_CREAT, 0o644)
        except PermissionError:  # exists, but this user may not create files here
            return os.open(self.path, os.O_RDONLY)
        with suppress(OSError):  # someone else's file: left as it is
            os.fchmod(fd, 0o644)  # ours: readable for the other user whatever the umask
        return fd

    @staticmethod
    def _lock_posix(fd: int, blocking: bool) -> bool:
        if fcntl is None:
            raise OSError("file locks are not available on this system")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def _lock_windows(self, fd: int, blocking: bool) -> bool:
        if msvcrt is None:
            raise OSError("file locks are not available on this system")
        while True:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                # LK_NBLCK, also to wait: LK_LOCK gives up after ten seconds.
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return True
            except PermissionError:  # EACCES: the byte is locked by another handle
                if not blocking:
                    return False
                time.sleep(self.POLL_S)


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------

def popen_group_kwargs() -> dict:
    """Keyword arguments that make a child process the leader of its own group.

    For ``subprocess.Popen`` / ``subprocess.run`` and
    ``asyncio.create_subprocess_exec``. Such a child can be killed together
    with its children (:func:`kill_process_tree`), and it does not get the
    Ctrl-C that stops its parent: the parent has to kill it when it goes.
    ``start_new_session`` on POSIX, ``CREATE_NEW_PROCESS_GROUP`` on Windows.

    On Windows a process group is only about console events; what ties the
    child's own children to it is :func:`adopt_child`, called right after
    the start. Its Win32 binding is loaded here already, so that call takes
    microseconds also the first time.
    """
    if IS_WINDOWS:
        with suppress(Exception):  # adopt_child() reports what is wrong
            with _child_jobs_lock:
                _job_api()
        return {"creationflags": _CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def adopt_child(pid: int) -> bool:
    """Tie the child process ``pid``, and all it starts from now on, to this process. Never raises.

    Call it once, right after starting the child, and :func:`release_child`
    when the child has been waited for. Returns whether the child is tied.

    Windows: the child is put into a job object of its own that only this
    process holds. :func:`kill_process_tree` then ends the processes of
    that job, which are known for certain, instead of asking ``taskkill``
    for the tree: that one is built from the parent id every process
    carries, and a process whose parent has exited keeps that id while
    Windows gives it to a new process, so a tree can hold strangers. And
    Windows ends the job's processes itself when this process ends, however
    it ends (the Task Manager included): nothing can stay behind.

    A process the child had started before this call is not in the job (a
    launcher that starts the real program at once can be that fast); a JVM
    starts none. When the job cannot be set up, the child is left as it is
    and :func:`kill_process_tree` falls back to ``taskkill``.

    POSIX: nothing to do, False. The session of :func:`popen_group_kwargs`
    is what :func:`kill_process_tree` signals there.
    """
    if not IS_WINDOWS:
        return False
    try:
        pid = int(pid)
        with _child_jobs_lock:
            api = _job_api()
            job = api.create()
            try:
                api.assign(job, pid)
            except BaseException:
                api.close(job)
                raise
            stale = _child_jobs.pop(pid, None)  # an earlier process of that id, never released
            _child_jobs[pid] = job
            if stale is not None:
                api.close(stale)
        return True
    except Exception as exc:  # no job objects here, or the process is already gone
        logger.debug("Process %s was not put into a job object: %s", pid, exc)
        return False


def release_child(pid: int) -> None:
    """Forget a child given to :func:`adopt_child`. Never raises; nothing to do on POSIX.

    For a child that has exited. Closing its job ends whatever the child
    left running, so nothing it started outlives it.
    """
    try:
        with _child_jobs_lock:
            job = _child_jobs.pop(int(pid), None)
            if job is not None:
                _job_api().close(job)
    except Exception as exc:
        logger.debug("Could not close the job object of process %s: %s", pid, exc)


# Job handles of the adopted children, by process id (Windows only). The lock
# is held for every use of a handle: a handle closed by one thread must not be
# used by another, to whom Windows may have given its number for a new object.
_child_jobs: dict[int, int] = {}
_child_jobs_lock = threading.Lock()
_job_api_instance = None


def _job_api():
    """The job-object calls, bound on first use (call with ``_child_jobs_lock`` held)."""
    global _job_api_instance
    if _job_api_instance is None:
        _job_api_instance = _JobApi()
    return _job_api_instance


def _job_limits_type():
    """``JOBOBJECT_EXTENDED_LIMIT_INFORMATION`` as a ctypes structure.

    In fixed-width types, so the layout is the Win32 one wherever this is
    imported (``ctypes.wintypes.DWORD`` is 8 bytes wide on Linux and macOS).
    """
    import ctypes

    dword, size_t = ctypes.c_uint32, ctypes.c_size_t  # DWORD; SIZE_T and ULONG_PTR

    class BasicLimits(ctypes.Structure):  # JOBOBJECT_BASIC_LIMIT_INFORMATION
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", dword),
            ("MinimumWorkingSetSize", size_t),
            ("MaximumWorkingSetSize", size_t),
            ("ActiveProcessLimit", dword),
            ("Affinity", size_t),
            ("PriorityClass", dword),
            ("SchedulingClass", dword),
        ]

    class IoCounters(ctypes.Structure):  # IO_COUNTERS
        _fields_ = [(name, ctypes.c_uint64) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimits),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", size_t),
            ("JobMemoryLimit", size_t),
            ("PeakProcessMemoryUsed", size_t),
            ("PeakJobMemoryUsed", size_t),
        ]

    return ExtendedLimits


class _JobApi:
    """Job objects through ``kernel32`` (the standard library has no binding for them).

    Windows only. A failed call raises ``OSError`` with the Windows error.
    """

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        self._limits_type = _job_limits_type()
        # An instance of our own: the argument types set here are nobody else's.
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, restype, argtypes in (
            ("CreateJobObjectW", wintypes.HANDLE, (wintypes.LPVOID, wintypes.LPCWSTR)),
            ("SetInformationJobObject", wintypes.BOOL,
             (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)),
            ("AssignProcessToJobObject", wintypes.BOOL, (wintypes.HANDLE, wintypes.HANDLE)),
            ("TerminateJobObject", wintypes.BOOL, (wintypes.HANDLE, wintypes.UINT)),
            ("OpenProcess", wintypes.HANDLE, (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)),
            ("CloseHandle", wintypes.BOOL, (wintypes.HANDLE,)),
        ):
            function = getattr(k32, name)
            function.restype, function.argtypes = restype, argtypes
        self._k32 = k32

    def _error(self) -> OSError:
        return self._ctypes.WinError(self._ctypes.get_last_error())

    def create(self) -> int:
        """Handle of a new job whose processes are ended when the handle is closed."""
        job = self._k32.CreateJobObjectW(None, None)  # no name; the handle is not inherited
        if not job:
            raise self._error()
        limits = self._limits_type()
        limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self._k32.SetInformationJobObject(
                job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                self._ctypes.byref(limits), self._ctypes.sizeof(limits)):
            error = self._error()
            self._k32.CloseHandle(job)
            raise error
        return job

    def assign(self, job: int, pid: int) -> None:
        """Put the process ``pid`` into ``job``; the processes it starts afterwards follow it."""
        process = self._k32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
        if not process:
            raise self._error()
        try:
            if not self._k32.AssignProcessToJobObject(job, process):
                raise self._error()
        finally:
            self._k32.CloseHandle(process)

    def terminate(self, job: int) -> None:
        """End every process of ``job``, at once."""
        if not self._k32.TerminateJobObject(job, 1):
            raise self._error()

    def close(self, job: int) -> None:
        self._k32.CloseHandle(job)


def kill_process_tree(pid: int) -> None:
    """Kill the process ``pid`` and its children, at once. Never raises.

    POSIX: ``SIGKILL`` for the process group when ``pid`` leads one (a child
    started with :func:`popen_group_kwargs`), so its children die with it;
    for the process alone otherwise. A group is also signalled when its
    leader has already exited: what is left of it should go. The caller's
    own group is never signalled.

    Windows: the job of a child given to :func:`adopt_child` is terminated,
    which ends exactly the processes that child started. Any other process:
    ``taskkill /T /F``, which takes for its children every process that
    names it as its parent (see :func:`adopt_child` for what that can
    include); when that cannot run, ``TerminateProcess`` for the process
    alone.

    The process is not reaped here: the caller still waits for it
    (``Popen.wait()`` / ``await proc.wait()``). A process that is already
    gone is not an error. As with every kill by process id, the caller has
    to know the id is still its child's (not waited for yet).
    """
    try:
        pid = int(pid)
        if pid <= 0 or pid == os.getpid():  # 0 and -1 would address whole groups of processes
            return
        if IS_WINDOWS:
            _kill_tree_windows(pid)
        else:
            _kill_tree_posix(pid)
    except Exception as exc:  # never the caller's problem: it is cleaning up
        logger.debug("Could not kill process %s: %s", pid, exc)


def _kill_tree_posix(pid: int) -> None:
    try:
        leads_group = os.getpgid(pid) == pid
    except ProcessLookupError:
        # Exited (on macOS also: exited and not reaped yet). Members of a
        # group it led may be left; without such a group killpg finds nothing.
        leads_group = True
    except OSError:
        leads_group = False
    try:
        if leads_group and pid != os.getpgrp():
            os.killpg(pid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGKILL)
    except OSError:  # already gone, or not ours any more
        pass


def _taskkill_command() -> str:
    """Full path of taskkill.exe (not looked up on PATH: the system's own copy)."""
    root = os.environ.get("SystemRoot") or os.environ.get("windir") or r"C:\Windows"
    candidate = os.path.join(root, "System32", "taskkill.exe")
    return candidate if os.path.isfile(candidate) else "taskkill"


def _terminate_job(pid: int) -> bool:
    """End the processes in the job of the adopted child ``pid``; False when it has none."""
    with _child_jobs_lock:
        job = _child_jobs.get(pid)
        if job is None:
            return False
        try:
            _job_api().terminate(job)
        except OSError as exc:
            logger.debug("Could not terminate the job object of process %s: %s", pid, exc)
            return False
        return True


def _kill_tree_windows(pid: int) -> None:
    if _terminate_job(pid):
        return
    try:
        done = subprocess.run(
            [_taskkill_command(), "/PID", str(pid), "/T", "/F"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=_TASKKILL_TIMEOUT_S, check=False, creationflags=_CREATE_NO_WINDOW,
        )
        if done.returncode in _TASKKILL_DONE:
            return
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("taskkill could not run for process %s: %s", pid, exc)
    # Every signal but the console events is TerminateProcess on Windows.
    with suppress(OSError):
        os.kill(pid, signal.SIGTERM)


def process_exists(pid: int) -> bool:
    """Whether a process with this id exists. Does not touch the process.

    Use this instead of ``os.kill(pid, 0)``, which terminates the process
    on Windows. On POSIX a process that has exited but was not reaped yet
    (a zombie) still exists; on Windows a process whose exit code is set
    does not, even while someone holds a handle to it.
    """
    pid = int(pid)
    if pid <= 0:
        return False
    if IS_WINDOWS:
        return _process_exists_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, belongs to another user
        return True
    except OSError:
        return False
    return True


def _process_exists_windows(pid: int) -> bool:
    import _winapi  # the standard library's own binding (subprocess, multiprocessing)

    try:
        handle = _winapi.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    except PermissionError:  # exists: a system process, or another user's
        return True
    except OSError:  # no process with this id
        return False
    try:
        return _winapi.GetExitCodeProcess(handle) == _winapi.STILL_ACTIVE
    except OSError:
        return True
    finally:
        _winapi.CloseHandle(handle)


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------

def harden_stdio() -> None:
    """Make ``sys.stdout`` unable to fail on characters its encoding lacks.

    This project prints and logs Chinese text. A Windows console window
    shows it whatever the code page, but output that is redirected to a
    file or a pipe is encoded with the system's ANSI code page (cp936,
    cp932, cp1252 ...), and ``print()`` raises ``UnicodeEncodeError`` on
    the first character that code page lacks. A stream that is not UTF-8
    and has an error handler that fails there gets ``backslashreplace``
    (what Python gives ``sys.stderr`` from the start), so such a character
    is written as ``\\uXXXX`` instead. Those handlers are ``strict`` and
    ``surrogateescape``: Python on Windows opens redirected output with the
    latter, which only passes lone surrogates through. UTF-8 streams, which
    is every stream on the Linux server and on macOS, are left alone, and
    so is the encoding itself: the output stays readable in the console it
    was made for.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = str(getattr(stream, "encoding", "") or "").lower().replace("-", "").replace("_", "")
        errors = getattr(stream, "errors", None)
        if encoding in ("utf8", "utf8sig") or errors not in ("strict", "surrogateescape"):
            continue
        reconfigure = getattr(stream, "reconfigure", None)  # not there on a replaced stream
        if reconfigure is None:
            continue
        with suppress(Exception):
            reconfigure(errors="backslashreplace")
