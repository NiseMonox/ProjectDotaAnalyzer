"""Disk storage for uploaded replays, parsed matches and AI coach reports.

All functions here are blocking (file I/O, hashing, gzip, JSON) and are
meant to be called through ``asyncio.to_thread`` from request handlers.

Layout (under the project root)::

    replays/<sha256[:32]>.dem          uploaded replays, content-hash named
                                       (always the unpacked demo file)
    replays/.incoming/<uuid>.part      uploads being written
    cache/matches/<safe_key>.json.gz   MatchData + metadata per match key
    cache/coaching/<id>_d<F>_v<N>.json AI coach reports (F = match store
                                       format, N = report schema)
"""
from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import logging
import os
import re
import threading
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterator

from .models import MatchData

logger = logging.getLogger(__name__)

# Source 2 demo files start with this 8-byte magic.
REPLAY_MAGIC = b"PBDEMS2\x00"
# Replays are distributed compressed. Valve's replay servers name the file
# ``<match>_<salt>.dem.bz2`` whatever it holds: bzip2 in the past, Zstandard
# today. The content decides how an upload is read, never its name.
BZIP2_MAGIC = b"BZh"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
# File-name endings accepted for an upload (lower case).
UPLOAD_SUFFIXES = (".dem", ".dem.bz2", ".dem.zst")
UPLOAD_HASH_LEN = 32  # hex chars of sha256 kept in names (128 bits)
_UPLOAD_NAME_RE = re.compile(r"^[0-9a-f]{%d}\.dem$" % UPLOAD_HASH_LEN)
_COPY_CHUNK = 1024 * 1024
_STALE_PART_AGE_S = 6 * 3600
# Zstandard input is fed to the decoder in pieces of this size. The format
# packs at most 128 KiB into 4 bytes, so one piece unpacks to 4 MiB at most.
# store_upload() holds two unpacked pieces at a time (the one it is writing
# and the next one coming out of the decoder): about 8 MiB per upload is all
# a decompression bomb can make the server hold, however much it claims to
# unpack to. (Pieces of 1024 bytes unpack a 111 MB replay in 0.14 s instead
# of 0.22 s, but hand out up to 32 MiB at once.)
_ZSTD_FEED = 128
# Longest Zstandard frame header: enough bytes to read a frame's window from.
_ZSTD_HEADER_MAX = 18
# Largest Zstandard window (decoder memory) accepted. Valve's files use
# 4 MiB and `zstd -19` uses 8 MiB; only `--ultra` / `--long` need more.
ZSTD_MAX_WINDOW = 32 * 1024 * 1024


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def safe_key(key: str, max_len: int = 80) -> str:
    """Map an arbitrary cache key to a safe file-name stem.

    Only ``[A-Za-z0-9_-]`` survive; when anything had to change a short
    hash of the original key is appended so distinct keys never collide.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", key)[:max_len]
    if cleaned != key or not cleaned:
        digest = hashlib.sha1(key.encode("utf-8", "surrogatepass")).hexdigest()[:12]
        cleaned = f"{cleaned}-{digest}" if cleaned else digest
    return cleaned


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp file + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def prune_files(files: list[Path], max_files: int, keep: set[Path] | None = None) -> list[Path]:
    """Delete the oldest (by mtime) of ``files`` beyond ``max_files``."""
    if len(files) <= max_files:
        return []
    keep = keep or set()

    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    removed: list[Path] = []
    excess = len(files) - max_files
    for path in sorted(files, key=_mtime):
        if excess <= 0:
            break
        if path in keep:
            continue
        try:
            path.unlink()
            removed.append(path)
            excess -= 1
        except FileNotFoundError:
            excess -= 1
        except OSError as exc:
            logger.warning("Failed to prune %s: %s", path, exc)
    return removed


def touch(path: Path) -> None:
    """Bump mtime (LRU marker); ignore races with pruning."""
    try:
        os.utime(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Uploaded replays
# ---------------------------------------------------------------------------

class UploadRejected(Exception):
    """Upload refused; ``code`` is a security.MESSAGES key."""

    def __init__(self, code: str, status: int = 400) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


@dataclass
class StoredReplay:
    path: Path
    digest: str
    size: int
    existed: bool  # an identical replay was already stored


def _zstd_module():
    """The optional ``zstandard`` package, or None when it is not installed."""
    try:
        import zstandard
    except ImportError:
        return None
    return zstandard


def _bz2_module():
    """The standard library's ``bz2``, or None on a Python built without libbz2
    (imported here so that such a build can still run the server)."""
    try:
        import bz2
    except ImportError:
        return None
    return bz2


def _bzip2_chunks(src: BinaryIO, bz2) -> Iterator[bytes]:
    """Unpacked content of a bzip2 file (several streams in a row included).

    A file that ends inside a stream raises EOFError, invalid data OSError.
    """
    reader = bz2.BZ2File(src)
    return iter(lambda: reader.read(_COPY_CHUNK), b"")


def _zstd_window(zstd, header: bytes) -> int:
    """Window size the Zstandard frame starting with ``header`` asks the
    decoder to allocate; 0 when ``header`` is not a complete frame header
    (the decoder reports what is wrong with it)."""
    try:
        return int(zstd.get_frame_parameters(header).window_size)
    except zstd.ZstdError:
        return 0


def _zstd_chunks(src: BinaryIO, zstd) -> Iterator[bytes]:
    """Unpacked content of a Zstandard file (several frames in a row included).

    Fed piece by piece instead of through the package's stream reader: that
    one ends silently when the input stops inside a frame, so an interrupted
    download would be analyzed as a short replay. Here it raises EOFError;
    invalid data raises ``zstd.ZstdError``. A frame that needs a larger
    window than ``ZSTD_MAX_WINDOW`` is an intact file this server does not
    unpack: refused with ``replay_window_too_large`` before anything is
    allocated for it.
    """
    decoder = None
    while True:
        data = src.read(_ZSTD_FEED)
        if not data:
            break
        while data:
            if decoder is None:
                if len(data) < _ZSTD_HEADER_MAX:  # a frame that starts at the end of a piece
                    data += src.read(_ZSTD_HEADER_MAX - len(data))
                window = _zstd_window(zstd, data)
                if window > ZSTD_MAX_WINDOW:
                    logger.info("Rejected a Zstandard upload that needs a %d MiB window", window >> 20)
                    raise UploadRejected("replay_window_too_large")
                decoder = zstd.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW).decompressobj()
            chunk = decoder.decompress(data)
            if chunk:
                yield chunk
            data = b""
            if decoder.eof:  # frame complete; whatever follows is the next one
                data, decoder = decoder.unused_data, None
    if decoder is not None:
        raise EOFError("Zstandard data ends inside a frame")


def _unpacked(chunks: Iterator[bytes], errors: tuple[type[BaseException], ...]) -> Iterator[bytes]:
    """``chunks`` with the decoder's own errors turned into an upload refusal."""
    try:
        yield from chunks
    except errors as exc:
        logger.info("Rejected a compressed upload that cannot be unpacked: %s", exc)
        raise UploadRejected("replay_decompress_failed") from None


def _replay_chunks(src: BinaryIO, head: bytes) -> Iterator[bytes]:
    """The replay in ``src`` piece by piece, unpacked when it is compressed.

    ``head`` is what was already read from ``src`` to tell the format.
    bzip2 needs only the standard library; Zstandard needs the optional
    ``zstandard`` package and is refused with ``replay_compressed`` (the
    user has to unpack the file) without it, as bzip2 is with
    ``replay_compressed_bzip2`` on a Python built without it.
    """
    if head.startswith(BZIP2_MAGIC):
        bz2 = _bz2_module()
        if bz2 is None:
            raise UploadRejected("replay_compressed_bzip2")
        src.seek(0)
        return _unpacked(_bzip2_chunks(src, bz2), (OSError, EOFError))
    if head.startswith(ZSTD_MAGIC):
        zstd = _zstd_module()
        if zstd is None:
            raise UploadRejected("replay_compressed")
        src.seek(0)
        return _unpacked(_zstd_chunks(src, zstd), (zstd.ZstdError, EOFError))
    return itertools.chain((head,), iter(lambda: src.read(_COPY_CHUNK), b""))


def store_upload(src: BinaryIO, replay_dir: Path, max_bytes: int) -> StoredReplay:
    """Copy an uploaded replay stream to ``replay_dir`` under its content hash.

    A bzip2 or Zstandard compressed upload is unpacked on the way (see
    ``_replay_chunks``), so the stored file, its hash and ``max_bytes``
    always refer to the demo file itself: the same replay gets the same
    name however it was uploaded, and a small archive cannot unpack into
    more than the upload limit. Validates the Source 2 demo magic and
    hashes while copying. Re-uploading identical bytes returns the
    existing file.
    """
    try:
        src.seek(0)
    except (AttributeError, OSError):
        pass
    chunks = _replay_chunks(src, src.read(len(REPLAY_MAGIC)) or b"")
    start = b""
    for chunk in chunks:  # a decoder may hand out the first bytes in several pieces
        start += chunk
        if len(start) >= len(REPLAY_MAGIC):
            break
    if not start.startswith(REPLAY_MAGIC):
        raise UploadRejected("replay_invalid")

    incoming = replay_dir / ".incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    tmp = incoming / f"{uuid.uuid4().hex}.part"
    hasher = hashlib.sha256()
    size = 0
    try:
        with open(tmp, "wb") as out:
            # Not itertools.chain((start,), chunks): that keeps the first
            # piece (megabytes of a compressed upload) alive for the whole copy.
            chunk, start = start, None
            while chunk is not None:
                size += len(chunk)
                if size > max_bytes:
                    raise UploadRejected("upload_too_large", 413)
                hasher.update(chunk)
                out.write(chunk)
                chunk = next(chunks, None)
        digest = hasher.hexdigest()[:UPLOAD_HASH_LEN]
        final = replay_dir / f"{digest}.dem"
        existed = final.exists()
        if existed:
            tmp.unlink(missing_ok=True)
            touch(final)
        else:
            os.replace(tmp, final)
        return StoredReplay(path=final, digest=digest, size=size, existed=existed)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def is_uploaded_replay(path: Path) -> bool:
    """Whether ``path`` is a content-hash named upload (safe to prune)."""
    return bool(_UPLOAD_NAME_RE.match(path.name))


def prune_replays(replay_dir: Path, max_files: int, keep: set[Path] | None = None) -> list[Path]:
    """Keep at most ``max_files`` uploaded replays (oldest deleted first).

    Only content-hash named uploads are touched; manually placed replays
    (e.g. the demo replay or parse-local files) are never deleted. Also
    clears stale partial uploads left by crashes.
    """
    incoming = replay_dir / ".incoming"
    if incoming.is_dir():
        cutoff = time.time() - _STALE_PART_AGE_S
        for part in incoming.glob("*.part"):
            try:
                if part.stat().st_mtime < cutoff:
                    part.unlink()
            except OSError:
                pass
    try:
        uploads = [p for p in replay_dir.iterdir() if p.is_file() and is_uploaded_replay(p)]
    except OSError as exc:
        logger.warning("Cannot list replay dir %s: %s", replay_dir, exc)
        return []
    removed = prune_files(uploads, max_files, keep)
    if removed:
        logger.info("Pruned %d old uploaded replay(s)", len(removed))
    return removed


def delete_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Failed to delete %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Persistent match store
# ---------------------------------------------------------------------------

class MatchStore:
    """Bounded on-disk store of parsed matches (gzip JSON per match key).

    Each file holds the ``MatchData`` plus route metadata (parse source,
    extra response fields). The analysis is recomputed on load, so stored
    entries always reflect the current analyzer, but the ``MatchData``
    itself is whatever the parser produced back then.

    :attr:`FORMAT` is the schema version of that stored data. Bump it with
    every change to what a parser puts into ``MatchData`` (new or
    re-defined fields, fixed values, key formats): files written with
    another version are deleted on load and the match is parsed or fetched
    again. AI coach reports are cached per ``FORMAT`` as well.

    History: 1 = backend wave 1; 2 = kill log, position log, scoreboard
    values, canonical item/hero keys, ward ``dewarded`` flag (wave 2);
    3 = talent names without template tokens ("+{s:...} Gush Damage");
    4 = game update to 7.41f: the analyzer reads replays of current game
    builds to the end (the previous parser library gave up early on some,
    and such partial matches were stored), Roshan's Banner is no longer a
    Roshan kill, item / ability names and keys come from the regenerated
    catalogs;
    5 = the replay analyzer equals OpenDota: hero healing and gold / XP per
    minute from the post-game record, team scores without deaths to
    neutrals, tower kills with denied towers, kill-log assists, the minute
    series (gold / XP / last hits / denies and the advantage curves) sampled
    like OpenDota's, teamfight chaining on its rounded clock,
    ``KillEvent.killer_team`` (which side a fountain or creep kill belongs
    to), skill builds under the 7.40 talent rules (also the estimated skill
    times of OpenDota imports). Every AI coach report cached before it was
    also written by the coach of an earlier round;
    6 = OpenDota imports: the team scores of a match the API answers 0 : 0
    for (matches of 2013) are the kills of each side's players, and with
    them the kill participation; estimated skill times never lie after the
    end of the match. The AI coach reports cached before it had CS
    benchmark sentences that disagreed with their own benchmark column,
    empty phase cards for scoreboard-only matches and the root cause of a
    death repeated in the analysis text.
    ``MatchData.talent_names_patch`` was added without a bump: its default
    ("" = unknown) is what every reader assumes for older files, and a bump
    would also discard the cached AI coach reports.
    """

    FORMAT = 6
    SUFFIX = ".json.gz"

    def __init__(self, directory: Path, max_files: Callable[[], int]) -> None:
        self.directory = directory
        self._max_files = max_files
        # Serializes writers: update_meta() rewrites a file it has just read
        # and must not put an older match back over a concurrent save().
        self._write_lock = threading.Lock()

    def path_for(self, key: str) -> Path:
        return self.directory / f"{safe_key(key)}{self.SUFFIX}"

    def save(self, key: str, match: MatchData, meta: dict) -> None:
        header = json.dumps(
            {"format": self.FORMAT, "key": key, "saved_at": int(time.time()), "meta": meta},
            ensure_ascii=False,
        )
        # Splice pydantic's fast JSON for the (large) match into the envelope.
        payload = header[:-1] + ',"match":' + match.model_dump_json() + "}"
        data = gzip.compress(payload.encode("utf-8"), compresslevel=6)
        with self._write_lock:
            atomic_write_bytes(self.path_for(key), data)
        self.prune()

    def update_meta(self, key: str, changes: dict) -> bool:
        """Merge ``changes`` into the stored metadata of ``key``.

        The match itself is left as stored. Returns False when there is no
        current-format entry for the key (nothing is written then).
        """
        path = self.path_for(key)
        with self._write_lock:
            try:
                data = json.loads(gzip.decompress(path.read_bytes()))
            except FileNotFoundError:
                return False
            except (OSError, EOFError, ValueError, zlib.error) as exc:  # unreadable: load() discards it
                logger.warning("Cannot update stored match %s: %s", path, exc)
                return False
            if not isinstance(data, dict) or data.get("format") != self.FORMAT or data.get("key") != key:
                return False
            meta = data.get("meta")
            data["meta"] = {**(meta if isinstance(meta, dict) else {}), **changes}
            payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            atomic_write_bytes(path, gzip.compress(payload.encode("utf-8"), compresslevel=6))
        return True

    def load(self, key: str) -> tuple[MatchData, dict] | None:
        path = self.path_for(key)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            logger.warning("Cannot read stored match %s: %s", path, exc)
            return None
        try:
            data = json.loads(gzip.decompress(raw))
            if data.get("format") != self.FORMAT:
                logger.info(
                    "Dropping stored match %s (format %s, current %s)", path.name, data.get("format"), self.FORMAT,
                )
                delete_file(path)
                return None
            if data.get("key") != key:
                return None
            match = MatchData.model_validate(data["match"])
        except Exception as exc:  # corrupted / incompatible file
            logger.warning("Discarding unreadable stored match %s: %s", path, exc)
            delete_file(path)
            return None
        touch(path)
        return match, dict(data.get("meta") or {})

    def delete(self, key: str) -> None:
        delete_file(self.path_for(key))

    def prune(self) -> list[Path]:
        try:
            files = [p for p in self.directory.glob(f"*{self.SUFFIX}") if p.is_file()]
        except OSError:
            return []
        return prune_files(files, self._max_files())


# ---------------------------------------------------------------------------
# AI coach report cache
# ---------------------------------------------------------------------------

def coach_cache_path(cache_dir: Path, coach_id: str, schema_version: int) -> Path:
    """Disk path of a cached coach report; the name embeds the schema version."""
    return cache_dir / f"{safe_key(coach_id, max_len=120)}_v{int(schema_version)}.json"


def load_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable cache file %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        return None
    touch(path)
    return data


def save_json(path: Path, data: dict, max_files: int | None = None) -> None:
    atomic_write_bytes(path, json.dumps(data, ensure_ascii=False).encode("utf-8"))
    if max_files is not None:
        prune_files([p for p in path.parent.glob("*.json") if p.is_file()], max_files, keep={path})
