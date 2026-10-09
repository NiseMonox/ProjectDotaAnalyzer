"""Disk storage for uploaded replays, parsed matches and AI coach reports.

All functions here are blocking (file I/O, hashing, gzip, JSON) and are
meant to be called through ``asyncio.to_thread`` from request handlers.

Layout (under the project root)::

    replays/<sha256[:32]>.dem          uploaded replays, content-hash named
    replays/.incoming/<uuid>.part      uploads being written
    cache/matches/<safe_key>.json.gz   MatchData + metadata per match key
    cache/coaching/<id>_v<N>.json      AI coach reports (N = report schema)
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable

from .models import MatchData

logger = logging.getLogger(__name__)

# Source 2 demo files start with this 8-byte magic.
REPLAY_MAGIC = b"PBDEMS2\x00"
BZIP2_MAGIC = b"BZh"
UPLOAD_HASH_LEN = 32  # hex chars of sha256 kept in names (128 bits)
_UPLOAD_NAME_RE = re.compile(r"^[0-9a-f]{%d}\.dem$" % UPLOAD_HASH_LEN)
_COPY_CHUNK = 1024 * 1024
_STALE_PART_AGE_S = 6 * 3600


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


def store_upload(src: BinaryIO, replay_dir: Path, max_bytes: int) -> StoredReplay:
    """Copy an uploaded replay stream to ``replay_dir`` under its content hash.

    Validates the Source 2 demo magic, enforces ``max_bytes`` and hashes
    while copying. Re-uploading identical bytes returns the existing file.
    """
    try:
        src.seek(0)
    except (AttributeError, OSError):
        pass
    head = src.read(len(REPLAY_MAGIC)) or b""
    if head.startswith(BZIP2_MAGIC):
        raise UploadRejected("replay_compressed")
    if head != REPLAY_MAGIC:
        raise UploadRejected("replay_invalid")

    incoming = replay_dir / ".incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    tmp = incoming / f"{uuid.uuid4().hex}.part"
    hasher = hashlib.sha256()
    size = 0
    try:
        with open(tmp, "wb") as out:
            chunk = head
            while chunk:
                size += len(chunk)
                if size > max_bytes:
                    raise UploadRejected("upload_too_large", 413)
                hasher.update(chunk)
                out.write(chunk)
                chunk = src.read(_COPY_CHUNK)
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
    entries always reflect the current analyzer. Bump :attr:`FORMAT` when
    the stored ``MatchData`` semantics change incompatibly (old files are
    then ignored and eventually pruned).
    """

    FORMAT = 1
    SUFFIX = ".json.gz"

    def __init__(self, directory: Path, max_files: Callable[[], int]) -> None:
        self.directory = directory
        self._max_files = max_files

    def path_for(self, key: str) -> Path:
        return self.directory / f"{safe_key(key)}{self.SUFFIX}"

    def save(self, key: str, match: MatchData, meta: dict) -> None:
        header = json.dumps(
            {"format": self.FORMAT, "key": key, "saved_at": int(time.time()), "meta": meta},
            ensure_ascii=False,
        )
        # Splice pydantic's fast JSON for the (large) match into the envelope.
        payload = header[:-1] + ',"match":' + match.model_dump_json() + "}"
        atomic_write_bytes(self.path_for(key), gzip.compress(payload.encode("utf-8"), compresslevel=6))
        self.prune()

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
                logger.info("Dropping stored match %s (format %s)", path.name, data.get("format"))
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
