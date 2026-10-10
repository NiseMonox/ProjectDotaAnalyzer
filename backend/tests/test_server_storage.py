"""Unit tests for backend.app.storage (replay files, match store, caches)."""
import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import bz2
import gzip
import hashlib
import io
import json
import random
import struct
import time
import tracemalloc

import pytest

from backend.app import storage
from backend.app.models import MatchData

DEM = storage.REPLAY_MAGIC


def tiny_match(match_id: int = 5) -> MatchData:
    return MatchData(
        match_id=match_id, duration=60, radiant_win=False, game_mode="Turbo", start_time=0,
        radiant_score=0, dire_score=0, players=[], wards=[], smokes=[], teamfights=[],
        ability_events=[],
    )


@pytest.mark.parametrize("key", ["demo", "opendota_123", "replay_" + "a" * 32])
def test_safe_key_keeps_simple_keys(key):
    assert storage.safe_key(key) == key


@pytest.mark.parametrize("key", ["../etc/passwd", "..", "", "a/b", "local_x.dem", "x" * 200, "名字"])
def test_safe_key_sanitizes_and_disambiguates(key):
    safe = storage.safe_key(key)
    assert safe and all(ch.isalnum() or ch in "_-" for ch in safe)
    assert "/" not in safe and not safe.startswith(".")
    assert len(safe) <= 80 + 13
    assert storage.safe_key(key) == safe  # stable
    assert storage.safe_key(key + "!") != safe


def test_store_upload_hashes_and_dedupes(tmp_path):
    data = DEM + b"payload" * 1000
    stored = storage.store_upload(io.BytesIO(data), tmp_path, max_bytes=10**6)
    assert stored.digest == hashlib.sha256(data).hexdigest()[:32]
    assert stored.path == tmp_path / f"{stored.digest}.dem"
    assert stored.path.read_bytes() == data
    assert stored.size == len(data) and stored.existed is False

    again = storage.store_upload(io.BytesIO(data), tmp_path, max_bytes=10**6)
    assert again.existed is True and again.path == stored.path
    assert list((tmp_path / ".incoming").iterdir()) == []


@pytest.mark.parametrize("data,code,status", [
    (b"HL2DEMO\x00rest", "replay_invalid", 400),
    (b"BZh91AY", "replay_decompress_failed", 400),   # bzip2 header, nothing behind it
    (b"PBDEMS", "replay_invalid", 400),
    (b"", "replay_invalid", 400),
    (DEM + b"x" * 100, "upload_too_large", 413),
])
def test_store_upload_rejections_leave_no_files(tmp_path, data, code, status):
    with pytest.raises(storage.UploadRejected) as info:
        storage.store_upload(io.BytesIO(data), tmp_path, max_bytes=50)
    assert info.value.code == code and info.value.status == status
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


# ---------------------------------------------------------------------------
# Compressed uploads (bzip2: standard library; Zstandard: optional package)
# ---------------------------------------------------------------------------

def _pack(kind: str, data: bytes) -> bytes:
    if kind == "bzip2":
        return bz2.compress(data)
    zstandard = pytest.importorskip("zstandard")
    # With the content checksum (`zstd` writes one by default; Valve's files have none).
    return zstandard.ZstdCompressor(level=3, write_checksum=True).compress(data)


def _rejection(tmp_path, data: bytes, max_bytes: int = 10**6) -> storage.UploadRejected:
    with pytest.raises(storage.UploadRejected) as info:
        storage.store_upload(io.BytesIO(data), tmp_path, max_bytes=max_bytes)
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]   # nothing is left behind
    return info.value


PACKERS = ["bzip2", "zstd"]


@pytest.mark.parametrize("kind", PACKERS)
def test_store_upload_unpacks_compressed_replays(tmp_path, kind):
    """The stored file, its name and its size are those of the demo inside."""
    data = DEM + bytes(range(256)) * 2000        # 512 KB: several Zstandard feed pieces
    packed = _pack(kind, data)
    assert packed.startswith({"bzip2": storage.BZIP2_MAGIC, "zstd": storage.ZSTD_MAGIC}[kind])
    assert len(packed) < len(data) // 10

    stored = storage.store_upload(io.BytesIO(packed), tmp_path, max_bytes=len(data))
    assert stored.digest == hashlib.sha256(data).hexdigest()[:32]
    assert stored.path.read_bytes() == data
    assert stored.size == len(data) and stored.existed is False

    # The same replay uploaded as a plain .dem is the same file.
    plain = storage.store_upload(io.BytesIO(data), tmp_path, max_bytes=len(data))
    assert plain.existed is True and plain.path == stored.path
    assert list((tmp_path / ".incoming").iterdir()) == []


@pytest.mark.parametrize("kind", PACKERS)
def test_store_upload_reads_archives_made_of_several_streams(tmp_path, kind):
    """`cat a.bz2 b.bz2` / several Zstandard frames are one file to every decoder."""
    first, second = DEM + b"first half " * 3000, b"second half " * 3000
    stored = storage.store_upload(
        io.BytesIO(_pack(kind, first) + _pack(kind, second)), tmp_path, max_bytes=10**6)
    assert stored.path.read_bytes() == first + second


@pytest.mark.parametrize("kind", PACKERS)
def test_store_upload_refuses_damaged_archives(tmp_path, kind):
    """An interrupted download must not be analyzed as a short replay."""
    data = DEM + random.Random(7).randbytes(200_000)   # incompressible: the archive is as long
    packed = _pack(kind, data)
    middle = len(packed) // 2
    for broken in (
        packed[:middle],                                             # cut in the middle
        packed[:-1],                                                 # last byte missing
        packed[:middle] + bytes(64) + packed[middle + 64:],          # damaged inside
        packed[:4],                                                  # nothing but the magic
    ):
        assert _rejection(tmp_path, broken).code == "replay_decompress_failed"


@pytest.mark.parametrize("kind", PACKERS)
def test_store_upload_checks_what_is_inside_an_archive(tmp_path, kind):
    assert _rejection(tmp_path, _pack(kind, b"not a replay at all " * 100)).code == "replay_invalid"
    assert _rejection(tmp_path, _pack(kind, b"")).code == "replay_invalid"
    assert _rejection(tmp_path, _pack(kind, DEM[:6])).code == "replay_invalid"


@pytest.mark.parametrize("kind", PACKERS)
def test_store_upload_limit_applies_to_the_unpacked_size(tmp_path, kind):
    """A decompression bomb: a few hundred bytes that unpack to 8 MB."""
    bomb = _pack(kind, DEM + b"\0" * (8 * 1024 * 1024))
    assert len(bomb) < 2000
    error = _rejection(tmp_path, bomb, max_bytes=1024 * 1024)
    assert error.code == "upload_too_large" and error.status == 413
    # Exactly at the limit it still fits.
    stored = storage.store_upload(io.BytesIO(bomb), tmp_path, max_bytes=8 * 1024 * 1024 + len(DEM))
    assert stored.size == 8 * 1024 * 1024 + len(DEM)


def test_zstd_upload_without_the_package_is_refused_with_its_own_code(tmp_path, monkeypatch):
    """No decoder: the user is told to unpack the file, not that it is invalid."""
    monkeypatch.setattr(storage, "_zstd_module", lambda: None)
    assert _rejection(tmp_path, storage.ZSTD_MAGIC + b"\x00" * 64).code == "replay_compressed"
    # bzip2 never depends on it.
    data = DEM + b"payload" * 100
    assert storage.store_upload(io.BytesIO(bz2.compress(data)), tmp_path, 10**6).path.read_bytes() == data


def test_bzip2_upload_on_a_python_without_bz2_is_refused_with_its_own_code(tmp_path, monkeypatch):
    """A Python built without libbz2 still runs the server; only bzip2 uploads are turned away."""
    monkeypatch.setattr(storage, "_bz2_module", lambda: None)
    data = DEM + b"payload" * 100
    assert _rejection(tmp_path, bz2.compress(data)).code == "replay_compressed_bzip2"
    assert storage.store_upload(io.BytesIO(data), tmp_path, 10**6).path.read_bytes() == data
    zstandard = pytest.importorskip("zstandard")
    packed = zstandard.ZstdCompressor().compress(data)
    assert storage.store_upload(io.BytesIO(packed), tmp_path, 10**6).existed is True


def _zstd_frame(blocks: list[tuple[str, bytes, int]], window_log: int = 17) -> bytes:
    """A Zstandard frame written by hand (RFC 8878), no compressor involved.

    ``blocks``: ("raw", data, _) stores ``data`` as it is; ("rle", byte, n)
    unpacks to ``byte`` * n from 4 bytes, the densest block the format has
    (n <= 128 KiB). The header names a window of 2**window_log bytes.
    """
    out = [storage.ZSTD_MAGIC, b"\x00", bytes([(window_log - 10) << 3])]
    for index, (kind, payload, count) in enumerate(blocks):
        size, block_type = (len(payload), 0) if kind == "raw" else (count, 1)
        last = index == len(blocks) - 1
        out.append(struct.pack("<I", (size << 3) | (block_type << 1) | last)[:3] + payload)
    return b"".join(out)


def _zstd_bomb(total: int) -> bytes:
    """A replay header followed by ``total`` zero bytes at the format's best ratio (32768:1)."""
    block = 128 * 1024
    return _zstd_frame([("raw", DEM, 0)] + [("rle", b"\x00", block)] * (total // block))


def test_zstd_bomb_is_unpacked_in_small_pieces(tmp_path):
    """No matter what an archive unpacks to, the decoder hands out 4 MiB at a time."""
    zstandard = pytest.importorskip("zstandard")
    total = 48 * 1024 * 1024
    bomb = _zstd_bomb(total)
    assert len(bomb) < 2000
    pieces = [len(piece) for piece in storage._zstd_chunks(io.BytesIO(bomb), zstandard)]
    assert sum(pieces) == len(DEM) + total
    assert max(pieces) <= 4 * 1024 * 1024

    # ... and store_upload keeps two of them: the one it writes and the next one
    # (three if it held on to the first piece for the whole copy).
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        stored = storage.store_upload(io.BytesIO(bomb), tmp_path, max_bytes=total + len(DEM))
        peak = tracemalloc.get_traced_memory()[1] - before
    finally:
        if not was_tracing:
            tracemalloc.stop()
    assert stored.size == len(DEM) + total
    assert peak < 10 * 1024 * 1024, f"{peak / 2 ** 20:.1f} MiB held while unpacking"


def test_zstd_window_above_the_limit_is_refused_with_its_own_code(tmp_path, monkeypatch):
    """Such a file is intact (zstd --ultra / --long): not allocated for, and not called damaged."""
    zstandard = pytest.importorskip("zstandard")
    data = DEM + bytes(range(256)) * 4096       # 1 MiB
    params = zstandard.ZstdCompressionParameters.from_level(3, window_log=20)
    packed = zstandard.ZstdCompressor(compression_params=params).compress(data)
    assert zstandard.get_frame_parameters(packed).window_size == 2 ** 20
    assert storage.store_upload(io.BytesIO(packed), tmp_path, 10**7).size == len(data)

    # The real limit, with frames that only claim their window (49 bytes each).
    at_limit = _zstd_frame([("raw", DEM + b"x" * 32, 0)], window_log=25)       # 32 MiB: accepted
    assert storage.store_upload(io.BytesIO(at_limit), tmp_path, 10**6).size == 40
    (tmp_path / "other").mkdir()
    above = _zstd_frame([("raw", DEM + b"x" * 32, 0)], window_log=26)          # 64 MiB
    assert _rejection(tmp_path / "other", above).code == "replay_window_too_large"
    # A later frame is checked as well, also when its header is split between two reads.
    filler = storage._ZSTD_FEED - 3 - len(_zstd_frame([("raw", b"", 0)]))
    first = _zstd_frame([("raw", DEM + b"x" * (filler - len(DEM)), 0)])
    assert len(first) == storage._ZSTD_FEED - 3
    assert _rejection(tmp_path / "other", first + above).code == "replay_window_too_large"

    monkeypatch.setattr(storage, "ZSTD_MAX_WINDOW", 2 ** 19)
    assert _rejection(tmp_path / "other", packed, max_bytes=10**7).code == "replay_window_too_large"
    # The decoder enforces the same limit by itself (it never trusts the check above).
    monkeypatch.setattr(storage, "_zstd_window", lambda zstd, header: 0)
    assert _rejection(tmp_path / "other", packed, max_bytes=10**7).code == "replay_decompress_failed"


def test_prune_replays_only_touches_uploads(tmp_path):
    uploads = []
    for i in range(4):
        p = tmp_path / f"{i:032x}.dem"
        p.write_bytes(DEM)
        os.utime(p, (1000 + i, 1000 + i))
        uploads.append(p)
    manual = tmp_path / "8713267082.dem"
    manual.write_bytes(DEM)
    os.utime(manual, (1, 1))
    stale_part = tmp_path / ".incoming" / "old.part"
    stale_part.parent.mkdir()
    stale_part.write_bytes(b"x")
    os.utime(stale_part, (1, 1))

    removed = storage.prune_replays(tmp_path, max_files=2, keep={uploads[0]})
    assert sorted(removed) == sorted([uploads[1], uploads[2]])
    assert uploads[0].exists() and uploads[3].exists() and manual.exists()
    assert not stale_part.exists()


def test_match_store_roundtrip_prune_and_corruption(tmp_path):
    store = storage.MatchStore(tmp_path, max_files=lambda: 2)
    store.save("opendota_1", tiny_match(1), {"parse_source": "opendota", "extra": {"has_parsed_data": True}})
    loaded = store.load("opendota_1")
    assert loaded is not None
    match, meta = loaded
    assert match.match_id == 1
    assert meta == {"parse_source": "opendota", "extra": {"has_parsed_data": True}}
    assert store.load("missing") is None

    for i, key in enumerate(("k2", "k3")):
        time.sleep(0.01)
        store.save(key, tiny_match(i + 2), {})
    # Bounded: saving a third match pruned the least recently used one.
    assert sorted(p.name for p in tmp_path.glob("*.json.gz")) == ["k2.json.gz", "k3.json.gz"]
    os.utime(store.path_for("k2"), (1, 1))
    store.load("k2")  # a read refreshes recency
    store.save("k4", tiny_match(4), {})
    assert sorted(p.name for p in tmp_path.glob("*.json.gz")) == ["k2.json.gz", "k4.json.gz"]

    bad = store.path_for("broken")
    bad.write_bytes(b"not gzip")
    assert store.load("broken") is None
    assert not bad.exists()

    # Files from another store format are ignored.
    store.path_for("old").write_bytes(gzip.compress(b'{"format": 0, "key": "old", "match": {}}'))
    assert store.load("old") is None
    assert not store.path_for("old").exists()

    store.delete("k3")
    assert store.load("k3") is None


def test_match_store_drops_entries_written_before_the_current_format(tmp_path):
    """A stored MatchData is only as good as the parser that produced it."""
    store = storage.MatchStore(tmp_path, max_files=lambda: 10)
    store.save("replay_x", tiny_match(9), {"parse_source": "clarity", "extra": {}})
    path = store.path_for("replay_x")
    envelope = json.loads(gzip.decompress(path.read_bytes()))
    assert envelope["format"] == storage.MatchStore.FORMAT >= 5
    assert store.load("replay_x") is not None

    # The same, otherwise perfectly valid file as backend wave 1 wrote it, as the parser
    # before the 7.41f update did (which stopped early on some current replays), and as the
    # one before the kill log named the side of a fountain kill (score 3-29 instead of 4-29).
    for older in (1, 3, 4):
        envelope["format"] = older
        path.write_bytes(gzip.compress(json.dumps(envelope).encode()))
        assert store.load("replay_x") is None
        assert not path.exists()


def test_match_store_update_meta_keeps_the_match(tmp_path):
    store = storage.MatchStore(tmp_path, max_files=lambda: 10)
    match = tiny_match(7).model_copy(update={"radiant_gold_adv": [0, 150, -320], "game_mode": "全英雄选择"})
    store.save("opendota_7", match, {"parse_source": "opendota", "extra": {}, "fetched_at": 1234.5})

    assert store.update_meta("opendota_7", {"refetch": True}) is True
    loaded, meta = store.load("opendota_7")
    assert loaded == match
    assert meta == {"parse_source": "opendota", "extra": {}, "fetched_at": 1234.5, "refetch": True}
    # A later save replaces the metadata as a whole (the mark is gone).
    store.save("opendota_7", match, {"parse_source": "opendota", "extra": {}})
    assert "refetch" not in store.load("opendota_7")[1]

    # Nothing to update: unknown key, another store format, unreadable file.
    assert store.update_meta("missing", {"refetch": True}) is False
    assert not store.path_for("missing").exists()
    path = store.path_for("opendota_7")
    envelope = json.loads(gzip.decompress(path.read_bytes()))
    envelope["format"] = storage.MatchStore.FORMAT - 1
    old_bytes = gzip.compress(json.dumps(envelope).encode())
    path.write_bytes(old_bytes)
    assert store.update_meta("opendota_7", {"refetch": True}) is False
    assert path.read_bytes() == old_bytes
    path.write_bytes(b"not gzip")
    assert store.update_meta("opendota_7", {"refetch": True}) is False


def test_coach_cache_helpers(tmp_path):
    path = storage.coach_cache_path(tmp_path, "777_opendota-basic_0_zh", 2)
    assert path.name == "777_opendota-basic_0_zh_v2.json"
    assert storage.load_json(path) is None
    storage.save_json(path, {"a": "中文"}, max_files=1)
    assert storage.load_json(path) == {"a": "中文"}
    other = storage.coach_cache_path(tmp_path, "x", 2)
    os.utime(path, (1, 1))
    storage.save_json(other, {"b": 1}, max_files=1)
    assert not path.exists() and other.exists()
    other.write_text("[1, 2]")
    assert storage.load_json(other) is None
