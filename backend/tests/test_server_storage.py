"""Unit tests for backend.app.storage (replay files, match store, caches)."""
import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import gzip
import hashlib
import io
import time

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
    (b"BZh91AY", "replay_compressed", 400),
    (b"PBDEMS", "replay_invalid", 400),
    (DEM + b"x" * 100, "upload_too_large", 413),
])
def test_store_upload_rejections_leave_no_files(tmp_path, data, code, status):
    with pytest.raises(storage.UploadRejected) as info:
        storage.store_upload(io.BytesIO(data), tmp_path, max_bytes=50)
    assert info.value.code == code and info.value.status == status
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


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
