"""Acceptance run of the real analyzer JAR on real replays (opt-in).

Replays are 40-110 MB each and cannot live in the repository, so this module
is skipped unless ``CLARITY_TEST_REPLAYS`` names a directory with ``*.dem``
files (searched recursively), and a Java 17+ runtime and the JAR exist::

    CLARITY_TEST_REPLAYS=replays backend/venv/bin/python -m pytest backend/tests/test_parsing_replays.py

Every replay must be read to its end (``partial`` false: the reason for the
clarity 4 upgrade was a current-patch replay that stopped after five
minutes), without handler errors, and must satisfy the kill-log invariants
after the Python transform. To check the numbers themselves against Valve's,
compare the output with OpenDota's (tools/compare_with_opendota.py).
"""
from __future__ import annotations

import json
import os
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import pytest  # noqa: E402

from backend.app import parser  # noqa: E402
from backend.app.config import CLARITY_JAR_PATH, JAVA_PATH  # noqa: E402
from backend.app.parser import ClarityParser  # noqa: E402

_DIR = os.environ.get("CLARITY_TEST_REPLAYS", "").strip()
REPLAYS = sorted(Path(_DIR).rglob("*.dem")) if _DIR and Path(_DIR).is_dir() else []

pytestmark = pytest.mark.skipif(not REPLAYS, reason="set CLARITY_TEST_REPLAYS to a directory with .dem files")

# parse_errors keys that report something about the match, not a failed handler
_DIAGNOSTICS = {"score_credits"}


@pytest.fixture(scope="module")
def runtime():
    if not Path(CLARITY_JAR_PATH).is_file() or not parser._run_java_version(JAVA_PATH):
        pytest.skip("Java 17+ or tools/clarity-analyzer.jar is missing")
    return [JAVA_PATH, "-Xmx1536m", "-Dfile.encoding=UTF-8", "-jar", str(CLARITY_JAR_PATH)]


def _team(slot: int) -> str:
    return "radiant" if slot < 128 else "dire"


def _jar_version() -> str:
    """Implementation-Version of the JAR's manifest: what the analyzer writes as analyzer_version."""
    with zipfile.ZipFile(CLARITY_JAR_PATH) as jar:
        manifest = jar.read("META-INF/MANIFEST.MF").decode("utf-8")
    return next(line.split(": ", 1)[1] for line in manifest.splitlines() if line.startswith("Implementation-Version: "))


@pytest.mark.parametrize("replay", REPLAYS, ids=lambda path: path.stem)
def test_replay_is_read_to_its_end(runtime, replay):
    done = subprocess.run(runtime + [str(replay)], capture_output=True, timeout=300)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")[-2000:]
    raw = json.loads(done.stdout.decode("utf-8"))

    assert raw["schema_version"] == 3 and raw["partial"] is False and raw["game_started"] is True
    assert raw["analyzer_version"] == _jar_version()
    assert raw["duration"] > 300 and isinstance(raw["radiant_win"], bool)
    assert set(raw["parse_errors"]) <= _DIAGNOSTICS, raw["parse_errors"]
    # a complete replay ends with Valve's post-game record and has the team counters
    assert raw["match_record"] is True and raw["score_source"] == "scoreboard"
    assert len(raw["players"]) == 10 and len(raw["buildings"]) == 36
    for player in raw["players"]:
        assert player["hero_id"] > 0 and not player["hero_name"].startswith("unknown"), player["hero_name"]
        assert player["kda_source"] == player["damage_source"] == "scoreboard"
        assert player["healing_source"] == "match_record" and player["final_items_source"] == "inventory"
        assert player["ability_upgrades_source"] == "entity"
        assert len(player["gold_t"]) == raw["duration"] // 60 + 1

    match = ClarityParser()._transform_data(raw)
    assert match.partial is False and match.winner_known is True and len(match.players) == 10
    loose = raw["parse_errors"].get("score_credits", 0)
    for player in match.players:
        slot = player.player_slot
        assert sum(1 for k in match.kills_log if k.victim_slot == slot) == player.deaths, player.hero_name
        kills = sum(1 for k in match.kills_log if k.killer_slot == slot and not k.deny)
        assists = sum(1 for k in match.kills_log if slot in k.assist_slots and not k.deny)
        assert 0 <= player.kills - kills <= loose and 0 <= player.assists - assists <= loose, player.hero_name
        assert player.hero_slug and player.final_item_keys == [parser.canonical_item_key(k) for k in player.final_item_keys]
    # every death says when the hero was alive again (null: the game ended first), and every
    # ward that is gone says who ended it: an enemy, its own side, or nobody (it ran out)
    assert all(k.respawn_time is None or k.time < k.respawn_time <= match.duration for k in match.kills_log)
    assert any(k.respawn_time for k in match.kills_log)
    gone = [w for w in raw["wards"] if w["destroyed_time"] is not None]
    assert all(isinstance(w.get("dewarded"), bool) and isinstance(w.get("denied"), bool) for w in gone)
    for ward in match.wards:
        assert not (ward.dewarded and ward.denied), ward
        if ward.killer_slot is not None:
            assert (_team(ward.killer_slot) != ward.team) == ward.dewarded and ward.denied != ward.dewarded, ward
    # the score is Valve's counter; it never exceeds the deaths of the other side
    deaths = {team: sum(p.deaths for p in match.players if p.team == team) for team in ("radiant", "dire")}
    assert 0 < match.radiant_score <= deaths["dire"] and 0 < match.dire_score <= deaths["radiant"]
    # ... and the kill log explains it: a death scores for the side of the credited hero, else
    # for the side of the unit that landed the kill (killer_team: a tower, a creep, the fountain)
    scored = Counter((_team(k.killer_slot) if k.killer_slot is not None else k.killer_team, _team(k.victim_slot))
                     for k in match.kills_log if not k.deny)
    assert abs(scored["radiant", "dire"] - match.radiant_score) <= loose, scored
    assert abs(scored["dire", "radiant"] - match.dire_score) <= loose, scored
