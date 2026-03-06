"""Replay parser abstraction layer.

Supports multiple parser backends:
- ClarityParser: Uses the clarity Java library (most complete)
- OpenDotaParser: Fetches data from OpenDota API by match ID
- MockParser: Returns mock data for development/testing
"""
from __future__ import annotations

import json
import logging
import subprocess
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from pathlib import Path

from .config import CLARITY_JAR_PATH, JAVA_PATH
from .mock_data import generate_mock_match
from .models import (
    AbilityCast,
    AbilityEvent,
    GoldXpSnapshot,
    ItemEvent,
    MatchData,
    PlayerData,
    PositionSnapshot,
    SkillEvent,
    SmokeEvent,
    TeamfightEvent,
    TeamfightPlayer,
    WardEvent,
)

logger = logging.getLogger(__name__)


class ReplayParser(ABC):
    """Abstract base class for replay parsers."""

    @abstractmethod
    async def parse(self, file_path: str) -> MatchData:
        """Parse a .dem replay file and return structured match data."""
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class MockParser(ReplayParser):
    """Returns mock data for development and testing."""

    async def parse(self, file_path: str) -> MatchData:
        return generate_mock_match()

    def name(self) -> str:
        return "mock"


class ClarityParser(ReplayParser):
    """Parse replays using the clarity Java library.

    Requires:
    - Java runtime (java command available)
    - clarity-analyzer.jar in tools/ directory

    The clarity analyzer outputs JSON with combat log entries,
    entity data, and game events.
    """

    def __init__(self, jar_path: Path | None = None, java_path: str | None = None):
        self.jar_path = jar_path or CLARITY_JAR_PATH
        self.java_path = java_path or JAVA_PATH

    def name(self) -> str:
        return "clarity"

    def _check_requirements(self) -> None:
        if not self.jar_path.exists():
            raise FileNotFoundError(
                f"Clarity JAR not found at {self.jar_path}. "
                "Run 'bash tools/setup_parser.sh' to set up the parser."
            )
        try:
            subprocess.run(
                [self.java_path, "-version"],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "Java runtime not found. Install Java 11+ to use the clarity parser."
            ) from e

    async def parse(self, file_path: str) -> MatchData:
        self._check_requirements()

        result = subprocess.run(
            [
                self.java_path,
                "-jar",
                str(self.jar_path),
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Clarity parser failed: {result.stderr}")

        raw_data = json.loads(result.stdout)
        return self._transform_data(raw_data)

    def _transform_data(self, raw: dict) -> MatchData:
        """Transform clarity JSON output into our data models."""
        players = []
        all_wards: list[WardEvent] = []
        all_ability_events: list[AbilityEvent] = []

        for p in raw.get("players", []):
            slot = p["player_slot"]
            team = "radiant" if slot < 128 else "dire"

            items = [
                ItemEvent(item_name=i["name"], time=i["time"])
                for i in p.get("item_purchases", [])
            ]

            skills = [
                SkillEvent(ability_name=s["name"], level=s["level"], time=s["time"])
                for s in p.get("ability_upgrades", [])
            ]

            wards = [
                WardEvent(
                    ward_type=w["type"],
                    x=w["x"],
                    y=w["y"],
                    time=w["time"],
                    player_slot=slot,
                    destroyed_time=w.get("destroyed_time"),
                )
                for w in p.get("wards_placed", [])
            ]
            all_wards.extend(wards)

            ability_events = [
                AbilityEvent(
                    player_slot=slot,
                    hero_name=p.get("hero_name", ""),
                    ability_name=a["name"],
                    time=a["time"],
                    target=a.get("target"),
                    x=a.get("x"),
                    y=a.get("y"),
                )
                for a in p.get("ability_casts", [])
            ]
            all_ability_events.extend(ability_events)

            players.append(PlayerData(
                player_slot=slot,
                hero_id=p.get("hero_id", 0),
                hero_name=p.get("hero_name", "Unknown"),
                player_name=p.get("player_name", f"Player {slot}"),
                team=team,
                kills=p.get("kills", 0),
                deaths=p.get("deaths", 0),
                assists=p.get("assists", 0),
                level=p.get("level", 1),
                last_hits=p.get("last_hits", 0),
                denies=p.get("denies", 0),
                gold_per_min=p.get("gold_per_min", 0),
                xp_per_min=p.get("xp_per_min", 0),
                hero_damage=p.get("hero_damage", 0),
                tower_damage=p.get("tower_damage", 0),
                hero_healing=p.get("hero_healing", 0),
                net_worth=p.get("net_worth", 0),
                items=items,
                backpack=p.get("backpack", []),
                final_items=p.get("final_items", []),
                skills=skills,
                wards=wards,
                ability_events=ability_events,
            ))

        smokes = [
            SmokeEvent(
                time=s["time"],
                team=s["team"],
                player_slots=s["players"],
                x=s.get("x", 0.0),
                y=s.get("y", 0.0),
                end_time=s.get("end_time", s["time"]),
                path=s.get("path", []),
            )
            for s in raw.get("smokes", [])
        ]

        teamfights = [
            TeamfightEvent(
                start_time=tf["start"],
                end_time=tf["end"],
                radiant_deaths=tf.get("radiant_deaths", 0),
                dire_deaths=tf.get("dire_deaths", 0),
                players=[
                    TeamfightPlayer(
                        player_slot=tp["player_slot"],
                        hero_name=tp.get("hero_name", ""),
                        hero_slug=tp.get("hero_slug", ""),
                        x=tp.get("x", 0),
                        y=tp.get("y", 0),
                        damage_dealt=tp.get("damage_dealt", 0),
                        damage_taken=tp.get("damage_taken", 0),
                        kills=tp.get("kills", 0),
                        deaths=tp.get("deaths", 0),
                        abilities_used=tp.get("abilities_used", []),
                        positions=[
                            PositionSnapshot(x=p["x"], y=p["y"], time_offset=p["time_offset"])
                            for p in tp.get("positions", [])
                        ],
                        ability_casts=[
                            AbilityCast(
                                ability_name=a["name"], ability_key=a.get("key", ""),
                                time_offset=a["time_offset"], x=a["x"], y=a["y"],
                            )
                            for a in tp.get("ability_casts", [])
                        ],
                    )
                    for tp in tf.get("players", [])
                ],
            )
            for tf in raw.get("teamfights", [])
        ]

        return MatchData(
            match_id=raw.get("match_id", 0),
            duration=raw.get("duration", 0),
            radiant_win=raw.get("radiant_win", False),
            game_mode=raw.get("game_mode", "Unknown"),
            start_time=raw.get("start_time", 0),
            radiant_score=raw.get("radiant_score", 0),
            dire_score=raw.get("dire_score", 0),
            players=players,
            wards=sorted(all_wards, key=lambda w: w.time),
            smokes=smokes,
            teamfights=teamfights,
            ability_events=sorted(all_ability_events, key=lambda e: e.time),
        )


class OpenDotaParser(ReplayParser):
    """Fetch and parse match data from OpenDota API."""

    API_BASE = "https://api.opendota.com/api"
    HEADERS = {"User-Agent": "DotaAnalyzer/1.0"}
    GAME_MODES = {
        1: "All Pick", 2: "Captain's Mode", 3: "Random Draft",
        4: "Single Draft", 5: "All Random", 22: "All Pick (Ranked)",
        23: "Turbo",
    }

    def name(self) -> str:
        return "opendota"

    def _api_get(self, path: str) -> dict:
        url = f"{self.API_BASE}{path}"
        req = urllib.request.Request(url, headers=self.HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def _api_post(self, path: str) -> dict:
        url = f"{self.API_BASE}{path}"
        req = urllib.request.Request(url, method="POST", headers=self.HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def request_parse(self, match_id: int) -> dict:
        """Request OpenDota to parse a match. Returns job info."""
        return self._api_post(f"/request/{match_id}")

    def _load_item_map(self) -> dict[int, tuple[str, str]]:
        """Load item id -> (name, key) map from static data."""
        items_path = Path(__file__).parent.parent.parent / "frontend" / "static" / "data" / "items.json"
        if not items_path.exists():
            return {}
        with open(items_path) as f:
            data = json.load(f)
        result = {}
        for key, info in data.get("by_key", {}).items():
            result[info["id"]] = (info["name"], info["key"])
        return result

    def _load_hero_map(self) -> dict[int, dict]:
        """Load hero id -> {name, slug} map from static data."""
        heroes_path = Path(__file__).parent.parent.parent / "frontend" / "static" / "data" / "heroes.json"
        if not heroes_path.exists():
            return {}
        with open(heroes_path) as f:
            data = json.load(f)
        result = {}
        for hid, info in data.get("by_id", {}).items():
            result[int(hid)] = {"name": info.get("name", ""), "slug": info.get("slug", "")}
        return result

    async def parse(self, match_id_str: str) -> MatchData:
        match_id = int(match_id_str)
        raw = self._api_get(f"/matches/{match_id}")
        hero_map = self._load_hero_map()
        item_map = self._load_item_map()
        return self._transform(raw, hero_map, item_map)

    def _transform(self, raw: dict, hero_map: dict, item_map: dict) -> MatchData:
        players = []
        all_wards: list[WardEvent] = []
        all_ability_events: list[AbilityEvent] = []

        has_parsed = raw.get("players", [{}])[0].get("gold_t") is not None

        for p in raw.get("players", []):
            slot = p["player_slot"]
            team = "radiant" if slot < 128 else "dire"
            hero_id = p.get("hero_id", 0)
            hero_info = hero_map.get(hero_id, {})
            hero_name = hero_info.get("name", f"Hero {hero_id}")
            hero_slug = hero_info.get("slug", "")

            # Items timeline from purchase_log
            items: list[ItemEvent] = []
            for purchase in p.get("purchase_log", []) or []:
                iname = purchase.get("key", "")
                itime = purchase.get("time", 0)
                items.append(ItemEvent(
                    item_name=iname.replace("_", " ").title(),
                    time=itime,
                    item_key=iname,
                ))

            # Final items from item_0..item_5
            final_items = []
            final_keys = []
            for idx in range(6):
                iid = p.get(f"item_{idx}", 0)
                if iid and iid in item_map:
                    name, key = item_map[iid]
                    final_items.append(name)
                    final_keys.append(key)

            # Skill build from ability_upgrades_arr
            # We don't have ability name mapping from IDs, use generic
            skills: list[SkillEvent] = []
            ability_arr = p.get("ability_upgrades_arr") or []
            seen_abilities: dict[int, int] = {}
            for i, ab_id in enumerate(ability_arr):
                seen_abilities[ab_id] = seen_abilities.get(ab_id, 0) + 1
                skills.append(SkillEvent(
                    ability_name=f"ability_{ab_id}",
                    ability_key=str(ab_id),
                    level=seen_abilities[ab_id],
                    time=(i + 1) * 60,  # approximate: 1 level per minute
                ))

            # Ward placements (if parsed)
            # Build obs destruction time lookup from obs_left_log
            obs_left = {
                w.get("ehandle"): w.get("time")
                for w in p.get("obs_left_log", []) or []
                if w.get("ehandle")
            }
            sen_left = {
                w.get("ehandle"): w.get("time")
                for w in p.get("sen_left_log", []) or []
                if w.get("ehandle")
            }

            wards: list[WardEvent] = []
            for w in p.get("obs_log", []) or []:
                ehandle = w.get("ehandle")
                destroyed = obs_left.get(ehandle) if ehandle else None
                wards.append(WardEvent(
                    ward_type="observer",
                    x=w.get("x", 0) * 16000 / 256 - 8000,  # cell x → game x
                    y=w.get("y", 0) * 16000 / 256 - 8000,  # cell y → game y
                    time=w.get("time", 0),
                    player_slot=slot,
                    destroyed_time=destroyed,
                ))
            for w in p.get("sen_log", []) or []:
                ehandle = w.get("ehandle")
                destroyed = sen_left.get(ehandle) if ehandle else None
                wards.append(WardEvent(
                    ward_type="sentry",
                    x=w.get("x", 0) * 16000 / 256 - 8000,  # cell x → game x
                    y=w.get("y", 0) * 16000 / 256 - 8000,  # cell y → game y
                    time=w.get("time", 0),
                    player_slot=slot,
                    destroyed_time=destroyed,
                ))
            all_wards.extend(wards)

            players.append(PlayerData(
                player_slot=slot,
                hero_id=hero_id,
                hero_name=hero_name,
                hero_slug=hero_slug,
                player_name=p.get("personaname") or f"Player {slot}",
                team=team,
                kills=p.get("kills", 0),
                deaths=p.get("deaths", 0),
                assists=p.get("assists", 0),
                level=p.get("level", 1),
                last_hits=p.get("last_hits", 0),
                denies=p.get("denies", 0),
                gold_per_min=p.get("gold_per_min", 0),
                xp_per_min=p.get("xp_per_min", 0),
                hero_damage=p.get("hero_damage", 0),
                tower_damage=p.get("tower_damage", 0),
                hero_healing=p.get("hero_healing", 0),
                net_worth=p.get("net_worth") or p.get("total_gold", 0),
                items=items,
                backpack=[],
                final_items=final_items,
                final_item_keys=final_keys,
                skills=skills,
                wards=wards,
                ability_events=[],
            ))

        # Teamfights (if parsed)
        teamfights: list[TeamfightEvent] = []
        for tf in raw.get("teamfights", []) or []:
            tf_players = []
            for i, tp in enumerate(tf.get("players", [])):
                slot = i if i < 5 else 128 + (i - 5)
                hero_id = raw["players"][i].get("hero_id", 0)
                hero_info = hero_map.get(hero_id, {})
                tf_players.append(TeamfightPlayer(
                    player_slot=slot,
                    hero_name=hero_info.get("name", ""),
                    hero_slug=hero_info.get("slug", ""),
                    x=0, y=0,
                    damage_dealt=tp.get("damage", 0),
                    damage_taken=0,
                    kills=tp.get("kills", 0),
                    deaths=tp.get("deaths", 0),
                    abilities_used=list((tp.get("ability_uses") or {}).keys()),
                ))
            r_deaths = sum(tp.deaths for tp in tf_players[:5])
            d_deaths = sum(tp.deaths for tp in tf_players[5:])
            teamfights.append(TeamfightEvent(
                start_time=int(tf.get("start", 0)),
                end_time=int(tf.get("end", 0)),
                radiant_deaths=r_deaths,
                dire_deaths=d_deaths,
                players=tf_players,
            ))

        # Gold/XP timeline (if parsed)
        gold_xp_timeline: list[GoldXpSnapshot] = []
        if has_parsed:
            r_players = [p for p in raw["players"] if p["player_slot"] < 128]
            d_players = [p for p in raw["players"] if p["player_slot"] >= 128]
            min_len = min(
                min((len(p.get("gold_t") or []) for p in r_players), default=0),
                min((len(p.get("gold_t") or []) for p in d_players), default=0),
            )
            for minute in range(min_len):
                r_gold = sum(p["gold_t"][minute] for p in r_players)
                d_gold = sum(p["gold_t"][minute] for p in d_players)
                r_xp = sum(p["xp_t"][minute] for p in r_players)
                d_xp = sum(p["xp_t"][minute] for p in d_players)
                gold_xp_timeline.append(GoldXpSnapshot(
                    time=minute * 60,
                    radiant_gold=r_gold, dire_gold=d_gold,
                    radiant_xp=r_xp, dire_xp=d_xp,
                ))

        # Smokes (if parsed via objectives)
        smokes: list[SmokeEvent] = []

        game_mode_id = raw.get("game_mode", 0)
        game_mode = self.GAME_MODES.get(game_mode_id, f"Mode {game_mode_id}")

        return MatchData(
            match_id=raw.get("match_id", 0),
            duration=raw.get("duration", 0),
            radiant_win=raw.get("radiant_win", False),
            game_mode=game_mode,
            start_time=raw.get("start_time", 0),
            radiant_score=raw.get("radiant_score", 0),
            dire_score=raw.get("dire_score", 0),
            players=players,
            wards=sorted(all_wards, key=lambda w: w.time),
            smokes=smokes,
            teamfights=teamfights,
            ability_events=sorted(all_ability_events, key=lambda e: e.time),
            gold_xp_timeline=gold_xp_timeline,
        )


def get_parser(parser_type: str = "auto") -> ReplayParser:
    """Get the appropriate parser based on availability.

    Args:
        parser_type: "clarity", "mock", or "auto" (tries clarity first)
    """
    if parser_type == "mock":
        return MockParser()

    if parser_type == "clarity":
        return ClarityParser()

    # Auto: try clarity, fall back to mock
    if parser_type == "auto":
        clarity = ClarityParser()
        try:
            clarity._check_requirements()
            return clarity
        except (FileNotFoundError, RuntimeError):
            logger.warning("Clarity parser not available, using mock data")
            return MockParser()

    raise ValueError(f"Unknown parser type: {parser_type}")
