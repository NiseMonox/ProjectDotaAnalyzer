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
    LaningCreepKill,
    LaningDamageEvent,
    LaningRegenEvent,
    LaningSnapshot,
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

    def _load_slug_to_hero(self) -> dict[str, dict]:
        """Load slug/internal_name -> {id, name, slug} reverse mapping.

        The Java clarity analyzer outputs hero_name as internal names
        (e.g. 'drow_ranger', 'hoodwink') that match the slug field in
        heroes.json. This map lets us look up hero_id and display name
        from that internal name.
        """
        heroes_path = Path(__file__).parent.parent.parent / "frontend" / "static" / "data" / "heroes.json"
        if not heroes_path.exists():
            return {}
        with open(heroes_path) as f:
            data = json.load(f)
        slug_to_hero: dict[str, dict] = {}
        for hid, info in data.get("by_id", {}).items():
            slug = info.get("slug", "")
            if slug:
                slug_to_hero[slug] = {
                    "id": int(hid),
                    "name": info.get("name", ""),
                    "slug": slug,
                }
        return slug_to_hero

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
                f"Java runtime not found or failed at '{self.java_path}'. "
                "Install Java 17+ to use the clarity parser (clarity 3.x requires Java 17)."
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

        hero_map = self._load_hero_map()
        slug_to_hero = self._load_slug_to_hero()

        # Consumable items to exclude when approximating final items
        CONSUMABLES = {
            "tango", "clarity", "enchanted_mango", "flask", "faerie_fire",
            "ward_observer", "ward_sentry", "dust", "smoke_of_deceit",
            "tome_of_knowledge", "tp_scroll", "tpscroll", "town_portal_scroll",
            "ward_dispenser", "infused_raindrop",
        }

        GAME_MODES = {
            1: "All Pick", 2: "Captain's Mode", 3: "Random Draft",
            4: "Single Draft", 5: "All Random", 22: "All Pick (Ranked)",
            23: "Turbo",
        }

        for p in raw.get("players", []):
            slot = p["player_slot"]
            team = "radiant" if slot < 128 else "dire"
            hero_id = p.get("hero_id", 0)
            internal_name = p.get("hero_name", "Unknown")
            hero_name = internal_name
            hero_slug = ""

            # Try to resolve hero info from the slug reverse map first
            # (Java clarity outputs hero_name as internal slug e.g. "drow_ranger")
            if slug_to_hero and internal_name:
                hero_info = slug_to_hero.get(internal_name, {})
                if hero_info:
                    hero_id = hero_info.get("id", hero_id)
                    hero_name = hero_info.get("name", internal_name)
                    hero_slug = hero_info.get("slug", internal_name)

            # Fallback: resolve by hero_id if we got a valid one from the data
            if not hero_slug and hero_id and hero_map:
                hero_info = hero_map.get(hero_id, {})
                if hero_info:
                    hero_name = hero_info.get("name", hero_name)
                    hero_slug = hero_info.get("slug", "")

            # Items timeline from purchase_log
            items = [
                ItemEvent(
                    item_name=i.get("key", "").replace("_", " ").title(),
                    time=i.get("time", 0),
                    item_key=i.get("key", ""),
                )
                for i in p.get("purchase_log", []) or p.get("item_purchases", []) or []
            ]

            # Use starting_items snapshot (from entity inventory at game start)
            # to replace unreliable purchase_log starting items
            starting_items_snapshot = p.get("starting_items") or []
            if starting_items_snapshot:
                # Remove purchase_log items with time <= 0 (unreliable)
                items = [it for it in items if it.time > 0]
                # Add snapshot items as starting items
                for skey in starting_items_snapshot:
                    items.insert(0, ItemEvent(
                        item_name=skey.replace("_", " ").title(),
                        time=-1,
                        item_key=skey,
                    ))

            # Skill build — filter out non-skill-upgrade entries
            _SKIP_ABILITIES = {
                "ability_lamp_use", "ability_capture", "plus_high_five",
                "twin_gate_portal_warp", "plus_guild_banner",
            }
            raw_skills = p.get("ability_upgrades", []) or []
            skills = []
            for s in raw_skills:
                name = s.get("name", "")
                if not name or name in _SKIP_ABILITIES:
                    continue
                if "_release" in name:  # e.g. hoodwink_sharpshooter_release
                    continue
                skills.append(SkillEvent(
                    ability_name=name,
                    ability_key=s.get("key", "") or name,
                    level=s.get("level", 1),
                    time=s.get("time", 0),
                ))

            # Ward placements
            wards = [
                WardEvent(
                    ward_type=w.get("type", "observer"),
                    x=w.get("x", 0),
                    y=w.get("y", 0),
                    time=w.get("time", 0),
                    player_slot=slot,
                    destroyed_time=w.get("destroyed_time"),
                )
                for w in p.get("wards_placed", []) or []
            ]
            all_wards.extend(wards)

            # Ability cast events
            ability_events = [
                AbilityEvent(
                    player_slot=slot,
                    hero_name=hero_name,
                    ability_name=a.get("name", ""),
                    time=a.get("time", 0),
                    target=a.get("target"),
                    x=a.get("x"),
                    y=a.get("y"),
                )
                for a in p.get("ability_casts", []) or []
            ]
            all_ability_events.extend(ability_events)

            # Per-minute timelines
            gold_t = p.get("gold_t") or []
            xp_t = p.get("xp_t") or []
            lh_t = p.get("lh_t") or []

            # Damage matrices
            killed = p.get("killed") or {}
            killed_by = p.get("killed_by") or {}
            damage = p.get("damage") or {}
            damage_taken = p.get("damage_taken") or {}
            damage_inflictor = p.get("damage_inflictor") or {}

            # Economy reasons
            gold_reasons = p.get("gold_reasons") or {}
            xp_reasons = p.get("xp_reasons") or {}

            # Purchase log
            purchase_log = p.get("purchase_log") or p.get("item_purchases") or []

            # Final items: use provided data, or approximate from purchase_log
            final_items = p.get("final_items") or []
            final_item_keys: list[str] = []
            if final_items:
                # Java analyzer returns keys directly (e.g. "staff_of_wizardry")
                # Use them as both display names and keys
                final_item_keys = [item for item in final_items]
            elif purchase_log:
                # Fallback: use last ~6 non-consumable purchases
                non_consumable = [
                    entry for entry in purchase_log
                    if entry.get("key", "") not in CONSUMABLES
                ]
                last_items = non_consumable[-6:] if len(non_consumable) >= 6 else non_consumable
                final_items = [
                    entry.get("key", "").replace("_", " ").title()
                    for entry in last_items
                ]
                final_item_keys = [entry.get("key", "") for entry in last_items]

            players.append(PlayerData(
                player_slot=slot,
                hero_id=hero_id,
                hero_name=hero_name,
                hero_slug=hero_slug,
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
                final_items=final_items,
                final_item_keys=final_item_keys,
                skills=skills,
                wards=wards,
                ability_events=ability_events,
                gold_t=gold_t,
                xp_t=xp_t,
                lh_t=lh_t,
                lane=p.get("lane", 0) or 0,
                lane_role=p.get("lane_role", 0) or 0,
                killed=killed,
                killed_by=killed_by,
                damage=damage,
                damage_taken=damage_taken,
                damage_inflictor=damage_inflictor,
                gold_reasons=gold_reasons,
                xp_reasons=xp_reasons,
                buyback_count=p.get("buyback_count", 0) or 0,
                roshans_killed=p.get("roshans_killed", 0) or 0,
                courier_kills=p.get("courier_kills", 0) or 0,
                purchase_log=purchase_log,
                laning_snapshots=[LaningSnapshot(**s) for s in p.get("laning_snapshots", [])],
                laning_creep_kills=[LaningCreepKill(**c) for c in p.get("laning_creep_kills", [])],
                laning_denies=[LaningCreepKill(**d) for d in p.get("laning_denies", [])],
                laning_damage_events=[LaningDamageEvent(**e) for e in p.get("laning_damage_events", [])],
                laning_regen_events=[LaningRegenEvent(**r) for r in p.get("laning_regen_events", [])],
            ))

        # Top-level wards from clarity (Java outputs wards at match level)
        # Ward coords from Java are already in game space (cellX*128 - 8192)
        for w in raw.get("wards", []) or []:
            all_wards.append(WardEvent(
                ward_type=w.get("type", "observer"),
                x=w.get("x", 0),
                y=w.get("y", 0),
                time=w.get("time", 0),
                player_slot=w.get("player_slot", -1),
                destroyed_time=w.get("destroyed_time"),
            ))

        # Smokes
        smokes = [
            SmokeEvent(
                time=s["time"],
                team=s.get("team", "radiant"),
                player_slots=s.get("player_slots", s.get("players", [])),
                x=s.get("x", 0.0),
                y=s.get("y", 0.0),
                end_time=s.get("end_time", s["time"]),
                path=s.get("path", []),
                result=s.get("result", "unknown"),
                kills_gained=s.get("kills_gained", 0),
                deaths_suffered=s.get("deaths_suffered", 0),
            )
            for s in raw.get("smokes", []) or []
        ]

        # Teamfights - resolve hero names via slug_to_hero
        teamfights = []
        for tf in raw.get("teamfights", []) or []:
            tf_players = []
            for tp in tf.get("players", []) or []:
                tf_hero_internal = tp.get("hero_name", "")
                tf_hero_info = slug_to_hero.get(tf_hero_internal, {})
                tf_hero_name = tf_hero_info.get("name", tf_hero_internal)
                tf_hero_slug = tf_hero_info.get("slug", tf_hero_internal)
                tf_players.append(TeamfightPlayer(
                    player_slot=tp["player_slot"],
                    hero_name=tf_hero_name,
                    hero_slug=tf_hero_slug,
                    x=tp.get("x", 0),
                    y=tp.get("y", 0),
                    damage_dealt=tp.get("damage_dealt", 0),
                    damage_taken=tp.get("damage_taken", 0),
                    healing=tp.get("healing", 0),
                    kills=tp.get("kills", 0),
                    deaths=tp.get("deaths", 0),
                    abilities_used=tp.get("abilities_used", []),
                    gold_delta=tp.get("gold_delta", 0),
                    xp_delta=tp.get("xp_delta", 0),
                    item_uses=tp.get("item_uses") or {},
                    buybacks=tp.get("buybacks", 0),
                    death_time_offset=tp.get("death_time_offset"),
                    positions=[
                        PositionSnapshot(x=ps["x"], y=ps["y"], time_offset=ps["time_offset"])
                        for ps in tp.get("positions", []) or []
                    ],
                    ability_casts=[
                        AbilityCast(
                            ability_name=a.get("name", ""), ability_key=a.get("key", ""),
                            time_offset=a.get("time_offset", 0), x=a.get("x", 0), y=a.get("y", 0),
                        )
                        for a in tp.get("ability_casts", []) or []
                    ],
                ))
            teamfights.append(TeamfightEvent(
                start_time=tf["start"],
                end_time=tf["end"],
                radiant_deaths=tf.get("radiant_deaths", 0),
                dire_deaths=tf.get("dire_deaths", 0),
                gold_delta=tf.get("gold_delta", 0),
                players=tf_players,
            ))

        # Gold/XP timeline from per-player data
        gold_xp_timeline: list[GoldXpSnapshot] = []
        r_players = [p for p in players if p.team == "radiant"]
        d_players = [p for p in players if p.team == "dire"]
        min_len = min(
            min((len(p.gold_t) for p in r_players), default=0),
            min((len(p.gold_t) for p in d_players), default=0),
        )
        for minute in range(min_len):
            r_gold = sum(p.gold_t[minute] for p in r_players)
            d_gold = sum(p.gold_t[minute] for p in d_players)
            r_xp = sum(p.xp_t[minute] for p in r_players if len(p.xp_t) > minute)
            d_xp = sum(p.xp_t[minute] for p in d_players if len(p.xp_t) > minute)
            gold_xp_timeline.append(GoldXpSnapshot(
                time=minute * 60,
                radiant_gold=r_gold, dire_gold=d_gold,
                radiant_xp=r_xp, dire_xp=d_xp,
            ))

        # Advantage data
        radiant_gold_adv = raw.get("radiant_gold_adv") or []
        radiant_xp_adv = raw.get("radiant_xp_adv") or []

        # Objectives
        objectives = raw.get("objectives") or []

        # Chat
        chat = raw.get("chat") or []

        # Building status
        tower_status_radiant = raw.get("tower_status_radiant", 0) or 0
        tower_status_dire = raw.get("tower_status_dire", 0) or 0
        barracks_status_radiant = raw.get("barracks_status_radiant", 0) or 0
        barracks_status_dire = raw.get("barracks_status_dire", 0) or 0

        # First blood
        first_blood_time = raw.get("first_blood_time", 0) or 0

        # Game mode
        game_mode_id = raw.get("game_mode", 0)
        game_mode = GAME_MODES.get(game_mode_id, f"Mode {game_mode_id}") if isinstance(game_mode_id, int) else str(game_mode_id)

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
            radiant_gold_adv=radiant_gold_adv,
            radiant_xp_adv=radiant_xp_adv,
            picks_bans=raw.get("picks_bans") or [],
            objectives=objectives,
            chat=chat,
            tower_status_radiant=tower_status_radiant,
            tower_status_dire=tower_status_dire,
            barracks_status_radiant=barracks_status_radiant,
            barracks_status_dire=barracks_status_dire,
            first_blood_time=first_blood_time,
            region=raw.get("region", 0) or 0,
            patch=str(raw.get("patch", "") or ""),
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

    # Typical starting items and their gold costs
    _STARTING_ITEM_COSTS: dict[str, int] = {
        "tango": 90, "flask": 110, "faerie_fire": 70, "enchanted_mango": 65,
        "clarity": 50, "branches": 50, "circlet": 150, "slippers": 135,
        "gauntlets": 135, "mantle": 135, "quelling_blade": 100, "magic_stick": 200,
        "ring_of_protection": 175, "orb_of_venom": 275, "blight_stone": 300,
        "wind_lace": 250, "blood_grenade": 65,
        "ward_observer": 0, "ward_sentry": 50, "smoke_of_deceit": 50,
    }
    _STARTING_GOLD = 600
    _STARTING_INFER_WINDOW = 90  # seconds after game start

    def _infer_starting_items(self, items: list[ItemEvent]) -> list[ItemEvent]:
        """Reclassify early purchases as starting items when purchase_log is incomplete.

        OpenDota's purchase_log often drops pre-game purchases. This method
        checks if the recorded starting gold is too low, and promotes early-game
        typical starting items into the starting phase (time = -1).
        """
        starting_cost = sum(
            self._STARTING_ITEM_COSTS.get(it.item_key, 0)
            for it in items if it.time <= 0
        )
        if starting_cost >= self._STARTING_GOLD - 50:
            return items  # already accounted for, no fix needed

        budget = self._STARTING_GOLD - starting_cost
        result = list(items)
        for i, it in enumerate(result):
            if it.time <= 0:
                continue  # already a starting item
            if it.time > self._STARTING_INFER_WINDOW:
                break  # past the inference window
            cost = self._STARTING_ITEM_COSTS.get(it.item_key, 0)
            if cost == 0 and it.item_key not in self._STARTING_ITEM_COSTS:
                continue  # not a typical starting item
            if cost <= budget:
                result[i] = ItemEvent(
                    item_name=it.item_name,
                    time=-1,
                    item_key=it.item_key,
                )
                budget -= cost
                if budget < 50:
                    break
        # Re-sort so starting items come first
        result.sort(key=lambda x: x.time)
        return result

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

            # Infer missing starting items: OpenDota purchase_log often drops
            # pre-game purchases. If recorded starting cost < 600g, reclassify
            # early typical starting items into starting phase.
            items = self._infer_starting_items(items)

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

            # Per-minute timelines
            gold_t = p.get("gold_t") or []
            xp_t = p.get("xp_t") or []
            lh_t = p.get("lh_t") or []

            # Lane data
            lane = p.get("lane", 0) or 0
            lane_role = p.get("lane_role", 0) or 0
            is_roaming = bool(p.get("is_roaming", False))
            lane_efficiency = p.get("lane_efficiency", 0.0) or 0.0
            lane_efficiency_pct = p.get("lane_efficiency_pct", 0.0) or 0.0

            # Damage
            killed = p.get("killed") or {}
            killed_by = p.get("killed_by") or {}
            damage = p.get("damage") or {}
            damage_taken = p.get("damage_taken") or {}
            damage_inflictor = p.get("damage_inflictor") or {}

            # Economy reasons
            gold_reasons = p.get("gold_reasons") or {}
            xp_reasons = p.get("xp_reasons") or {}

            # Benchmarks
            benchmarks = p.get("benchmarks") or {}

            # Runes
            runes_log = p.get("runes_log") or []
            runes = p.get("runes") or {}

            # Stats
            stuns = p.get("stuns", 0.0) or 0.0
            actions_per_min = p.get("actions_per_min", 0) or 0
            teamfight_participation = p.get("teamfight_participation", 0.0) or 0.0
            buyback_count = len(p.get("buyback_log") or [])
            obs_placed = p.get("obs_placed", 0) or 0
            sen_placed = p.get("sen_placed", 0) or 0
            observer_kills = p.get("observer_kills", 0) or 0
            sentry_kills = p.get("sentry_kills", 0) or 0
            lane_kills = p.get("lane_kills", 0) or 0
            neutral_kills = p.get("neutral_kills", 0) or 0
            ancient_kills = p.get("ancient_kills", 0) or 0
            tower_kills = p.get("tower_kills", 0) or 0
            roshans_killed = p.get("roshans_killed", 0) or 0
            courier_kills = p.get("courier_kills", 0) or 0
            multi_kills = {str(k): v for k, v in (p.get("multi_kills") or {}).items()}
            kill_streaks = {str(k): v for k, v in (p.get("kill_streaks") or {}).items()}
            purchase_log = p.get("purchase_log") or []
            lane_pos = p.get("lane_pos") or {}

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
                gold_t=gold_t,
                xp_t=xp_t,
                lh_t=lh_t,
                lane=lane,
                lane_role=lane_role,
                is_roaming=is_roaming,
                lane_efficiency=lane_efficiency,
                lane_efficiency_pct=lane_efficiency_pct,
                killed=killed,
                killed_by=killed_by,
                damage=damage,
                damage_taken=damage_taken,
                damage_inflictor=damage_inflictor,
                gold_reasons=gold_reasons,
                xp_reasons=xp_reasons,
                benchmarks=benchmarks,
                runes_log=runes_log,
                runes=runes,
                stuns=stuns,
                actions_per_min=actions_per_min,
                teamfight_participation=teamfight_participation,
                buyback_count=buyback_count,
                obs_placed=obs_placed,
                sen_placed=sen_placed,
                observer_kills=observer_kills,
                sentry_kills=sentry_kills,
                lane_kills=lane_kills,
                neutral_kills=neutral_kills,
                ancient_kills=ancient_kills,
                tower_kills=tower_kills,
                roshans_killed=roshans_killed,
                courier_kills=courier_kills,
                multi_kills=multi_kills,
                kill_streaks=kill_streaks,
                purchase_log=purchase_log,
                lane_pos=lane_pos,
            ))

        # Teamfights (if parsed)
        teamfights: list[TeamfightEvent] = []
        for tf in raw.get("teamfights", []) or []:
            tf_players = []
            r_gold_delta = 0
            for i, tp in enumerate(tf.get("players", [])):
                slot = i if i < 5 else 128 + (i - 5)
                hero_id = raw["players"][i].get("hero_id", 0)
                hero_info = hero_map.get(hero_id, {})
                gd = tp.get("gold_delta", 0) or 0
                if i < 5:
                    r_gold_delta += gd
                else:
                    r_gold_delta -= gd
                tf_players.append(TeamfightPlayer(
                    player_slot=slot,
                    hero_name=hero_info.get("name", ""),
                    hero_slug=hero_info.get("slug", ""),
                    x=0, y=0,
                    damage_dealt=tp.get("damage", 0),
                    damage_taken=0,
                    healing=tp.get("healing", 0) or 0,
                    kills=tp.get("kills", 0),
                    deaths=tp.get("deaths", 0),
                    abilities_used=list((tp.get("ability_uses") or {}).keys()),
                    gold_delta=gd,
                    xp_delta=tp.get("xp_delta", 0) or 0,
                    item_uses=tp.get("item_uses") or {},
                    buybacks=tp.get("buybacks", 0) or 0,
                ))
            r_deaths = sum(tp.deaths for tp in tf_players[:5])
            d_deaths = sum(tp.deaths for tp in tf_players[5:])
            teamfights.append(TeamfightEvent(
                start_time=int(tf.get("start", 0)),
                end_time=int(tf.get("end", 0)),
                radiant_deaths=r_deaths,
                dire_deaths=d_deaths,
                gold_delta=r_gold_delta,
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

        # Advantage data (directly from API)
        radiant_gold_adv = raw.get("radiant_gold_adv") or []
        radiant_xp_adv = raw.get("radiant_xp_adv") or []

        # Draft
        picks_bans = raw.get("picks_bans") or []

        # Objectives
        objectives = raw.get("objectives") or []

        # Chat
        chat = raw.get("chat") or []

        # Building status
        tower_status_radiant = raw.get("tower_status_radiant", 0) or 0
        tower_status_dire = raw.get("tower_status_dire", 0) or 0
        barracks_status_radiant = raw.get("barracks_status_radiant", 0) or 0
        barracks_status_dire = raw.get("barracks_status_dire", 0) or 0

        # First blood
        first_blood_time = raw.get("first_blood_time", 0) or 0

        # Region/patch
        region = raw.get("region", 0) or 0
        patch_num = raw.get("patch", 0) or 0

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
            radiant_gold_adv=radiant_gold_adv,
            radiant_xp_adv=radiant_xp_adv,
            picks_bans=picks_bans,
            objectives=objectives,
            chat=chat,
            tower_status_radiant=tower_status_radiant,
            tower_status_dire=tower_status_dire,
            barracks_status_radiant=barracks_status_radiant,
            barracks_status_dire=barracks_status_dire,
            first_blood_time=first_blood_time,
            region=region,
            patch=str(patch_num),
        )


def get_parser(parser_type: str = "auto") -> ReplayParser:
    """Get the appropriate parser based on availability.

    Args:
        parser_type: "clarity", "opendota", "mock", or "auto" (tries clarity first)
    """
    if parser_type == "mock":
        return MockParser()

    if parser_type == "clarity":
        return ClarityParser()

    if parser_type == "opendota":
        return OpenDotaParser()

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
