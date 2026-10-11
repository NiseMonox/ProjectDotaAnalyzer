"""Data models for Dota 2 match analysis.

Coordinates: every x/y below (wards, smokes incl. path, teamfight positions and
ability casts, ability events, laning snapshots, buildings) is in Dota 2 world
units (see ``geo.py``). The only exception is ``PlayerData.lane_pos``, which is
OpenDota's raw cell heatmap.

Enumerations are stable lowercase codes (no localized labels); the frontend
translates them. Every enumeration and key format is documented on the field
that carries it. The three sources (Clarity replay parse, OpenDota API, mock)
fill the same fields with the same formats:

- Item keys (``ItemEvent.item_key``, ``final_item_keys``, ``backpack``,
  ``neutral_item``, ``purchase_log[].key``): the OpenDota items.json key
  without the ``item_`` prefix (``travel_boots``, ``dust``, ``invis_sword``,
  ``bottle``, ``boots``). A key that is missing from the static catalog is kept
  as it is; an OpenDota item id that cannot be resolved becomes ``item_<id>``.
- Ability keys (``SkillEvent.ability_key``, ``AbilityCast.ability_key``,
  ``AbilityEvent.ability_key``, ``TeamfightPlayer.abilities_used``,
  ``damage_inflictor`` keys): the internal ability name
  (``antimage_mana_break``). Talents are ``special_bonus_*``; a talent pick is
  any ``special_bonus_`` key other than ``special_bonus_attributes`` (the
  attribute bonus bought with a skill point).
- Hero names (``PlayerData.hero_name`` and every field documented as a hero
  name): the display name from heroes.json ("Wraith King").
- Slots: ``player_slot`` is 0-4 for Radiant and 128-132 for Dire.
"""
from __future__ import annotations

from pydantic import BaseModel


class ItemEvent(BaseModel):
    item_name: str  # English display name (title-cased key when not in the catalog)
    # Game time of the purchase in seconds (negative = before the horn; a
    # strategy-time purchase carries the pre-game start, about -90). -1 also
    # marks a starting item of older replay-parser output whose purchase time
    # is unknown.
    time: int
    item_key: str = ""  # item key (see module docstring); also the CDN icon name
    charges: int | None = None


class SkillEvent(BaseModel):
    # English display name (the key when not in the catalog). A talent name
    # never carries a "{s:...}" template token: a number the catalogs do not
    # have is left out ("+Gush Damage"), or reads "X" inside a sentence
    # ("Tidebringer Ignores X% Armor"); a talent in no catalog is named after
    # its key ("Wraith King Facet 1"). "special_bonus_attributes" is
    # "Attribute Bonus". This is the name at parse time; the analyzer names
    # a talent again from ``ability_key`` (the fetched game data may only
    # have arrived later), see ``MatchData.talent_names_patch``.
    ability_name: str
    # Ability key (see module docstring); "special_bonus_*" for talents. An
    # OpenDota ability id that cannot be resolved is kept as the id string.
    ability_key: str = ""
    level: int  # level of this ability after the upgrade (1-based)
    time: int   # game time in seconds (estimated from the XP curve for OpenDota)


class WardEvent(BaseModel):
    ward_type: str  # "observer" or "sentry"
    x: float        # world units
    y: float
    time: int       # game time of placement (negative = before the horn)
    player_slot: int  # owner; -1 when the owner is unknown (see ``team``)
    # Game time the ward died (dewarded or expired); None = alive at game end
    destroyed_time: int | None = None
    team: str = ""  # "radiant"/"dire"; empty = derive from player_slot
    # True only when an enemy destroyed the ward; False = expired naturally,
    # denied or still alive at game end. Consumers must use this, not a
    # lifetime guess.
    dewarded: bool = False
    # True only when its own side destroyed it before its time (an ally's or
    # the owner's attack; never the expiry). Replay parser from analyzer
    # 1.5.0 and OpenDota; older analyzer output cannot tell and leaves False.
    denied: bool = False
    # player_slot of the hero that dewarded or denied it (the owner of a
    # summon that did); None when neither happened, or not by a hero, or
    # unknown.
    killer_slot: int | None = None


class SmokeEvent(BaseModel):
    time: int  # game time of the activation in seconds
    team: str  # "radiant" or "dire"
    player_slots: list[int]  # heroes that received the smoke
    player_slot: int | None = None  # hero that used the smoke; None when unknown
    x: float = 0.0              # smoke activation position (world units)
    y: float = 0.0
    end_time: int = 0           # when smoke breaks
    path: list[list[float]] = []  # [[x1,y1], ...] waypoints (world units)
    # Result code, derived from the counts below with the precedence
    # "traded" (kills and own deaths) > "kill" > "objective" > "caught" (only
    # own deaths) > "no_contact"; "unknown" when the source has no result.
    result: str = "unknown"
    kills_gained: int = 0       # enemy deaths near the group, -5..+30s around the break
    deaths_suffered: int = 0    # own deaths near the group in the same window
    objective_taken: str | None = None  # "tower", "barracks", "ancient", "roshan" or None


class KillEvent(BaseModel):
    """One real hero death (no illusions, no reincarnation/aegis respawns).

    Clarity and mock: one entry per scoreboard death. OpenDota: built from
    players[].kills_log, which has no deaths without a hero killer and no
    positions outside teamfights (see ``confirmed``).
    """
    time: float                     # game time in seconds
    victim_slot: int                # player_slot of the hero that died
    killer_slot: int | None = None  # player_slot of the killing hero; None = no hero killer
    # npc name of the unit that landed the kill when it is not the killer hero
    # itself: a summon of killer_slot ("npc_dota_shadow_shaman_ward_3"), or a
    # tower, creep or neutral when killer_slot is None; "" otherwise/unknown
    killer_unit: str = ""
    # side of the unit that landed the kill: "radiant", "dire" or "neutral"
    # (camp creeps, Roshan, Tormentor); None = the source does not tell. It is
    # what names the side of a unit whose name does not ("dota_fountain"). A
    # deny has the victim's own side. With a credited hero (killer_slot) that
    # hero's side counts: the scoreboard also credits kills a neutral landed.
    killer_team: str | None = None
    x: float | None = None          # victim position (world units); None when unknown
    y: float | None = None
    # player_slots credited with an assist (may be incomplete; [] for OpenDota)
    assist_slots: list[int] = []
    deny: bool = False              # killed by an ally
    # False when the source cannot tell a real death from a reincarnation
    # (OpenDota kills_log of a victim with more entries than scoreboard deaths).
    confirmed: bool = True
    # Game time the victim was alive again (respawn or buyback; the replay
    # parser sees it up to a second late). None = still dead when the data
    # ends, or the source does not tell (OpenDota, analyzer before 1.5.0).
    respawn_time: float | None = None


class PositionSnapshot(BaseModel):
    """Hero position at a point during a teamfight."""
    x: float
    y: float
    time_offset: float  # seconds from teamfight start


class AbilityCast(BaseModel):
    """A single ability cast during a teamfight."""
    ability_name: str
    ability_key: str = ""
    time_offset: float  # seconds from teamfight start
    x: float
    y: float


class TeamfightPlayer(BaseModel):
    player_slot: int
    hero_name: str
    hero_slug: str = ""
    x: float
    y: float
    damage_dealt: int
    damage_taken: int
    healing: int = 0
    kills: int
    deaths: int
    abilities_used: list[str]  # ability keys cast during the fight
    positions: list[PositionSnapshot] = []
    ability_casts: list[AbilityCast] = []
    # Seconds from teamfight start to this player's (first) death in the
    # fight; None when the player survived. Filled by every source.
    death_time_offset: float | None = None
    # [{"target": <victim PlayerData.hero_name>, "target_slot": int,
    #   "time_offset": float | None}] - one entry per hero this player killed
    kill_targets: list[dict] = []
    gold_delta: int = 0
    xp_delta: int = 0
    item_uses: dict[str, int] = {}  # {item key: uses} during the fight
    buybacks: int = 0


class TeamfightEvent(BaseModel):
    # Game time in seconds. The window is padded by 15 s around the first and
    # last death (OpenDota, analyzer schema 3, mock); older analyzer output
    # starts at the first death and ends at the last.
    start_time: int
    end_time: int
    radiant_deaths: int
    dire_deaths: int
    # Radiant minus Dire sum of the players' gold_delta / xp_delta (gold and
    # XP gained from kills and creeps minus gold lost on death, inside the
    # window); positive = radiant advantage
    gold_delta: int = 0
    xp_delta: int = 0
    players: list[TeamfightPlayer]


class AbilityEvent(BaseModel):
    player_slot: int
    hero_name: str
    ability_name: str
    ability_key: str = ""  # internal key for CDN icon lookup
    time: int
    target: str | None = None
    x: float | None = None
    y: float | None = None


class LaningSnapshot(BaseModel):
    """Hero state snapshot during laning phase (every 5 seconds, 0-600s)."""
    time: int = 0
    x: float = 0.0
    y: float = 0.0
    hp: int = 0
    max_hp: int = 0
    mana: int = 0
    max_mana: int = 0
    level: int = 0
    last_hits: int = 0
    denies: int = 0
    net_worth: int = 0


class LaningCreepKill(BaseModel):
    """A single last-hit or deny event during laning."""
    time: int = 0
    creep_type: str = ""  # "melee", "ranged", "siege", "neutral" or "unknown"
    gold: int = 0


class LaningDamageEvent(BaseModel):
    """Hero-vs-hero damage event during laning."""
    time: int = 0
    target_idx: int = 0  # index of the target in match.players (0-9), NOT player_slot
    damage: int = 0
    inflictor: str = ""  # ability/item key; "attack" for a plain attack


class LaningRegenEvent(BaseModel):
    """Regen item usage during laning."""
    time: int = 0
    item: str = ""  # item key


class PlayerData(BaseModel):
    player_slot: int  # 0-4 Radiant, 128-132 Dire
    hero_id: int      # Valve hero id; 0 when unknown
    # Display name from heroes.json ("Wraith King"). A hero that is missing
    # from the catalog keeps its prettified internal name ("Largo"); when
    # nothing identifies it the code "hero_<id>" or "unknown_<player index>"
    # is used (the frontend translates those two forms).
    hero_name: str
    hero_slug: str = ""  # npc name without "npc_dota_hero_" (CDN icon name); "" when unknown
    # Steam persona name; "" when anonymous/unknown (the frontend shows a
    # localized "Player N" from the slot).
    player_name: str
    team: str  # "radiant" or "dire"
    kills: int
    deaths: int
    assists: int
    level: int
    last_hits: int
    denies: int
    gold_per_min: int
    xp_per_min: int
    hero_damage: int
    tower_damage: int
    # Valve's post-game value (the replay's match record, OpenDota). It also
    # credits what the in-game scoreboard leaves out, e.g. the regeneration
    # Urn of Shadows gives an ally.
    hero_healing: int
    net_worth: int              # real net worth when the source provides it
    # Purchases in time order, including the starting items (time <= 0).
    # OpenDota does not log strategy-time purchases on current game builds:
    # its starting items can be incomplete, the replay parser's are not.
    items: list[ItemEvent]
    backpack: list[str]         # item keys in the backpack at game end
    final_items: list[str]      # display names of the end-of-game inventory
    final_item_keys: list[str] = []  # item keys of the 6 main slots at game end
    neutral_item: str = ""      # item key of the neutral item slot, "" if none/unknown
    skills: list[SkillEvent]
    wards: list[WardEvent]
    ability_events: list[AbilityEvent]

    # Per-minute timelines: index m = value at minute m, from minute 0 to
    # floor(duration / 60); nothing after the game ended. [] when unknown.
    gold_t: list[int] = []      # total gold earned at each minute
    xp_t: list[int] = []        # total xp at each minute
    lh_t: list[int] = []        # last hits at each minute
    # Denies at each minute: the whole series from OpenDota and from replay
    # analyzer output of schema 3; older analyzer output only has the laning
    # phase (minutes 0-10).
    dn_t: list[int] = []

    # Lane data
    # lane is ABSOLUTE (OpenDota convention): 1=bottom, 2=mid, 3=top, 4=jungle, 0=unknown
    lane: int = 0
    # lane_role is RELATIVE to the team: 1=safe, 2=mid, 3=off, 4=jungle/roaming, 0=unknown
    lane_role: int = 0
    is_roaming: bool = False
    # Share of the gold available in lane that was earned by 10:00
    # (gold_t[10] / 4948, OpenDota's formula); 0.0 when gold_t is too short.
    lane_efficiency: float = 0.0
    lane_efficiency_pct: float = 0.0    # the same value as 0-100, rounded

    # Kill and damage breakdowns. Hero keys are the PlayerData.hero_name
    # (display name) of a hero in this match, from every source.
    # killed / killed_by hold heroes only (a death to a tower or creep is in
    # MatchData.kills_log with its killer_unit). damage / damage_taken never
    # list the hero itself; OpenDota also lists other units under the unit
    # name it has for them ("npc_dota_creep_badguys_melee", "dota_fountain",
    # "illusion_npc_dota_hero_viper"), the replay parser heroes only.
    killed: dict[str, int] = {}         # {hero_name: count} - heroes this player killed
    killed_by: dict[str, int] = {}      # {hero_name: count} - heroes that killed this player
    damage: dict[str, int] = {}         # damage dealt to each unit
    damage_taken: dict[str, int] = {}   # damage received from each unit
    # Damage by ability/item key; a plain attack is "attack".
    damage_inflictor: dict[str, int] = {}

    # Economy reasons (numeric reason ids as string keys, OpenDota numbering)
    gold_reasons: dict[str, int] = {}   # {reason: amount}
    xp_reasons: dict[str, int] = {}     # {reason: amount}

    # Benchmarks
    benchmarks: dict[str, dict] = {}    # {stat: {raw, pct}}

    # Runes (rune type ids as string keys, OpenDota numbering)
    runes_log: list[dict] = []          # [{time, key}]
    runes: dict[str, int] = {}          # {rune_type: count}

    # Additional stats
    stuns: float = 0.0                  # seconds of stun dealt (scoreboard)
    actions_per_min: int = 0
    # Share of the team's kills this player had a part in (0-1). The game's
    # own counter when the source has it, else (kills + assists) / team kills.
    teamfight_participation: float = 0.0
    buyback_count: int = 0
    buyback_log: list[dict] = []        # [{time}] one entry per buyback, in time order
    obs_placed: int = 0                 # wards this player placed (= his entries in ``wards``)
    sen_placed: int = 0
    observer_kills: int = 0
    sentry_kills: int = 0
    lane_kills: int = 0                 # enemy lane creeps killed, siege creeps included
    neutral_kills: int = 0
    ancient_kills: int = 0
    tower_kills: int = 0                # the game's counter: a tower the hero denied counts too
    roshans_killed: int = 0             # Roshan himself (not Roshan's Banner)
    courier_kills: int = 0
    # The game's own announcements, one count each time the value is reached
    # (a triple kill also counts under "2"; a streak of 5 under "3", "4", "5").
    multi_kills: dict[str, int] = {}    # {kills in a row ("2" = double kill): times}
    kill_streaks: dict[str, int] = {}   # {streak length ("3" = killing spree): times}
    # [{time, key[, charges]}] purchase timeline with item keys, in time
    # order: the same purchases as ``items``. Ward stack merges
    # ("ward_dispenser") are not purchases and are left out; recipes
    # ("recipe_*") are listed by the replay parser only.
    purchase_log: list[dict] = []

    # Lane position heatmap
    lane_pos: dict = {}                 # {x: {y: count}} - raw cell data for heatmap

    # Hero position over the whole game, one sample every 5 s from the first
    # spawn to the end: [time_s, x, y, hp_pct] with world-unit ints and
    # hp_pct 0-100 (0 = dead, -1 = unknown). Empty when the source has no
    # positions (OpenDota).
    position_log: list[list[int]] = []

    # Enhanced laning phase data (Clarity and mock only)
    laning_snapshots: list[LaningSnapshot] = []
    laning_creep_kills: list[LaningCreepKill] = []
    laning_denies: list[LaningCreepKill] = []
    laning_damage_events: list[LaningDamageEvent] = []
    laning_regen_events: list[LaningRegenEvent] = []


class GoldXpSnapshot(BaseModel):
    """Team gold/xp at a point in time (for economy graph)."""
    time: int  # game time in seconds
    radiant_gold: int
    dire_gold: int
    radiant_xp: int
    dire_xp: int


class MatchData(BaseModel):
    match_id: int  # full 64-bit match id; 0 when unknown
    duration: int  # seconds from the horn to the end of the game
    radiant_win: bool  # only meaningful when winner_known is True
    # Code from parser.GAME_MODES: "unknown", "all_pick", "captains_mode",
    # "random_draft", "single_draft", "all_random", "intro", "diretide",
    # "reverse_captains_mode", "greeviling", "tutorial", "mid_only",
    # "least_played", "limited_heroes", "compendium_matchmaking", "custom",
    # "captains_draft", "balanced_draft", "ability_draft", "event",
    # "all_random_death_match", "one_v_one_mid", "ranked_all_pick", "turbo",
    # "mutation", "coaches_challenge", "bot_challenge", or "mode_<id>" for an
    # unlisted id.
    game_mode: str
    start_time: int  # unix time of the match start; 0 when unknown
    # The game's team kill counters. A hero that dies to neutral creeps with
    # nobody credited is no kill for the other side, so a score can be lower
    # than the deaths of the other team.
    radiant_score: int
    dire_score: int
    players: list[PlayerData]
    wards: list[WardEvent]
    smokes: list[SmokeEvent]
    teamfights: list[TeamfightEvent]
    ability_events: list[AbilityEvent]
    gold_xp_timeline: list[GoldXpSnapshot] = []  # per-minute snapshots

    # Every real hero death in time order. The primary source for deaths,
    # killers and the score at time T. Empty when the source has none.
    kills_log: list[KillEvent] = []

    # Advantage graphs
    radiant_gold_adv: list[int] = []    # per-minute radiant gold advantage
    radiant_xp_adv: list[int] = []     # per-minute radiant xp advantage

    # Draft: [{is_pick: bool, hero_id: int, team: 0 | 1 | None, order: int}].
    # hero_id is the real hero id (never 0 for a pick of a known hero); team
    # is 0 (radiant) / 1 (dire), or None for a ban that no team owns (all-pick
    # ban phase). ``order`` is the draft order. An all-pick replay lists the
    # picks first, in the order the heroes were locked in (slot order when the
    # replay does not have it), and the bans after them with team None.
    picks_bans: list[dict] = []

    # Objectives in time order: [{time, type, key?, player_slot?, team?, ...}].
    # - "building_kill": ``key`` = the destroyed building's unit name
    #   ("npc_dota_goodguys_tower1_top"), ``building_team`` = its owner,
    #   ``team`` = the side that took it ("radiant"/"dire"), ``player_slot`` =
    #   the killing hero when there is one, ``denied`` = True for a deny.
    # - "CHAT_MESSAGE_ROSHAN_KILL": ``team`` = "radiant"/"dire" (the killers).
    # - "CHAT_MESSAGE_COURIER_LOST": ``team`` = "radiant"/"dire", the side that
    #   LOST the courier; ``player_slot`` = the killing hero when there is one,
    #   ``value`` = the gold bounty.
    # - "CHAT_MESSAGE_FIRSTBLOOD": ``player_slot`` = the killer's player_slot
    #   (absent when unknown), ``key`` = the victim's player_slot as a string
    #   ("128"; absent when unknown).
    # - "CHAT_MESSAGE_AEGIS", "CHAT_MESSAGE_AEGIS_STOLEN",
    #   "CHAT_MESSAGE_DENIED_AEGIS": ``player_slot`` = the hero involved.
    # Other OpenDota objective types are passed through unchanged.
    objectives: list[dict] = []

    # Chat in time order: [{time, type, key, player_slot[, slot][, hero_id]}].
    # type "chat": ``key`` = the message text; type "chatwheel": ``key`` =
    # "chatwheel_<id>" (the analysis resolves the id to its phrase through
    # the chat-wheel catalog) and ``hero_id`` = the hero the phrase names
    # ("<hero> is missing!"; replay parser from analyzer 1.5.0 only).
    chat: list[dict] = []

    # Building status at game end (OpenDota/Valve bit layout, set bit = alive).
    # Towers: bit0-2 top t1-t3, bit3-5 mid t1-t3, bit6-8 bot t1-t3,
    # bit9 ancient top (t4), bit10 ancient bottom (t4).
    # Barracks: bit0 top melee, bit1 top ranged, bit2 mid melee, bit3 mid ranged,
    # bit4 bot melee, bit5 bot ranged.
    tower_status_radiant: int = 0
    tower_status_dire: int = 0
    barracks_status_radiant: int = 0
    barracks_status_dire: int = 0

    # Buildings with real positions (world units) when the source provides them:
    # [{"name": "npc_dota_goodguys_tower1_top", "team": "radiant"|"dire",
    #   "type": "tower"|"barracks"|"fort"|"other", "x": float, "y": float,
    #   "destroyed_time": int | None}]
    # Empty when unknown; consumers then fall back to geo.TOWER_POSITIONS.
    buildings: list[dict] = []

    # True when the replay was truncated/corrupted and only part of it was parsed.
    partial: bool = False
    # False when the winner could not be determined (truncated replay);
    # radiant_win is then meaningless and must not be shown as a result.
    winner_known: bool = True

    # First blood: game time in seconds (0 when it happened before the horn
    # or is unknown)
    first_blood_time: int = 0

    # Server cluster and region (Valve ids, e.g. cluster 151 -> region 5 =
    # Singapore); 0 when unknown. The frontend translates the region id.
    cluster: int = 0
    region: int = 0
    # Game version of the match ("7.39"); "" when the source does not name it:
    # a replay only carries the build number below, and the analysis payload
    # then derives the version from start_time (data_manager.patch_for_time).
    patch: str = ""
    game_build: int = 0  # Dota build of the server ("6762"), replay parser only; 0 when unknown
    # Patch of the fetched game data that named the talents when the match
    # was parsed (``SkillEvent.ability_name``); "" when there was none, and
    # for matches stored before this field existed. Talent keys are reused
    # between patches, so those names are the right ones exactly when this
    # equals the match's own patch (see analyzer.keeps_stored_talent_names).
    talent_names_patch: str = ""
