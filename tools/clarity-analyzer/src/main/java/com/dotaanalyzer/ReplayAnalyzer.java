package com.dotaanalyzer;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import skadistats.clarity.Clarity;
import skadistats.clarity.io.decoder.Decoder;
import skadistats.clarity.io.decoder.LongVarUnsignedDecoder;
import skadistats.clarity.io.s2.S2DecoderFactory;
import skadistats.clarity.model.CombatLogEntry;
import skadistats.clarity.model.DTClass;
import skadistats.clarity.model.Entity;
import skadistats.clarity.model.FieldPath;
import skadistats.clarity.model.StringTable;
import skadistats.clarity.processor.entities.Entities;
import skadistats.clarity.processor.entities.OnEntityCreated;
import skadistats.clarity.processor.entities.OnEntityDeleted;
import skadistats.clarity.processor.entities.OnEntityUpdated;
import skadistats.clarity.processor.entities.UsesEntities;
import skadistats.clarity.processor.gameevents.OnCombatLogEntry;
import skadistats.clarity.processor.reader.OnMessage;
import skadistats.clarity.processor.reader.OnTickStart;
import skadistats.clarity.processor.runner.Context;
import skadistats.clarity.processor.runner.SimpleRunner;
import skadistats.clarity.processor.stringtables.StringTables;
import skadistats.clarity.processor.stringtables.UsesStringTable;
import skadistats.clarity.source.MappedFileSource;
import skadistats.clarity.wire.dota.common.proto.DOTAUserMessages.CDOTAUserMsg_ChatEvent;
import skadistats.clarity.wire.dota.common.proto.DOTAUserMessages.CDOTAUserMsg_ChatMessage;
import skadistats.clarity.wire.dota.common.proto.DOTAUserMessages.CDOTAUserMsg_ChatWheel;
import skadistats.clarity.wire.dota.common.proto.DOTAUserMessages.CDOTAUserMsg_SpectatorPlayerUnitOrders;
import skadistats.clarity.wire.dota.s2.proto.DOTAS2GcMessagesCommon.CMsgDOTAMatch;
import skadistats.clarity.wire.shared.common.proto.CommonNetworkBaseTypes.CNETMsg_Tick;
import skadistats.clarity.wire.shared.demo.proto.Demo.CDemoFileHeader;
import skadistats.clarity.wire.shared.demo.proto.Demo.CDemoFileInfo;
import skadistats.clarity.wire.shared.demo.proto.Demo.CGameInfo;
import skadistats.clarity.wire.shared.s2.proto.S2UserMessages.CUserMessageSayText2;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Dota 2 Replay Analyzer using the Clarity 4.x library.
 * Parses .dem replay files and prints one JSON document (schema_version 3) to stdout.
 * Diagnostics go to stderr; set CLARITY_DEBUG=1 for verbose output.
 *
 * Notes:
 *  - Positions are emitted in Dota 2 world units: world = cell * 128 + vec - 16384
 *    (CBodyComponent.m_cellX/m_vecX), matching backend/app/geo.py.
 *  - Hero entity m_iPlayerID has been observed both as 0..9 and as 0,2,..,18 (doubled).
 *    The scheme is detected from (player id, team) pairs; combat-log assist / chat player
 *    ids are resolved separately at the end of the run (they use their own scheme).
 *  - Game time comes from combat log timestamps (DOTA_COMBATLOG_GAME_STATE 5 = horn).
 *  - Scoreboard values (K/D/A, damage, net worth, gold, XP, last hits, tower / Roshan kills,
 *    team score) are Valve's own counters (CDOTA_PlayerResource, CDOTA_DataRadiant/Dire,
 *    CDOTATeam) frozen at GAME END; the replay keeps running for about two minutes of
 *    post-game that must not leak into them.
 *  - A complete replay ends with Valve's post-game match record (CMsgDOTAMatch). Hero healing,
 *    GPM and XPM are taken from it: the in-game healing counter leaves out what the record
 *    credits (e.g. the regeneration of Urn of Shadows), and the earned-XP counter stops at
 *    the level cap. Without the record (truncated replay) the counters are used.
 *  - Every entity property read is null-safe: a missing property falls back to the
 *    previous heuristic so older/newer replays still produce output.
 *
 * Exit codes: 0 ok (output may carry "partial": true), 64 usage, 65 not a replay /
 * nothing usable, 66 file missing or unreadable, 70 internal error.
 *
 * The state and the resolve steps that ReplayAnalyzerTest drives directly (kill credits,
 * building and ward deaths, teamfight chains, skill-ups) are package-private instead of private.
 */
@UsesEntities
@UsesStringTable("EntityNames")
public class ReplayAnalyzer {

    static final int SCHEMA_VERSION = 3;

    /**
     * Version of the analyzer that wrote an output: `version` of build.gradle, which the build
     * puts into the JAR manifest. "" when the classes do not run from the JAR (unit tests).
     */
    static String analyzerVersion() {
        String version = ReplayAnalyzer.class.getPackage().getImplementationVersion();
        return version != null ? version : "";
    }

    static final int EXIT_OK = 0;
    static final int EXIT_USAGE = 64;
    static final int EXIT_INVALID_REPLAY = 65;
    static final int EXIT_NO_INPUT = 66;
    static final int EXIT_INTERNAL = 70;

    static final boolean DEBUG = "1".equals(System.getenv("CLARITY_DEBUG"))
            || Boolean.getBoolean("dotaanalyzer.debug");

    // ---- Map geometry ----
    static final float CELL_SIZE = 128f;
    static final float WORLD_ORIGIN_OFFSET = 16384f;

    /** A kill counts for a smoke when it happens this close (world units) to the smoked group. */
    static final float SMOKE_KILL_RADIUS = 2500f;
    /** An objective counts for a smoke when it is this close to the smoke path. */
    static final float SMOKE_OBJECTIVE_RADIUS = 3000f;
    /** Two hero deaths belong to the same teamfight when they are less than this many seconds apart. */
    static final int TEAMFIGHT_GAP_S = 15;
    /** A teamfight window starts this long before its first death and ends this long after its last. */
    static final float TEAMFIGHT_PAD_S = 15f;
    /** position_log sampling interval in seconds (game time). */
    static final int POSITION_LOG_INTERVAL_S = 5;
    /** Pre-game length: the horn sounds this long after the pre-game state starts. */
    static final float PRE_GAME_S = 90f;
    /** A scoreboard kill/assist increment belongs to a death at most this many ticks away (normally 0). */
    private static final int SCORE_MATCH_TICKS = 30;
    /** A combat-log ward death belongs to the ward whose life state changed within this many ticks. */
    private static final int WARD_DEATH_MATCH_TICKS = 60;

    static final int TEAM_RADIANT = 2;
    static final int TEAM_DIRE = 3;

    private static final int PID_UNKNOWN = 0;
    private static final int PID_DIRECT = 1;
    private static final int PID_DOUBLED = 2;

    /** dotaunitorder_t DOTA_UNIT_ORDER_TRAIN_ABILITY: the player spends a skill point on an ability. */
    private static final int ORDER_TRAIN_ABILITY = 11;
    /** A train order belongs to a level-up seen at most this many ticks later (normally the same tick). */
    private static final int TRAIN_ORDER_MATCH_TICKS = 30;
    /** Team number of neutral units (camp creeps, Roshan, Tormentor) in the combat log. */
    private static final int TEAM_NEUTRALS = 4;

    private static final int MAX_ABILITY_SLOTS = 35;
    private static final int ABILITY_POLL_TICKS = 15;
    private static final int NEUTRAL_ITEM_SLOT = 16;
    private static final int MAX_ERRORS_LOGGED_PER_HANDLER = 3;
    private static final int MAX_PENDING_HERO_ROWS = 4000;
    /** Strategy-time purchases are in the inventory this many ticks after the hero appeared. */
    private static final int SPAWN_ITEMS_DELAY_TICKS = 30;

    /** Building unit name, e.g. goodguys_tower1_top, badguys_melee_rax_mid, goodguys_fort. */
    static final Pattern BUILDING_RE = Pattern.compile(
            "(goodguys|badguys)_(tower[1-4](?:_(?:top|mid|bot))?|(?:melee|range)_rax_(?:top|mid|bot)|fort)");
    /** Barracks map entity name, e.g. good_rax_melee_top / bad_rax_range_mid. */
    static final Pattern BARRACKS_ENTITY_RE = Pattern.compile("(good|bad)_rax_(melee|range)_(top|mid|bot)");
    /** Demo header server name, e.g. "Valve Dota 2 Singapore Server (srcds1006-sgp2.151.19)". */
    static final Pattern SERVER_CLUSTER_RE = Pattern.compile("\\(srcds\\d+-[^.()]*\\.(\\d+)\\.\\d+\\)");
    /** Demo header game directory, e.g. "/opt/srcds/dota/dota_v6762/dota". */
    static final Pattern GAME_BUILD_RE = Pattern.compile("dota_v(\\d+)");

    // Pre-formatted property names (avoid String.format on hot paths)
    private static final String[] ITEM_SLOT_PROPS = new String[17];
    private static final String[] ABILITY_SLOT_PROPS = new String[MAX_ABILITY_SLOTS];
    private static final String[][] DATA_TEAM_PROPS = new String[5][];
    private static final String[][] PLAYER_RESOURCE_PROPS = new String[10][];
    static {
        for (int i = 0; i < ITEM_SLOT_PROPS.length; i++) {
            ITEM_SLOT_PROPS[i] = "m_hItems." + String.format("%04d", i);
        }
        for (int i = 0; i < MAX_ABILITY_SLOTS; i++) {
            ABILITY_SLOT_PROPS[i] = "m_vecAbilities." + String.format("%04d", i);
        }
        for (int i = 0; i < 5; i++) {
            String p = String.format("m_vecDataTeam.%04d.", i);
            DATA_TEAM_PROPS[i] = new String[]{
                    p + "m_iTotalEarnedGold", p + "m_iTotalEarnedXP",
                    p + "m_iLastHitCount", p + "m_iDenyCount", p + "m_iNetWorth",
                    p + "m_flHeroDamage", p + "m_flTowerDamage", p + "m_fHealing", p + "m_fStuns",
                    p + "m_iTowerKills", p + "m_iRoshanKills",
            };
        }
        for (int i = 0; i < 10; i++) {
            String p = String.format("m_vecPlayerTeamData.%04d.", i);
            PLAYER_RESOURCE_PROPS[i] = new String[]{
                    p + "m_iKills", p + "m_iDeaths", p + "m_iAssists",
                    p + "m_nSelectedHeroID", p + "m_unPickOrder",
                    String.format("m_vecPlayerData.%04d.m_iPlayerTeam", i),
                    p + "m_flTeamFightParticipation",
            };
        }
    }

    /** Server cluster -> region id (OpenDota constants/cluster.json; region ids as in constants/region.json). */
    static final Map<Integer, Integer> CLUSTER_REGION = new HashMap<>();
    static {
        int[][] table = {
                {111, 1}, {112, 1}, {113, 1}, {114, 1}, {117, 1}, {118, 1},
                {121, 2}, {122, 2}, {123, 2}, {124, 2},
                {131, 3}, {132, 3}, {133, 3}, {134, 3}, {135, 3}, {136, 3}, {137, 3}, {138, 3},
                {141, 19}, {142, 19}, {143, 19}, {144, 19}, {145, 19},
                {151, 5}, {152, 5}, {153, 5}, {154, 5}, {155, 5}, {156, 5},
                {161, 6}, {162, 6}, {163, 6}, {171, 7}, {172, 7},
                {181, 8}, {182, 8}, {183, 8}, {184, 8}, {185, 8}, {186, 8}, {187, 8}, {188, 8},
                {191, 9}, {192, 9}, {193, 9},
                {200, 10}, {201, 10}, {202, 10}, {203, 10}, {204, 10},
                {211, 11}, {212, 11}, {213, 11}, {214, 11},
                {221, 12}, {222, 12}, {223, 18}, {224, 12}, {225, 17}, {227, 20},
                {231, 13}, {232, 25}, {235, 13}, {236, 13},
                {241, 14}, {242, 14}, {251, 15}, {261, 16},
                {271, 3}, {272, 3}, {273, 3}, {274, 3}, {346, 38}, {347, 38},
        };
        for (int[] row : table) CLUSTER_REGION.put(row[0], row[1]);
    }

    // ---- Match-level data ----
    private long matchId = 0;
    private String matchIdSource = "none";
    private int duration = 0;
    private boolean radiantWin = false;
    private boolean winnerKnown = false;
    private int gameMode = 0;
    private long startTime = 0;
    private boolean partial = false;
    private CDemoFileInfo fileInfo = null;
    private String serverName = "";
    private int serverStartTick = 0;
    private int cluster = 0;
    private int gameBuild = 0;

    // ---- Per-player data (indexed 0-9, 0-4 radiant, 5-9 dire) ----
    final PlayerState[] playerStates = new PlayerState[10];

    // ---- Events ----
    final List<WardRecord> wardRecords = new ArrayList<>();
    private final Map<Integer, WardRecord> wardsByEntityIndex = new HashMap<>();
    final List<Map<String, Object>> objectiveEvents = new ArrayList<>();
    private final List<SmokeGroup> smokeGroups = new ArrayList<>();
    private final Map<Integer, SmokeGroup> activeSmokeByPlayer = new HashMap<>();
    final Map<Integer, BuildingRecord> buildingsByEntityIndex = new LinkedHashMap<>();
    /** Combat-log building deaths by unit name; the two tier-4 towers of a team share one name. */
    final Map<String, List<Float>> buildingDeathRaw = new HashMap<>();
    final List<WardDeath> wardDeaths = new ArrayList<>();
    private final List<RawChatEvent> chatEvents = new ArrayList<>();
    private final List<Map<String, Object>> chatMessages = new ArrayList<>();
    /** Unit orders per player-controller id (same id scheme as the hero entities), for APM. */
    private final Map<Integer, Integer> unitOrders = new HashMap<>();
    /** Ability entity index -> tick of the last order to train it (see pollPlayerAbilities). */
    final Map<Integer, Integer> trainOrderTicks = new HashMap<>();
    /** Valve's kill counter per side (CDOTATeam m_iHeroKills): {radiant, dire}, -1 = unread. */
    final int[] teamHeroKills = {-1, -1};
    /** The replay carries Valve's post-game match record (see onMatchRecord). */
    private boolean matchRecordSeen = false;

    // ---- Time-series data ----
    private final Map<Integer, List<Integer>> goldTimeline = new HashMap<>();
    private final Map<Integer, List<Integer>> xpTimeline = new HashMap<>();
    private final Map<Integer, List<Integer>> lhTimeline = new HashMap<>();
    private final Map<Integer, List<Integer>> denyTimeline = new HashMap<>();

    // ---- Real hero deaths (teamfights, smokes, score) ----
    final List<KillEvent> killLog = new ArrayList<>();
    /** Raw times of reincarnation / aegis deaths: they keep a teamfight going but are no deaths. */
    final List<Float> reincarnationRaw = new ArrayList<>();
    /** Scoreboard kill / assist counter increments: {tick, player idx, 0 = kill | 1 = assist}. */
    final List<int[]> scoreIncrements = new ArrayList<>();
    private int unmatchedScoreIncrements = 0;
    /** Skill points seen spent (m_iAbilityPoints went down) that no ability level-up explained. */
    private int unattributedSkillPoints = 0;

    // ---- Clock ----
    float gameStartTime = 0;   // combat log timestamp of the horn (state 5)
    float gameEndTime = 0;     // combat log timestamp of state 6 (post game)
    boolean gameStarted = false;
    boolean gameEnded = false;
    boolean clockAnchored = false;  // gameStartTime is usable as the game-time origin
    private boolean gamePaused = false;
    private float preGameStartRaw = 0; // game rules m_flPreGameStartTime (horn = this + 90 s)
    float lastCombatLogTime = 0;
    private int lastCombatLogTick = 0;
    int currentTick = 0;
    private float secondsPerTick = 1f / 30f;
    private int lastMinuteRecorded = -1;
    private int lastLaneSnapshotSecond = -5;
    private int lastAbilityPollTick = -1000;
    private boolean abilityPollRequested = false;
    private int lastSampleSecond = Integer.MIN_VALUE;
    private float sampleOrigin = 0f;
    // OpenDota's sampling clock (see frameClockSeconds): the server tick and the pause counters.
    private int serverTick = -1;
    private int totalPausedTicks = -1;
    private int pauseStartTick = -1;
    private float lastTickRaw = Float.NaN;  // nowRaw() at the previous tick start
    /** Frame-clock second -> {gold, xp, last hits, denies} per player, last seconds before the horn. */
    private final Map<Integer, int[][]> preHornSamples = new LinkedHashMap<>();
    private int rulesStartTick = -1;
    private int rulesEndTick = -1;
    private float rulesStartRaw = 0;
    private float rulesEndRaw = 0;
    private static final int GAME_STATE_FALLBACK_TICKS = 150;

    // ---- Starting items snapshot ----
    private boolean startingItemsCollected = false;
    private int preGameTick = -1;           // tick at which the game rules entered PRE_GAME (4)
    private boolean spawnItemsDone = false; // strategy-time purchases collected for every hero

    // ---- Ward purchase tracking (fallback owner attribution) ----
    private final List<WardPurchase> recentWardPurchases = new ArrayList<>();

    // ---- Draft tracking ----
    private boolean draftCollected = false;
    private int draftOrder = 0;
    private final List<Map<String, Object>> draftEvents = new ArrayList<>();

    // ---- Hero bookkeeping ----
    private final Map<String, Integer> heroNameMap = new HashMap<>();
    /** Gold / XP / purchase rows of heroes that are not registered yet (replayed once they are). */
    private final List<PendingHeroRow> pendingHeroRows = new ArrayList<>();
    private final Map<Integer, Integer> heroEntityIndexToIdx = new HashMap<>();
    private final Set<Integer> secondaryHeroEntities = new HashSet<>();
    private int entityPidScheme = PID_UNKNOWN;
    private int firstHeroTick = -1;

    // ---- Context / processors ----
    private Context lastCtx = null;
    private Entity gameRulesEntity = null;
    private Entity playerResourceEntity = null;
    private final int[][] lastScoreboard = new int[10][];  // {kills, assists} last seen per player

    // ---- Property access cache (J16) ----
    private static final Object MISSING = new Object();
    private final Map<DTClass, Map<String, Object>> fieldPathCache = new HashMap<>();

    // ---- Diagnostics ----
    private final Map<String, Integer> errorCounts = new LinkedHashMap<>();
    /** parse_errors keys that describe the match; every other key counts a failed handler. */
    private static final Set<String> DIAGNOSTIC_KEYS = Set.of("score_credits", "skill_points");
    private int entityNameHits = 0;
    private int entityNameMisses = 0;

    // ---- Consumable items to exclude from the purchase-based final items fallback ----
    private static final Set<String> CONSUMABLE_ITEMS = new HashSet<>(Arrays.asList(
            "tango", "flask", "clarity", "enchanted_mango", "faerie_fire",
            "ward_observer", "ward_sentry", "ward_dispenser", "dust",
            "smoke_of_deceit", "tome_of_knowledge", "tpscroll",
            "blood_grenade", "observer_and_sentry_wards",
            "famango", "great_famango", "greater_famango", "madstone_bundle", "royal_jelly",
            "foragers_health", "foragers_mana", "foragers_stats"
    ));

    /** Non-skill abilities that never consume an ability point. */
    private static final String[] GENERIC_ABILITY_PREFIXES = {
            "generic_hidden", "ability_", "plus_", "twin_gate_portal_warp", "seasonal_",
            "dota_", "high_five", "abyssal_underlord_portal_warp", "cny_", "frostivus",
    };

    private static final String[] ANCIENT_CREEP_FRAGMENTS = {
            "black_dragon", "black_drake", "granite_golem", "rock_golem", "big_thunder_lizard",
            "small_thunder_lizard", "elder_jungle_stalker", "jungle_stalker", "prowler_acolyte",
            "prowler_shaman", "frostbitten_golem", "ice_shaman", "ancient_frog",
    };

    public ReplayAnalyzer() {
        for (int i = 0; i < 10; i++) {
            playerStates[i] = new PlayerState(i);
        }
    }

    // ======== Diagnostics ========

    private void recordError(String handler, Throwable t) {
        int n = errorCounts.merge(handler, 1, Integer::sum);
        if (n <= MAX_ERRORS_LOGGED_PER_HANDLER) {
            System.err.println("[warn] " + handler + " failed (#" + n + "): " + t);
            if (DEBUG) t.printStackTrace(System.err);
        }
    }

    private static void debug(String msg) {
        if (DEBUG) System.err.println(msg);
    }

    // ======== Property access (cached FieldPaths, null-safe) ========

    private Object prop(Entity e, String name) {
        if (e == null || name == null) return null;
        try {
            DTClass cls = e.getDtClass();
            if (cls == null) return null;
            Map<String, Object> byName = fieldPathCache.get(cls);
            if (byName == null) {
                byName = new HashMap<>();
                fieldPathCache.put(cls, byName);
            }
            Object fp = byName.get(name);
            if (fp == null) {
                FieldPath resolved = null;
                try {
                    resolved = cls.getFieldPathForName(name);
                } catch (Exception ignored) {
                    // unknown property on this class
                }
                fp = resolved != null ? resolved : MISSING;
                byName.put(name, fp);
            }
            if (fp == MISSING) return null;
            return e.getPropertyForFieldPath((FieldPath) fp);
        } catch (Exception ex) {
            return null;
        }
    }

    private Integer intProp(Entity e, String... names) {
        for (String n : names) {
            Object v = prop(e, n);
            if (v instanceof Number) return ((Number) v).intValue();
            if (v instanceof Boolean) return ((Boolean) v) ? 1 : 0;
        }
        return null;
    }

    private Float floatProp(Entity e, String... names) {
        for (String n : names) {
            Object v = prop(e, n);
            if (v instanceof Number) return ((Number) v).floatValue();
        }
        return null;
    }

    private Long longProp(Entity e, String... names) {
        for (String n : names) {
            Object v = prop(e, n);
            if (v instanceof Number) return ((Number) v).longValue();
        }
        return null;
    }

    private Entities entities() {
        return lastCtx != null ? lastCtx.getProcessor(Entities.class) : null;
    }

    private Entity entityByHandle(Integer handle) {
        if (handle == null || handle < 0 || handle == 0xFFFFFF) return null;
        Entities ents = entities();
        if (ents == null) return null;
        try {
            return ents.getByHandle(handle);
        } catch (Exception ex) {
            return null;
        }
    }

    /**
     * Entity designer name from the EntityNames string table, e.g. "item_tango". The index
     * property was renamed between game builds (m_nameStringableIndex until early 2026,
     * m_nameStringTableIndex after); both are read. Hits and misses are counted so that the
     * next rename shows up in parse_errors instead of silently degrading every name.
     */
    private String entityName(Entity e) {
        String name = lookupEntityName(e);
        if (name != null) entityNameHits++;
        else entityNameMisses++;
        return name;
    }

    private String lookupEntityName(Entity e) {
        Integer idx = intProp(e, "m_pEntity.m_nameStringTableIndex", "m_pEntity.m_nameStringableIndex");
        if (idx == null || idx < 0 || lastCtx == null) return null;
        try {
            StringTables tables = lastCtx.getProcessor(StringTables.class);
            if (tables == null) return null;
            StringTable table = tables.forName("EntityNames");
            if (table == null || !table.hasIndex(idx)) return null;
            String name = table.getNameByIndex(idx);
            return name == null || name.isEmpty() ? null : name;
        } catch (Exception ex) {
            return null;
        }
    }

    /** Entity origin in world units (J1). Falls back to the cell centre when m_vecX/Y is missing. */
    private float[] worldPos(Entity e) {
        Integer cx = intProp(e, "CBodyComponent.m_cellX");
        Integer cy = intProp(e, "CBodyComponent.m_cellY");
        if (cx == null || cy == null) return null;
        Float vx = floatProp(e, "CBodyComponent.m_vecX");
        Float vy = floatProp(e, "CBodyComponent.m_vecY");
        float x = cx * CELL_SIZE + (vx != null ? vx : CELL_SIZE / 2) - WORLD_ORIGIN_OFFSET;
        float y = cy * CELL_SIZE + (vy != null ? vy : CELL_SIZE / 2) - WORLD_ORIGIN_OFFSET;
        if (Float.isNaN(x) || Float.isNaN(y)) return null;
        return new float[]{x, y};
    }

    // ======== Clock ========

    /** Best estimate of the current combat-log (server) time, advanced by ticks between entries. */
    private float nowRaw() {
        if (gamePaused || lastCombatLogTick <= 0 || currentTick <= lastCombatLogTick) return lastCombatLogTime;
        float extra = (currentTick - lastCombatLogTick) * secondsPerTick;
        return lastCombatLogTime + Math.min(extra, 3.0f);
    }

    private int toGameTime(float raw) {
        return (int) (clockAnchored ? raw - gameStartTime : raw);
    }

    /** Game time with one decimal. */
    private double toGameTimeTenths(float raw) {
        return Math.round((clockAnchored ? raw - gameStartTime : raw) * 10.0) / 10.0;
    }

    /** OpenDota's game clock: the event time and the horn time are each rounded to whole seconds. */
    private int toRoundedGameTime(float raw) {
        return Math.round(raw) - (clockAnchored ? Math.round(gameStartTime) : 0);
    }

    /**
     * OpenDota's clock at the start of a tick, in whole seconds: the server tick of the frame
     * before (the tick message of the current frame has not been read yet) minus the paused
     * ticks, at 30 ticks a second, rounded. Null when the replay has no tick messages or pause
     * counters.
     */
    private Integer frameClockSeconds() {
        if (serverTick < 0 || totalPausedTicks < 0) return null;
        return frameClockSeconds(serverTick, totalPausedTicks, gamePaused ? pauseStartTick : -1);
    }

    /** pauseStartTick >= 0 while the game is paused: the clock stands at that tick. */
    static int frameClockSeconds(int serverTick, int totalPausedTicks, int pauseStartTick) {
        int tick = pauseStartTick >= 0 ? pauseStartTick : serverTick;
        return Math.round((tick - totalPausedTicks) / 30f);
    }

    private static int teamOfIdx(int idx) {
        return idx < 5 ? TEAM_RADIANT : TEAM_DIRE;
    }

    private static String teamName(int team) {
        return team == TEAM_RADIANT ? "radiant" : team == TEAM_DIRE ? "dire" : null;
    }

    /** Side of a unit in the combat log: teamName plus "neutral" (camp creeps, Roshan, Tormentor). */
    static String unitTeamName(int team) {
        return team == TEAM_NEUTRALS ? "neutral" : teamName(team);
    }

    // ======== Player ID mapping (J13) ========

    private void observeEntityPid(int pid, Integer team) {
        if (entityPidScheme != PID_UNKNOWN) return;
        int detected = PID_UNKNOWN;
        if (pid >= 10 && pid <= 19) {
            detected = PID_DOUBLED;
        } else if (team != null) {
            if (team == TEAM_DIRE && pid >= 5 && pid <= 9) detected = PID_DIRECT;
            else if (team == TEAM_RADIANT && (pid == 6 || pid == 8)) detected = PID_DOUBLED;
            else if (team == TEAM_RADIANT && (pid == 1 || pid == 3)) detected = PID_DIRECT;
        }
        if (detected != PID_UNKNOWN) {
            entityPidScheme = detected;
            debug("Hero entity player-id scheme: " + (detected == PID_DOUBLED ? "doubled" : "direct")
                    + " (pid=" + pid + ", team=" + team + ")");
        }
    }

    /** Map a hero entity m_iPlayerID to our 0-9 index, or -1 when it cannot (yet) be decided. */
    private int entityPidToIdx(int pid, Integer team) {
        if (pid < 0) return -1;
        observeEntityPid(pid, team);
        int scheme = entityPidScheme;
        if (scheme == PID_UNKNOWN) {
            boolean waitedLongEnough = gameStarted
                    || (firstHeroTick >= 0 && currentTick - firstHeroTick > 900);
            if (!waitedLongEnough) return -1;  // defer: heroes are seen again on the next update
            entityPidScheme = PID_DOUBLED;      // scheme observed in current (2024+) replays
            scheme = PID_DOUBLED;
            System.err.println("[warn] Hero player-id scheme undetermined; assuming doubled ids");
        }
        if (scheme == PID_DOUBLED && pid % 2 != 0) return -1;
        int idx = scheme == PID_DOUBLED ? pid / 2 : pid;
        if (idx < 0 || idx >= 10) return -1;
        if (team != null && (team == TEAM_RADIANT || team == TEAM_DIRE)) {
            if ((idx < 5) != (team == TEAM_RADIANT)) return -1;
        }
        return idx;
    }

    /**
     * Combat-log assists, chat and chat events carry their own player ids. Decide (once, at the
     * end of the run) whether those are 0..9 or doubled by checking which scheme keeps every
     * assisting player on the killer's team.
     */
    private int eventPidScheme = PID_UNKNOWN;

    private int resolveEventPidScheme() {
        if (eventPidScheme != PID_UNKNOWN) return eventPidScheme;
        int directVotes = 0, doubledVotes = 0;
        for (KillEvent k : killLog) {
            if (k.assistRaw == null || k.deny) continue;
            boolean killerRadiant = k.targetIdx >= 5;
            for (int id : k.assistRaw) {
                boolean validDirect = killerRadiant ? (id >= 0 && id <= 4) : (id >= 5 && id <= 9);
                boolean validDoubled = id % 2 == 0 && (killerRadiant ? (id >= 0 && id <= 8) : (id >= 10 && id <= 18));
                if (validDirect && !validDoubled) directVotes++;
                if (validDoubled && !validDirect) doubledVotes++;
            }
        }
        eventPidScheme = doubledVotes > directVotes ? PID_DOUBLED : PID_DIRECT;
        debug("Event player-id scheme: " + (eventPidScheme == PID_DOUBLED ? "doubled" : "direct")
                + " (votes direct=" + directVotes + ", doubled=" + doubledVotes + ")");
        return eventPidScheme;
    }

    private int eventPidToIdx(int pid) {
        if (pid < 0) return -1;
        int scheme = resolveEventPidScheme();
        if (scheme == PID_DOUBLED) {
            if (pid % 2 != 0) return -1;
            pid = pid / 2;
        }
        return pid < 10 ? pid : -1;
    }

    // ======== Combat Log Processing ========

    @OnCombatLogEntry
    public void onCombatLog(Context ctx, CombatLogEntry cle) {
        lastCtx = ctx;
        float time;
        try {
            time = cle.getTimestamp();
        } catch (Exception e) {
            recordError("combatlog.timestamp", e);
            return;
        }
        if (time >= lastCombatLogTime) {
            lastCombatLogTime = time;
            lastCombatLogTick = currentTick;
        }

        recordLaningSnapshots(time);

        String typeName = "unknown";
        try {
            if (cle.getType() == null) return;
            typeName = cle.getType().name();
            switch (cle.getType()) {
                case DOTA_COMBATLOG_GAME_STATE:
                    handleGameState(cle, time);
                    break;
                case DOTA_COMBATLOG_DEATH:
                    handleDeath(cle, time);
                    break;
                case DOTA_COMBATLOG_DAMAGE:
                    handleDamage(cle, time);
                    break;
                case DOTA_COMBATLOG_HEAL:
                    handleHeal(cle, time);
                    break;
                case DOTA_COMBATLOG_PURCHASE:
                    handlePurchase(cle, time);
                    break;
                case DOTA_COMBATLOG_ABILITY:
                    handleAbility(cle, time);
                    break;
                case DOTA_COMBATLOG_GOLD:
                    handleGold(cle, time);
                    break;
                case DOTA_COMBATLOG_XP:
                    handleXp(cle, time);
                    break;
                case DOTA_COMBATLOG_BUYBACK:
                    handleBuyback(cle, time);
                    break;
                case DOTA_COMBATLOG_MODIFIER_ADD:
                    handleModifierAdd(cle, time);
                    break;
                case DOTA_COMBATLOG_MODIFIER_REMOVE:
                    handleModifierRemove(cle, time);
                    break;
                case DOTA_COMBATLOG_HERO_LEVELUP:
                    handleHeroLevelUp(cle, time);
                    break;
                case DOTA_COMBATLOG_ITEM:
                    handleItem(cle, time);
                    break;
                case DOTA_COMBATLOG_MULTIKILL:
                    handleKillAnnouncement(cle, true);
                    break;
                case DOTA_COMBATLOG_KILLSTREAK:
                    handleKillAnnouncement(cle, false);
                    break;
                default:
                    break;
            }
        } catch (Exception e) {
            recordError("combatlog." + typeName, e);
        }
    }

    /**
     * Per-minute series (gold_t, xp_t, lh_t, dn_t), sampled exactly where OpenDota samples them
     * so that the values are equal for the same match: at the start of the first tick whose
     * frame clock (see frameClockSeconds) has reached round(horn time) + 60 * minute. That is
     * the state after the first frame of that rounded second, up to a second before the true
     * minute. The replay keeps running after the ancient falls: the series stop there.
     *
     * Without the frame clock the combat-log clock of the previous tick stands in for it.
     */
    private void recordMinuteSeries(float previousTickRaw) {
        if (!gameStarted) rememberPreHornSample();
        if (!gameStarted || gameEnded || lastCtx == null) return;
        try {
            Integer frameClock = frameClockSeconds();
            if (frameClock == null && Float.isNaN(previousTickRaw)) return;
            int clock = frameClock != null ? frameClock : Math.round(previousTickRaw);
            int minute = Math.floorDiv(clock - Math.round(gameStartTime), 60);
            if (minute > lastMinuteRecorded && minute >= 0 && minute < 300) {
                readTeamData(lastCtx);
                lastMinuteRecorded = minute;
                recordMinuteSnapshot(minute);
                captureAllInventories(false);
            }
        } catch (Exception e) {
            recordError("snapshots", e);
        }
    }

    /**
     * Minute 0 of OpenDota is sampled when its clock reaches the ROUNDED horn time, i.e. up to
     * a second before the horn. The horn time is only known once it sounds, so the values of the
     * last seconds before it are kept, one sample at the first tick of each frame-clock second.
     */
    private void rememberPreHornSample() {
        Integer clock = frameClockSeconds();
        if (clock == null || preHornSamples.containsKey(clock)) return;
        int[][] sample = new int[10][];
        for (int i = 0; i < 10; i++) {
            PlayerState ps = playerStates[i];
            sample[i] = new int[]{ps.totalGold, ps.totalXp, ps.lastHits, ps.denies};
        }
        preHornSamples.put(clock, sample);
        if (preHornSamples.size() > 3) preHornSamples.remove(preHornSamples.keySet().iterator().next());
    }

    /**
     * 5-second laning snapshots (0:00-10:00). Called on every tick and combat-log entry so that
     * a sample is taken right when its second starts, not at the next combat-log row.
     */
    private void recordLaningSnapshots(float rawTime) {
        if (!gameStarted || gameEnded || lastCtx == null) return;
        try {
            int gameSec = (int) (rawTime - gameStartTime);
            if (gameSec >= 0 && gameSec <= 600 && gameSec >= lastLaneSnapshotSecond + 5) {
                recordLaningSnapshot(gameSec);
            }
        } catch (Exception e) {
            recordError("snapshots", e);
        }
    }

    private void handleGameState(CombatLogEntry cle, float time) {
        int state = cle.getValue();
        debug("Game state: " + state + " at time " + time);

        // State 5 = DOTA_GAMERULES_STATE_GAME_IN_PROGRESS
        if (state == 5 && !gameStarted) {
            markGameStarted(time);
        }
        // State 6 = DOTA_GAMERULES_STATE_POST_GAME
        if (state == 6 && gameStarted && !gameEnded) {
            markGameEnded(time);
        }
    }

    private void markGameStarted(float rawTime) {
        gameStarted = true;
        clockAnchored = true;
        gameStartTime = rawTime;
        spawnItemsDone = true;
        debug("Game started at combat log time " + rawTime);
        snapshotStartingItems();
    }

    /**
     * Game end (state 6): freeze everything the scoreboard shows. The recording continues
     * through post-game, where gold, net worth and inventories still change.
     */
    private void markGameEnded(float rawTime) {
        debug("Game ended at combat log time " + rawTime);
        if (lastCtx != null) {
            readTeamData(lastCtx);
            if (playerResourceEntity != null) updatePlayerResource(playerResourceEntity);
        }
        gameEnded = true;
        gameEndTime = rawTime;
        sampleHeroes(rawTime);
        captureAllInventories(true);
        pollAbilityLevels();
    }

    /**
     * Snapshot each hero's inventory at game start to capture starting items.
     * Slots 0-5 = main inventory, 6-8 = backpack.
     */
    private void snapshotStartingItems() {
        if (startingItemsCollected || lastCtx == null) return;
        startingItemsCollected = true;
        for (int i = 0; i < 10; i++) {
            PlayerState ps = playerStates[i];
            if (ps.heroEntity == null) continue;
            for (int slot = 0; slot <= 8; slot++) {
                try {
                    String itemName = itemNameForHandle(intProp(ps.heroEntity, ITEM_SLOT_PROPS[slot]));
                    if (itemName != null && !itemName.isEmpty()) {
                        ps.startingItems.add(itemName);
                    }
                } catch (Exception ex) {
                    recordError("starting_items", ex);
                }
            }
            debug("Starting items for player " + i + " (" + ps.heroName + "): " + ps.startingItems);
        }
    }

    /**
     * Purchases made during strategy time have no combat-log row: the items are simply in the
     * inventory when the hero appears. Shortly after pre-game starts, each hero's inventory
     * (minus what the combat log already explains) is added to the purchase log: one row per
     * purchase, i.e. a stack counts current charges / charges per purchase (3 mangoes are 3
     * rows, a 3-charge tango is 1).
     */
    private void collectSpawnItems() {
        boolean allDone = true;
        for (PlayerState ps : playerStates) {
            if (ps.spawnItemsCollected) continue;
            if (ps.heroEntity == null || currentTick - Math.max(preGameTick, ps.registeredTick) < SPAWN_ITEMS_DELAY_TICKS) {
                allDone = false;
                continue;
            }
            ps.spawnItemsCollected = true;
            List<String> held = new ArrayList<>();
            for (int slot = 0; slot <= 8; slot++) {
                Entity item = entityByHandle(intProp(ps.heroEntity, ITEM_SLOT_PROPS[slot]));
                String name = itemName(item);
                if (name == null || name.isEmpty()) continue;
                Integer charges = intProp(item, "m_iCurrentCharges");
                if (name.equals("ward_dispenser")) {
                    // A merged stack: charges are observers, secondary charges sentries.
                    Integer secondary = intProp(item, "m_iSecondaryCharges");
                    for (int k = 0; k < Math.min(charges != null ? charges : 1, 20); k++) held.add("ward_observer");
                    for (int k = 0; k < Math.min(secondary != null ? secondary : 1, 20); k++) held.add("ward_sentry");
                    continue;
                }
                Integer perPurchase = intProp(item, "m_iInitialCharges");
                int n = 1;
                if (charges != null && charges > 1 && perPurchase != null && perPurchase > 0) {
                    n = Math.min((charges + perPurchase - 1) / perPurchase, 20);
                }
                for (int k = 0; k < n; k++) held.add(name);
            }
            // Bought after spawning (already a combat-log row), not during strategy time.
            for (Map<String, Object> logged : ps.purchaseLog) held.remove(logged.get("key"));
            float when = preGameStartRaw > 0 ? preGameStartRaw : nowRaw();
            for (int k = 0; k < held.size(); k++) {
                Map<String, Object> purchase = new LinkedHashMap<>();
                purchase.put("raw_time", when);
                purchase.put("key", held.get(k));
                ps.purchaseLog.add(k, purchase);
            }
            debug("Strategy-time purchases of " + ps.heroName + ": " + held);
        }
        if (allDone) spawnItemsDone = true;
    }

    // Entity class suffix -> item key for names that don't follow CamelCase-to-snake_case.
    // Only a safety net: item keys normally come from the EntityNames table (see entityName).
    private static final Map<String, String> ITEM_NAME_ALIASES = new HashMap<>();
    static {
        ITEM_NAME_ALIASES.put("IronwoodBranch", "branches");
        ITEM_NAME_ALIASES.put("Ward_Dispenser", "ward_dispenser");
        ITEM_NAME_ALIASES.put("Ward_Sentry", "ward_sentry");
        ITEM_NAME_ALIASES.put("Ward_Observer", "ward_observer");
        ITEM_NAME_ALIASES.put("Flask", "flask");
        ITEM_NAME_ALIASES.put("Clarity", "clarity");
        ITEM_NAME_ALIASES.put("Dust", "dust");
        ITEM_NAME_ALIASES.put("BootsOfSpeed", "boots");
        ITEM_NAME_ALIASES.put("StoutShield", "stout_shield");
        ITEM_NAME_ALIASES.put("RingOfProtection", "ring_of_protection");
        ITEM_NAME_ALIASES.put("RingOfRegeneration", "ring_of_regen");
        ITEM_NAME_ALIASES.put("InfusedRaindrop", "infused_raindrop");
        ITEM_NAME_ALIASES.put("MagicWand", "magic_wand");
        ITEM_NAME_ALIASES.put("MagicStick", "magic_stick");
        ITEM_NAME_ALIASES.put("ObserverWard", "ward_observer");
        ITEM_NAME_ALIASES.put("SentryWard", "ward_sentry");
        ITEM_NAME_ALIASES.put("TeleportScroll", "tpscroll");
        ITEM_NAME_ALIASES.put("BlinkDagger", "blink");
        // Class names seen in 2026 replays whose conversion is not the item key.
        ITEM_NAME_ALIASES.put("BootsOfElven", "boots_of_elves");
        ITEM_NAME_ALIASES.put("BootsOfTravel", "travel_boots");
        ITEM_NAME_ALIASES.put("ChainMail", "chainmail");
        ITEM_NAME_ALIASES.put("DustofAppearance", "dust");
        ITEM_NAME_ALIASES.put("Eaglehorn", "eagle");
        ITEM_NAME_ALIASES.put("EmptyBottle", "bottle");
        ITEM_NAME_ALIASES.put("Forage_Health", "foragers_health");
        ITEM_NAME_ALIASES.put("GlovesOfHaste", "gloves");
        ITEM_NAME_ALIASES.put("Guardian_Shell", "defiant_shell");
        ITEM_NAME_ALIASES.put("InvisibilityEdge", "invis_sword");
        ITEM_NAME_ALIASES.put("MantaStyle", "manta");
        ITEM_NAME_ALIASES.put("MaskOfDeath", "lifesteal");
        ITEM_NAME_ALIASES.put("Perseverance", "pers");
        ITEM_NAME_ALIASES.put("PlaneswalkersCloak", "cloak");
        ITEM_NAME_ALIASES.put("PlateMail", "platemail");
        ITEM_NAME_ALIASES.put("Recipe_BootsOfTravel", "recipe_travel_boots");
        ITEM_NAME_ALIASES.put("Recipe_MantaStyle", "recipe_manta");
        ITEM_NAME_ALIASES.put("Recipe_Perseverance", "recipe_pers");
        ITEM_NAME_ALIASES.put("RobeOfMagi", "robe");
        ITEM_NAME_ALIASES.put("SacredRelic", "relic");
        ITEM_NAME_ALIASES.put("SplintMail", "splintmail");
        // More of them, from every item entity of seven 7.41f replays (271 class / key pairs).
        ITEM_NAME_ALIASES.put("CraniumBasher", "basher");
        ITEM_NAME_ALIASES.put("GemOfTrueSight", "gem");
        ITEM_NAME_ALIASES.put("GhostScepter", "ghost");
        ITEM_NAME_ALIASES.put("GreaterCritical", "greater_crit");
        ITEM_NAME_ALIASES.put("GrisGris", "grisgris");
        ITEM_NAME_ALIASES.put("LesserCritical", "lesser_crit");
        ITEM_NAME_ALIASES.put("OrchidMalevolence", "orchid");
        ITEM_NAME_ALIASES.put("RefresherOrb_Shard", "refresher_shard");
        ITEM_NAME_ALIASES.put("SheepStick", "sheepstick");
        ITEM_NAME_ALIASES.put("Recipe_CraniumBasher", "recipe_basher");
        ITEM_NAME_ALIASES.put("Recipe_GreaterCritical", "recipe_greater_crit");
        ITEM_NAME_ALIASES.put("Recipe_LesserCritical", "recipe_lesser_crit");
        ITEM_NAME_ALIASES.put("Recipe_OrchidMalevolence", "recipe_orchid");
        ITEM_NAME_ALIASES.put("Recipe_SheepStick", "recipe_sheepstick");
    }

    /**
     * Convert item entity class name to an item key (fallback when the EntityNames string
     * table is unavailable). "CDOTA_Item_MagicStick" -> "magic_stick".
     */
    static String entityClassToItemName(String dtClassName) {
        if (dtClassName == null) return null;
        String prefix = "CDOTA_Item_";
        if (!dtClassName.startsWith(prefix)) return null;
        String suffix = dtClassName.substring(prefix.length());
        if (suffix.isEmpty()) return null;

        String alias = ITEM_NAME_ALIASES.get(suffix);
        if (alias != null) return alias;

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < suffix.length(); i++) {
            char c = suffix.charAt(i);
            if (Character.isUpperCase(c) && i > 0 && suffix.charAt(i - 1) != '_') {
                sb.append('_');
            }
            sb.append(Character.toLowerCase(c));
        }
        return sb.toString();
    }

    /** Item key ("tango", "blink") for an item entity handle, or null. */
    private String itemNameForHandle(Integer handle) {
        return itemName(entityByHandle(handle));
    }

    private String itemName(Entity item) {
        if (item == null) return null;
        String name = entityName(item);
        if (name != null && name.startsWith("item_")) return name.substring("item_".length());
        try {
            return entityClassToItemName(item.getDtClass().getDtName());
        } catch (Exception ex) {
            return null;
        }
    }

    private void handleDeath(CombatLogEntry cle, float time) {
        String targetRaw = cle.getTargetName();
        String attackerRaw = cle.getAttackerName();
        String target = cleanHeroName(targetRaw);
        String attacker = cleanHeroName(attackerRaw);
        String targetLower = targetRaw != null ? targetRaw.toLowerCase(Locale.ROOT) : "";

        boolean targetIsHero;
        try {
            targetIsHero = cle.hasTargetHero() ? cle.isTargetHero() : targetLower.startsWith("npc_dota_hero_");
        } catch (Exception e) {
            targetIsHero = targetLower.startsWith("npc_dota_hero_");
        }
        boolean targetIsIllusion = false;
        try {
            targetIsIllusion = cle.isTargetIllusion();
        } catch (Exception ignored) {
        }
        boolean reincarnating = false;
        try {
            reincarnating = cle.hasWillReincarnate() && cle.isWillReincarnate();
        } catch (Exception ignored) {
        }

        int targetIdx = heroNameToPlayerIdx(target);
        int attackerIdx = heroNameToPlayerIdx(attacker);
        if (attackerIdx < 0) {
            // Kills by summons (bear, wards...) are credited to the owning hero via the damage source.
            try {
                if (cle.hasDamageSourceName()) {
                    attackerIdx = heroNameToPlayerIdx(cleanHeroName(cle.getDamageSourceName()));
                }
            } catch (Exception ignored) {
            }
        }
        int attackerTeam = 0;
        try {
            if (cle.hasAttackerTeam()) attackerTeam = cle.getAttackerTeam();
        } catch (Exception ignored) {
        }
        if (attackerTeam != TEAM_RADIANT && attackerTeam != TEAM_DIRE && attackerTeam != TEAM_NEUTRALS && attackerIdx >= 0) {
            attackerTeam = teamOfIdx(attackerIdx);
        }

        // ---- Real hero deaths (no illusions, no reincarnation/aegis deaths) ----
        if (targetIsHero && !targetIsIllusion && targetIdx >= 0 && reincarnating) {
            reincarnationRaw.add(time);
        }
        if (targetIsHero && !targetIsIllusion && targetIdx >= 0 && !reincarnating) {
            int targetTeam = teamOfIdx(targetIdx);
            boolean deny = attackerTeam == targetTeam;
            PlayerState victim = playerStates[targetIdx];
            victim.deaths++;

            KillEvent ke = new KillEvent(time, currentTick, attackerIdx, targetIdx, deny);
            ke.killerUnit = attackerRaw != null && !attackerRaw.equals("dota_unknown") ? attackerRaw : "";
            ke.attackerTeam = attackerTeam;
            if (victim.hasPosition) {
                ke.x = victim.lastX;
                ke.y = victim.lastY;
                ke.hasPos = true;
            } else {
                try {
                    if (cle.hasLocationX() && cle.hasLocationY()) {
                        ke.x = cle.getLocationX();
                        ke.y = cle.getLocationY();
                        ke.hasPos = true;
                    }
                } catch (Exception ignored) {
                }
            }

            // killed / killed_by are filled from the credited kill log (resolveKillCredits).
            if (!deny && attackerIdx >= 0 && attackerIdx != targetIdx) playerStates[attackerIdx].kills++;

            if (!deny) {
                try {
                    if (cle.hasAssistPlayers()) {
                        List<Integer> assists = cle.getAssistPlayers();
                        if (assists != null && !assists.isEmpty()) ke.assistRaw = new ArrayList<>(assists);
                    }
                } catch (Exception ignored) {
                }
            }
            killLog.add(ke);
        }

        String targetClean = cleanName(targetRaw).toLowerCase(Locale.ROOT);

        // ---- Buildings (J8: team from the building's own name) ----
        Matcher bm = BUILDING_RE.matcher(targetClean);
        if (bm.find()) {
            String key = bm.group(0);
            String buildingTeam = key.startsWith("goodguys") ? "radiant" : "dire";
            String takerTeam = "radiant".equals(buildingTeam) ? "dire" : "radiant";
            boolean denied = ("radiant".equals(buildingTeam) && attackerTeam == TEAM_RADIANT)
                    || ("dire".equals(buildingTeam) && attackerTeam == TEAM_DIRE);
            String type = key.contains("tower") ? "tower_kill" : key.contains("rax") ? "barracks_kill" : "ancient_kill";

            Map<String, Object> obj = new LinkedHashMap<>();
            obj.put("raw_time", time);
            obj.put("type", type);
            obj.put("key", key);
            obj.put("team", takerTeam);
            obj.put("building_team", buildingTeam);
            obj.put("denied", denied);
            if (attackerIdx >= 0) obj.put("player_slot", playerStates[attackerIdx].playerSlot);
            objectiveEvents.add(obj);
            buildingDeathRaw.computeIfAbsent("npc_dota_" + key, k -> new ArrayList<>()).add(time);
            // Like the game's own counter (m_iTowerKills), a denied tower counts for the denier.
            if (attackerIdx >= 0 && type.equals("tower_kill")) playerStates[attackerIdx].towerKills++;
        } else if (isRoshan(targetRaw) && !targetIsHero) {
            Map<String, Object> obj = new LinkedHashMap<>();
            obj.put("raw_time", time);
            obj.put("type", "roshan_kill");
            String team = teamName(attackerTeam);
            if (team != null) obj.put("team", team);
            if (attackerIdx >= 0) {
                obj.put("player_slot", playerStates[attackerIdx].playerSlot);
                playerStates[attackerIdx].roshansKilled++;
                if (playerStates[attackerIdx].hasPosition) {
                    obj.put("_x", playerStates[attackerIdx].lastX);
                    obj.put("_y", playerStates[attackerIdx].lastY);
                }
            }
            objectiveEvents.add(obj);
        } else if (targetClean.contains("courier") && attackerIdx >= 0) {
            playerStates[attackerIdx].courierKills++;
        }

        // ---- Wards (W4): who ended the ward; matched to the ward entity in buildOutput ----
        boolean observerDeath = targetClean.contains("observer_wards");
        if (!targetIsHero && (observerDeath || targetClean.contains("sentry_wards"))) {
            int wardTeam = 0;
            int ownerIdx = -1;
            try {
                if (cle.hasTargetTeam()) wardTeam = cle.getTargetTeam();
                ownerIdx = heroNameToPlayerIdx(cleanHeroName(cle.getTargetSourceName()));
            } catch (Exception ignored) {
            }
            boolean byEnemy = (attackerTeam == TEAM_RADIANT || attackerTeam == TEAM_DIRE)
                    && (wardTeam == TEAM_RADIANT || wardTeam == TEAM_DIRE) && attackerTeam != wardTeam;
            wardDeaths.add(new WardDeath(currentTick, time, observerDeath ? "observer" : "sentry", wardTeam,
                    byEnemy, byEnemy ? attackerIdx : -1, ownerIdx));
        }

        if (targetIsHero || attackerIdx < 0) return;

        // ---- Non-hero kills credited to a hero ----
        PlayerState ps = playerStates[attackerIdx];
        int attackerSide = teamOfIdx(attackerIdx);
        int targetTeamNum = 0;
        try {
            if (cle.hasTargetTeam()) targetTeamNum = cle.getTargetTeam();
        } catch (Exception ignored) {
        }
        if (targetClean.contains("observer_wards") && targetTeamNum != attackerSide) ps.observerKills++;
        if (targetClean.contains("sentry_wards") && targetTeamNum != attackerSide) ps.sentryKills++;

        boolean laneCreep = targetClean.contains("creep_goodguys") || targetClean.contains("creep_badguys")
                || targetClean.contains("goodguys_siege") || targetClean.contains("badguys_siege");
        boolean neutral = targetClean.startsWith("neutral_") || targetClean.contains("_neutral_");
        boolean creepIsRadiant = targetClean.contains("goodguys");
        boolean deny = laneCreep && creepIsRadiant == (attackerIdx < 5);

        if (laneCreep && !deny) ps.laneKills++;
        if (neutral) {
            ps.neutralKills++;
            for (String frag : ANCIENT_CREEP_FRAGMENTS) {
                if (targetClean.contains(frag)) {
                    ps.ancientKills++;
                    break;
                }
            }
        }

        // ---- Laning phase last hits / denies (0-600s) ----
        if (!gameStarted || !(laneCreep || neutral)) return;
        float gameTime = time - gameStartTime;
        if (gameTime < 0 || gameTime > 600) return;

        String creepType = "unknown";
        if (targetClean.contains("melee") || targetClean.contains("flagbearer")) creepType = "melee";
        else if (targetClean.contains("ranged")) creepType = "ranged";
        else if (targetClean.contains("siege")) creepType = "siege";
        else if (neutral) creepType = "neutral";

        Map<String, Object> event = new LinkedHashMap<>();
        event.put("time", Math.round(gameTime));
        event.put("creep_type", creepType);
        event.put("gold", Math.max(0, cle.getValue()));
        if (deny) {
            ps.laningDenies.add(event);
        } else {
            ps.laningCreepKills.add(event);
        }
    }

    /**
     * Roshan himself. Other units carry his name too and must not count as a Roshan kill:
     * Roshan's Banner (npc_dota_unit_roshans_banner, dropped by every second and later Roshan,
     * "dies" when it expires) and the event Roshans (npc_dota_roshan_halloween, ...).
     */
    static boolean isRoshan(String unitName) {
        return "npc_dota_roshan".equals(unitName);
    }

    /** Player index of the hero behind a combat-log row: the damage source (owner of a summon), else the attacker. */
    private int dealerIdx(CombatLogEntry cle) {
        int idx = -1;
        try {
            if (cle.hasDamageSourceName()) idx = heroNameToPlayerIdx(cleanHeroName(cle.getDamageSourceName()));
        } catch (Exception ignored) {
        }
        return idx >= 0 ? idx : heroNameToPlayerIdx(cleanHeroName(cle.getAttackerName()));
    }

    /**
     * Combat-log damage maps. The dealer is the damage SOURCE, so summons (wards, golems, forge
     * spirits) count for their hero; self damage (Soul Ring, Blood Grenade) and damage to allies
     * do not count, as on the scoreboard.
     */
    private void handleDamage(CombatLogEntry cle, float time) {
        String targetRaw = cle.getTargetName();
        String target = cleanHeroName(targetRaw);
        int value = cle.getValue();
        String inflictor = cleanName(cle.getInflictorName());
        // A plain attack has no inflictor; Clarity names that "dota_unknown".
        if (inflictor.isEmpty() || inflictor.equals("dota_unknown")) inflictor = "attack";

        int attackerIdx = dealerIdx(cle);
        int targetIdx = heroNameToPlayerIdx(target);

        boolean targetIsHero;
        try {
            targetIsHero = cle.isTargetHero();
        } catch (Exception e) {
            targetIsHero = targetRaw != null && targetRaw.contains("npc_dota_hero_");
        }
        boolean targetIsIllusion = false;
        try {
            targetIsIllusion = cle.isTargetIllusion();
        } catch (Exception ignored) {
        }

        if (attackerIdx >= 0 && targetIdx >= 0 && targetIsHero && !targetIsIllusion
                && teamOfIdx(attackerIdx) != teamOfIdx(targetIdx)) {
            PlayerState dealer = playerStates[attackerIdx];
            dealer.heroDamage += value;
            dealer.damageDealt.merge(target, value, Integer::sum);
            playerStates[targetIdx].damageTaken.merge(dealer.heroName, value, Integer::sum);
            dealer.damageInflictor.merge(inflictor, value, Integer::sum);

            dealer.heroDamageDealtLog.add(new float[]{time, value});
            playerStates[targetIdx].heroDamageTakenLog.add(new float[]{time, value});

            if (gameStarted) {
                float gameTime = time - gameStartTime;
                if (gameTime >= 0 && gameTime <= 600) {
                    Map<String, Object> evt = new LinkedHashMap<>();
                    evt.put("time", Math.round(gameTime));
                    evt.put("target_idx", targetIdx);
                    evt.put("damage", value);
                    evt.put("inflictor", inflictor);
                    dealer.laningDamageEvents.add(evt);
                }
            }
        }

        if (attackerIdx >= 0 && BUILDING_RE.matcher(cleanName(targetRaw).toLowerCase(Locale.ROOT)).find()) {
            playerStates[attackerIdx].towerDamage += value;
        }
    }

    /**
     * Healing done to real heroes. The scoreboard total counts only OTHER heroes (no lifesteal,
     * no own regeneration items); the per-teamfight healing keeps self heals, like OpenDota's.
     */
    private void handleHeal(CombatLogEntry cle, float time) {
        int value = cle.getValue();
        int srcIdx = dealerIdx(cle);
        if (srcIdx < 0) return;

        boolean targetIsHero;
        try {
            targetIsHero = cle.isTargetHero();
        } catch (Exception e) {
            targetIsHero = cle.getTargetName() != null && cle.getTargetName().contains("npc_dota_hero_");
        }
        boolean targetIsIllusion = false;
        try {
            targetIsIllusion = cle.isTargetIllusion();
        } catch (Exception ignored) {
        }
        int targetIdx = heroNameToPlayerIdx(cleanHeroName(cle.getTargetName()));

        if (targetIsHero && !targetIsIllusion) {
            if (targetIdx != srcIdx) playerStates[srcIdx].heroHealing += value;
            playerStates[srcIdx].heroHealLog.add(new float[]{time, value});
        }
    }

    private void handlePurchase(CombatLogEntry cle, float time) {
        String item = cleanName(cle.getValueName());
        // Not a purchase: the log emits this row whenever observer and sentry stacks merge.
        if (item.equals("ward_dispenser")) return;
        int buyerIdx = heroNameToPlayerIdx(cleanHeroName(cle.getTargetName()));
        if (buyerIdx < 0) {
            deferHeroRow(PendingHeroRow.PURCHASE, cle.getTargetName(), time, 0, item);
            return;
        }
        applyPurchase(buyerIdx, time, item);
    }

    private void applyPurchase(int buyerIdx, float time, String item) {
        Map<String, Object> purchase = new LinkedHashMap<>();
        purchase.put("raw_time", time);  // converted to game time in buildOutput()
        purchase.put("key", item);
        playerStates[buyerIdx].purchaseLog.add(purchase);

        if (item.equals("ward_observer")) {
            recentWardPurchases.add(new WardPurchase(buyerIdx, time, "observer"));
        } else if (item.equals("ward_sentry")) {
            recentWardPurchases.add(new WardPurchase(buyerIdx, time, "sentry"));
        }
    }

    /**
     * Heroes are registered only once the player-id scheme is known (ids 0, 2, 4 are valid in
     * both schemes), so the very first gold / XP / purchase rows can arrive for a hero that has
     * no player index yet. They are kept and replayed when the hero registers.
     */
    private void deferHeroRow(int kind, String heroRaw, float time, int value, String key) {
        if (heroRaw == null || !heroRaw.startsWith("npc_dota_hero_")) return;
        if (pendingHeroRows.size() >= MAX_PENDING_HERO_ROWS) return;
        for (PlayerState ps : playerStates) {
            if (ps.heroEntity == null) {
                pendingHeroRows.add(new PendingHeroRow(kind, cleanHeroName(heroRaw), time, value, key));
                return;
            }
        }
    }

    private void replayPendingHeroRows() {
        if (pendingHeroRows.isEmpty()) return;
        Iterator<PendingHeroRow> it = pendingHeroRows.iterator();
        while (it.hasNext()) {
            PendingHeroRow row = it.next();
            int idx = heroNameToPlayerIdx(row.hero);
            if (idx < 0) continue;
            it.remove();
            if (row.kind == PendingHeroRow.PURCHASE) {
                applyPurchase(idx, row.time, row.key);
            } else if (row.kind == PendingHeroRow.GOLD) {
                applyGold(idx, row.time, row.value, row.key);
            } else {
                applyXp(idx, row.time, row.value, row.key);
            }
        }
    }

    private void handleAbility(CombatLogEntry cle, float time) {
        String caster = cleanHeroName(cle.getAttackerName());
        String ability = cleanName(cle.getInflictorName());
        int casterIdx = heroNameToPlayerIdx(caster);
        if (casterIdx < 0) return;

        Map<String, Object> cast = new LinkedHashMap<>();
        cast.put("name", ability);
        cast.put("raw_time", time);
        playerStates[casterIdx].abilityCasts.add(cast);

        // Fallback skill-build source: an ability used at a higher level implies upgrades.
        try {
            if (cle.hasAbilityLevel()) {
                int abilityLevel = cle.getAbilityLevel();
                if (abilityLevel > 0 && !ability.isEmpty()) {
                    PlayerState ps = playerStates[casterIdx];
                    int prevLevel = ps.abilityLevels.getOrDefault(ability, 0);
                    if (abilityLevel > prevLevel) {
                        for (int lvl = prevLevel + 1; lvl <= abilityLevel; lvl++) {
                            Map<String, Object> upgrade = new LinkedHashMap<>();
                            upgrade.put("name", ability);
                            upgrade.put("level", lvl);
                            upgrade.put("raw_time", time);
                            ps.abilityUpgrades.add(upgrade);
                        }
                        ps.abilityLevels.put(ability, abilityLevel);
                    }
                }
            }
        } catch (Exception ignored) {
            // ability level not available
        }

        if (gameStarted) {
            float gameTimeLaning = time - gameStartTime;
            if (gameTimeLaning >= 0 && gameTimeLaning <= 600 && isRegenItem(ability)) {
                Map<String, Object> evt = new LinkedHashMap<>();
                evt.put("time", Math.round(gameTimeLaning));
                evt.put("item", ability);
                playerStates[casterIdx].laningRegenEvents.add(evt);
            }
        }
    }

    private static boolean isRegenItem(String name) {
        return name != null && (name.contains("tango") || name.contains("flask") || name.contains("clarity")
                || name.contains("mango") || name.contains("bottle") || name.contains("faerie_fire")
                || name.contains("salve"));
    }

    private void handleItem(CombatLogEntry cle, float time) {
        int casterIdx = heroNameToPlayerIdx(cleanHeroName(cle.getAttackerName()));
        if (casterIdx < 0) return;
        String item = cleanName(cle.getInflictorName());
        if (item.isEmpty() || item.equals("dota_unknown")) return;
        PlayerState ps = playerStates[casterIdx];
        ps.itemUseLog.add(new TimedKey(time, item));

        if (!gameStarted) return;
        float gameTime = time - gameStartTime;
        if (gameTime < 0 || gameTime > 600) return;
        if (isRegenItem(item)) {
            Map<String, Object> evt = new LinkedHashMap<>();
            evt.put("time", Math.round(gameTime));
            evt.put("item", item);
            ps.laningRegenEvents.add(evt);
        }
    }

    private void handleGold(CombatLogEntry cle, float time) {
        int reason = cle.hasGoldReason() ? cle.getGoldReason() : 0;
        int idx = heroNameToPlayerIdx(cleanHeroName(cle.getTargetName()));
        if (idx >= 0) {
            applyGold(idx, time, cle.getValue(), String.valueOf(reason));
        } else {
            deferHeroRow(PendingHeroRow.GOLD, cle.getTargetName(), time, cle.getValue(), String.valueOf(reason));
        }
    }

    private void applyGold(int idx, float time, int value, String reason) {
        playerStates[idx].goldReasons.merge(reason, value, Integer::sum);
        playerStates[idx].goldLog.add(new float[]{time, value});
    }

    private void applyXp(int idx, float time, int value, String reason) {
        playerStates[idx].xpReasons.merge(reason, value, Integer::sum);
        playerStates[idx].xpLog.add(new float[]{time, value});
    }

    private void handleXp(CombatLogEntry cle, float time) {
        int reason = cle.hasXpReason() ? cle.getXpReason() : 0;
        int idx = heroNameToPlayerIdx(cleanHeroName(cle.getTargetName()));
        if (idx >= 0) {
            applyXp(idx, time, cle.getValue(), String.valueOf(reason));
        } else {
            deferHeroRow(PendingHeroRow.XP, cle.getTargetName(), time, cle.getValue(), String.valueOf(reason));
        }
    }

    private void handleBuyback(CombatLogEntry cle, float time) {
        int idx = heroNameToPlayerIdx(cleanHeroName(cle.getTargetName()));
        if (idx >= 0) {
            playerStates[idx].buybackRaw.add(time);
        } else if (cle.hasValue()) {
            // Buyback entries usually carry the player id in "value" rather than a target name.
            pendingBuybacks.add(new float[]{cle.getValue(), time});
        }
    }

    /** Buybacks whose player id is resolved at the end of the run: {event player id, raw time}. */
    private final List<float[]> pendingBuybacks = new ArrayList<>();

    /**
     * The game's own multi-kill (2 = double kill, 3 = triple kill ...) and kill-streak (3 =
     * killing spree ...) announcements; one row each time a player reaches the value.
     */
    private void handleKillAnnouncement(CombatLogEntry cle, boolean multiKill) {
        int idx = heroNameToPlayerIdx(cleanHeroName(cle.getAttackerName()));
        int value = cle.getValue();
        if (idx < 0 || value <= 0) return;
        PlayerState ps = playerStates[idx];
        (multiKill ? ps.multiKills : ps.killStreaks).merge(String.valueOf(value), 1, Integer::sum);
    }

    /** Smoke of Deceit applied: join an open group of the same team (within 2s) or start one. */
    private void handleModifierAdd(CombatLogEntry cle, float time) {
        String inflictor = cle.getInflictorName() != null ? cle.getInflictorName() : "";
        if (!inflictor.contains("smoke_of_deceit")) return;
        boolean illusion = false;
        try {
            illusion = cle.isTargetIllusion();
        } catch (Exception ignored) {
        }
        if (illusion) return;
        int idx = heroNameToPlayerIdx(cleanHeroName(cle.getTargetName()));
        if (idx < 0 || activeSmokeByPlayer.containsKey(idx)) return;

        String team = idx < 5 ? "radiant" : "dire";
        SmokeGroup group = null;
        for (int i = smokeGroups.size() - 1; i >= 0; i--) {
            SmokeGroup g = smokeGroups.get(i);
            if (g.team.equals(team) && Math.abs(g.startRaw - time) <= 2.0f) {
                group = g;
                break;
            }
            if (time - g.startRaw > 10f) break;
        }
        PlayerState ps = playerStates[idx];
        if (group == null) {
            // The smoke belongs to the hero that used it, not to the first hero it reached.
            int userIdx = heroNameToPlayerIdx(cleanHeroName(cle.getAttackerName()));
            if (userIdx < 0 || teamOfIdx(userIdx) != teamOfIdx(idx)) userIdx = idx;
            PlayerState user = playerStates[userIdx];
            group = new SmokeGroup(team, time, userIdx);
            group.startX = user.hasPosition ? user.lastX : ps.lastX;
            group.startY = user.hasPosition ? user.lastY : ps.lastY;
            smokeGroups.add(group);
        }
        if (!group.members.contains(idx)) group.members.add(idx);
        activeSmokeByPlayer.put(idx, group);
        debug("Smoke applied to " + ps.heroName + " at " + time);
    }

    /** Smoke of Deceit removed (broken or expired). */
    private void handleModifierRemove(CombatLogEntry cle, float time) {
        String inflictor = cle.getInflictorName() != null ? cle.getInflictorName() : "";
        if (!inflictor.contains("smoke_of_deceit")) return;
        int idx = heroNameToPlayerIdx(cleanHeroName(cle.getTargetName()));
        if (idx < 0) return;
        SmokeGroup group = activeSmokeByPlayer.remove(idx);
        if (group == null) return;
        if (group.endRaw == null || time < group.endRaw) group.endRaw = time;
    }

    private void handleHeroLevelUp(CombatLogEntry cle, float time) {
        int targetIdx = heroNameToPlayerIdx(cleanHeroName(cle.getTargetName()));
        if (targetIdx < 0) return;
        int newLevel = 0;
        try {
            newLevel = cle.getValue();
        } catch (Exception ignored) {
        }
        if (newLevel <= 0) {
            try {
                newLevel = cle.getTargetHeroLevel();
            } catch (Exception ignored) {
            }
        }
        playerStates[targetIdx].levelUpTimes.add(new LevelUpEvent(time, newLevel));
    }

    // ======== Entity Tracking ========

    @OnEntityCreated
    public void onEntityCreated(Context ctx, Entity e) {
        lastCtx = ctx;
        String dtClass;
        try {
            dtClass = e.getDtClass().getDtName();
        } catch (Exception ex) {
            return;
        }
        try {
            if (dtClass.startsWith("CDOTA_Unit_Hero_")) {
                onHeroEntity(e);
            } else if (dtClass.equals("CDOTA_NPC_Observer_Ward") || dtClass.equals("CDOTA_NPC_Observer_Ward_TrueSight")) {
                trackWardCreated(e, dtClass);
            } else if (dtClass.equals("CDOTA_BaseNPC_Tower") || dtClass.equals("CDOTA_BaseNPC_Barracks")
                    || dtClass.equals("CDOTA_BaseNPC_Fort")) {
                trackBuilding(e, dtClass);
            } else if (dtClass.equals("CDOTA_PlayerResource")) {
                playerResourceEntity = e;
                updatePlayerResource(e);
            } else if (dtClass.equals("CDOTATeam")) {
                updateTeamScore(e);
            }
        } catch (Exception ex) {
            recordError("entity.created", ex);
        }
    }

    /** Demo file header: server name (cluster / region), recording start tick, game build. */
    @OnMessage(CDemoFileHeader.class)
    public void onFileHeader(Context ctx, CDemoFileHeader header) {
        try {
            if (header.hasServerName()) {
                serverName = header.getServerName();
                Matcher m = SERVER_CLUSTER_RE.matcher(serverName);
                if (m.find()) cluster = Integer.parseInt(m.group(1));
            }
            if (header.hasServerStartTick()) serverStartTick = Math.max(0, header.getServerStartTick());
            if (header.hasGameDirectory()) {
                Matcher m = GAME_BUILD_RE.matcher(header.getGameDirectory());
                if (m.find()) gameBuild = Integer.parseInt(m.group(1));
            }
            debug("Demo header: server=" + serverName + " cluster=" + cluster + " build=" + gameBuild
                    + " server_start_tick=" + serverStartTick);
        } catch (Exception e) {
            recordError("message.file_header", e);
        }
    }

    @OnEntityDeleted
    public void onEntityDeleted(Context ctx, Entity e) {
        lastCtx = ctx;
        try {
            int index = e.getIndex();
            WardRecord ward = wardsByEntityIndex.remove(index);
            if (ward != null && ward.destroyedRaw == null) {
                // Normally set earlier by the life-state change; the entity goes ~5 s after that.
                ward.destroyedRaw = nowRaw();
                ward.destroyedTick = currentTick;
            }
            BuildingRecord b = buildingsByEntityIndex.get(index);
            if (b != null && b.entityHandle == e.getHandle() && b.destroyedRaw == null && !gameEnded) {
                b.destroyedRaw = nowRaw();
            }
            secondaryHeroEntities.remove(index);
            Integer idx = heroEntityIndexToIdx.get(index);
            if (idx != null) {
                PlayerState ps = playerStates[idx];
                if (ps.heroEntity != null && ps.heroEntityIndex == index) {
                    // Keep the last known entity for end-of-game reads unless a new hero appears.
                    ps.heroEntityDeleted = true;
                }
                heroEntityIndexToIdx.remove(index);
            }
        } catch (Exception ex) {
            recordError("entity.deleted", ex);
        }
    }

    /**
     * Track game rules updates for match ID, game mode, winner. Also a fallback for the
     * horn / end time when the combat log GAME_STATE entries are missing.
     */
    @OnEntityUpdated(classPattern = "CDOTAGamerulesProxy")
    public void onGameRulesUpdated(Context ctx, Entity e, FieldPath[] fps, int count) {
        lastCtx = ctx;
        gameRulesEntity = e;
        try {
            updateGameRules(e);
        } catch (Exception ex) {
            recordError("entity.gamerules", ex);
        }
    }

    @OnEntityUpdated(classPattern = "CDOTA_Unit_Hero_.*")
    public void onHeroUpdated(Context ctx, Entity e, FieldPath[] fps, int count) {
        lastCtx = ctx;
        try {
            onHeroEntity(e);
        } catch (Exception ex) {
            recordError("entity.hero", ex);
        }
    }

    @OnEntityUpdated(classPattern = "CDOTA_PlayerResource")
    public void onPlayerResourceUpdated(Context ctx, Entity e, FieldPath[] fps, int count) {
        lastCtx = ctx;
        playerResourceEntity = e;
        try {
            updatePlayerResource(e);
        } catch (Exception ex) {
            recordError("entity.player_resource", ex);
        }
    }

    /** A talent (or attribute bonus) changed level: poll the skill build on the next tick. */
    @OnEntityUpdated(classPattern = "CDOTA_Ability_Special_Bonus.*")
    public void onTalentUpdated(Context ctx, Entity e, FieldPath[] fps, int count) {
        abilityPollRequested = true;
    }

    @OnEntityUpdated(classPattern = "CDOTA_DataRadiant")
    public void onRadiantDataUpdated(Context ctx, Entity e, FieldPath[] fps, int count) {
        lastCtx = ctx;
        updatePlayerData(e, 0);
    }

    @OnEntityUpdated(classPattern = "CDOTA_DataDire")
    public void onDireDataUpdated(Context ctx, Entity e, FieldPath[] fps, int count) {
        lastCtx = ctx;
        updatePlayerData(e, 5);
    }

    @OnEntityUpdated(classPattern = "CDOTATeam")
    public void onTeamUpdated(Context ctx, Entity e, FieldPath[] fps, int count) {
        lastCtx = ctx;
        updateTeamScore(e);
    }

    @OnEntityUpdated(classPattern = "CDOTA_NPC_Observer_Ward.*")
    public void onWardUpdated(Context ctx, Entity e, FieldPath[] fps, int count) {
        lastCtx = ctx;
        try {
            WardRecord w = wardsByEntityIndex.get(e.getIndex());
            if (w == null) return;
            if (!w.hasPos) {
                float[] pos = worldPos(e);
                if (pos != null) {
                    w.x = pos[0];
                    w.y = pos[1];
                    w.hasPos = true;
                }
            }
            if (w.ownerIdx < 0 || !"entity".equals(w.ownerSource)) {
                int owner = resolveOwnerIdx(e, w.team);
                if (owner >= 0) {
                    w.ownerIdx = owner;
                    w.ownerSource = "entity";
                }
            }
            // The ward dies when its life state leaves 0; the entity is deleted ~5 s later.
            if (w.destroyedRaw == null) {
                Integer life = intProp(e, "m_lifeState");
                if (life != null && life != 0) {
                    w.destroyedRaw = nowRaw();
                    w.destroyedTick = currentTick;
                    w.diedByLifeState = true;
                }
            }
        } catch (Exception ex) {
            recordError("entity.ward", ex);
        }
    }

    @OnEntityUpdated(classPattern = "CDOTA_BaseNPC_(Tower|Barracks|Fort)")
    public void onBuildingUpdated(Context ctx, Entity e, FieldPath[] fps, int count) {
        lastCtx = ctx;
        try {
            BuildingRecord b = buildingsByEntityIndex.get(e.getIndex());
            if (b == null || b.destroyedRaw != null || gameEnded) return;
            Integer life = intProp(e, "m_lifeState");
            if (life != null && life != 0) {
                b.destroyedRaw = nowRaw();
            }
        } catch (Exception ex) {
            recordError("entity.building", ex);
        }
    }

    private void updateGameRules(Entity e) {
        // The fallback for replays without file info (applyFileInfo has the id as well). The
        // value is a Long once the 64-bit decoder is registered (see registerMatchIdDecoder);
        // an Integer is Clarity's default decoding and holds only the low 32 bits.
        Object mid = prop(e, "m_pGameRules.m_unMatchID64");
        if (mid instanceof Number && ((Number) mid).longValue() > 0 && !"file_info".equals(matchIdSource)) {
            matchId = ((Number) mid).longValue();
            matchIdSource = mid instanceof Long ? "game_rules" : "game_rules_32bit";
        }
        Integer paused = intProp(e, "m_pGameRules.m_bGamePaused");
        gamePaused = paused != null && paused != 0;
        Integer pausedTicks = intProp(e, "m_pGameRules.m_nTotalPausedTicks");
        if (pausedTicks != null) totalPausedTicks = pausedTicks;
        Integer pauseStart = intProp(e, "m_pGameRules.m_nPauseStartTick");
        if (pauseStart != null) pauseStartTick = pauseStart;
        Float preGame = floatProp(e, "m_pGameRules.m_flPreGameStartTime");
        if (preGame != null && preGame > 0) preGameStartRaw = preGame;

        Integer mode = intProp(e, "m_pGameRules.m_iGameMode");
        if (mode != null && mode > 0) gameMode = mode;

        Integer winner = intProp(e, "m_pGameRules.m_nGameWinner");
        if (winner != null) {
            if (winner == TEAM_RADIANT) {
                radiantWin = true;
                winnerKnown = true;
            } else if (winner == TEAM_DIRE) {
                radiantWin = false;
                winnerKnown = true;
            }
        }

        // Remember when the game rules entered GAME_IN_PROGRESS (5) / POST_GAME (6). These are
        // only used when the combat log GAME_STATE entry does not arrive (see onTickStart).
        Integer state = intProp(e, "m_pGameRules.m_nGameState");
        if (state != null) {
            if (state == 4 && preGameTick < 0) preGameTick = currentTick;
            if (state == 5 && rulesStartTick < 0) {
                rulesStartTick = currentTick;
                rulesStartRaw = nowRaw();
            }
            if (state == 6 && rulesEndTick < 0) {
                rulesEndTick = currentTick;
                rulesEndRaw = nowRaw();
            }
        }
    }

    // ======== Heroes (J4, J13, J14) ========

    /**
     * Hero entity created/updated. Illusions, clones and Tempest Doubles are ignored so that
     * exactly one primary hero entity per player feeds positions, levels and inventory.
     */
    private void onHeroEntity(Entity e) {
        int entIndex = e.getIndex();
        if (secondaryHeroEntities.contains(entIndex)) return;

        Integer known = heroEntityIndexToIdx.get(entIndex);
        int idx;
        if (known != null) {
            idx = known;
        } else {
            if (isIllusionLike(e)) {
                secondaryHeroEntities.add(entIndex);
                return;
            }
            Integer pid = intProp(e, "m_iPlayerID", "m_nPlayerID");
            if (pid == null || pid < 0) return;
            if (firstHeroTick < 0) firstHeroTick = currentTick;
            idx = entityPidToIdx(pid, intProp(e, "m_iTeamNum"));
            if (idx < 0) return;  // undecided for now; retried on the next update

            PlayerState ps = playerStates[idx];
            if (ps.heroEntity != null && ps.heroEntityIndex != entIndex && !ps.heroEntityDeleted) {
                // A second non-illusion hero for the same player: Meepo clone, Tempest Double...
                secondaryHeroEntities.add(entIndex);
                return;
            }
            assignPrimaryHero(ps, e, idx);
        }
        updateHeroState(playerStates[idx], e);
    }

    private boolean isIllusionLike(Entity e) {
        Integer replicating = intProp(e, "m_hReplicatingOtherHeroModel");
        if (replicating != null) {
            Entity other = entityByHandle(replicating);
            if (other != null && other.getIndex() != e.getIndex()) return true;
        }
        Integer illusionFlag = intProp(e, "m_bIsIllusion");
        return illusionFlag != null && illusionFlag != 0;
    }

    private void assignPrimaryHero(PlayerState ps, Entity e, int idx) {
        if (ps.heroEntity != null) {
            heroEntityIndexToIdx.remove(ps.heroEntityIndex);
        }
        ps.heroEntity = e;
        ps.heroEntityIndex = e.getIndex();
        ps.heroEntityDeleted = false;
        ps.lastAbilityPoints = -1;  // a new entity starts its own skill-point history
        if (ps.registeredTick < 0) ps.registeredTick = currentTick;
        heroEntityIndexToIdx.put(e.getIndex(), idx);

        String npcName = entityName(e);
        String heroName = null;
        if (npcName != null && npcName.startsWith("npc_dota_hero_")) {
            ps.heroNpcName = npcName;
            heroName = npcName.substring("npc_dota_hero_".length());
        } else {
            heroName = entityClassToHeroName(e.getDtClass().getDtName());
        }
        if (heroName != null && !heroName.isEmpty() && !heroName.equals(ps.heroName)) {
            if (ps.heroName != null && !ps.heroName.isEmpty()) {
                heroNameMap.values().removeIf(v -> v == idx);
            }
            ps.heroName = heroName;
            heroNameMap.put(heroName, idx);
        }
        // The hero id comes from the player resource (hero entities do not carry it).
        debug("Primary hero: " + ps.heroName + " (" + ps.heroNpcName + ") -> idx=" + idx);
        replayPendingHeroRows();
    }

    private void updateHeroState(PlayerState ps, Entity e) {
        if (!gameEnded) {
            Integer lvl = intProp(e, "m_iCurrentLevel");
            if (lvl != null && lvl > ps.level) ps.level = lvl;
        }

        // Skill points: every observed decrease is a point the player spent (see pollPlayerAbilities).
        Integer points = intProp(e, "m_iAbilityPoints");
        if (points != null && points >= 0) {
            if (ps.lastAbilityPoints >= 0 && points < ps.lastAbilityPoints) {
                ps.pendingSpentPoints += ps.lastAbilityPoints - points;
                ps.pendingSpentTick = currentTick;
                abilityPollRequested = true;
            }
            ps.lastAbilityPoints = points;
        }

        float[] pos = worldPos(e);
        if (pos != null) {
            ps.lastX = pos[0];
            ps.lastY = pos[1];
            ps.hasPosition = true;
            float now = nowRaw();
            if (now - ps.lastPositionRecordTime >= 2.0f) {
                ps.positionHistory.add(new float[]{now, pos[0], pos[1]});
                ps.lastPositionRecordTime = now;
            }
        }
    }

    /** Per-team scoreboard data (CDOTA_DataRadiant / CDOTA_DataDire). Frozen at game end (W6). */
    private void updatePlayerData(Entity e, int offset) {
        if (gameEnded) return;
        try {
            for (int i = 0; i < 5; i++) {
                String[] props = DATA_TEAM_PROPS[i];
                PlayerState ps = playerStates[offset + i];
                Integer gold = intProp(e, props[0]);
                if (gold != null) ps.totalGold = gold;
                Integer xp = intProp(e, props[1]);
                if (xp != null) ps.totalXp = xp;
                Integer lh = intProp(e, props[2]);
                if (lh != null) ps.lastHits = lh;
                Integer dn = intProp(e, props[3]);
                if (dn != null) ps.denies = dn;
                Integer nw = intProp(e, props[4]);
                if (nw != null && nw > 0) ps.netWorth = nw;
                Float heroDamage = floatProp(e, props[5]);
                Float towerDamage = floatProp(e, props[6]);
                Float healing = floatProp(e, props[7]);
                if (heroDamage != null && towerDamage != null && healing != null) {
                    ps.sbHeroDamage = Math.round(heroDamage);
                    ps.sbTowerDamage = Math.round(towerDamage);
                    ps.sbHealing = Math.round(healing);
                }
                Float stuns = floatProp(e, props[8]);
                if (stuns != null) ps.stuns = stuns;
                Integer towerKills = intProp(e, props[9]);
                if (towerKills != null) ps.sbTowerKills = towerKills;
                Integer roshanKills = intProp(e, props[10]);
                if (roshanKills != null) ps.sbRoshanKills = roshanKills;
            }
        } catch (Exception ex) {
            recordError("entity.team_data", ex);
        }
    }

    /** Team score: Valve's kill counter of a side (CDOTATeam). Frozen at game end (W6). */
    private void updateTeamScore(Entity e) {
        if (gameEnded) return;
        Integer team = intProp(e, "m_iTeamNum");
        Integer kills = intProp(e, "m_iHeroKills");
        if (team == null || kills == null || kills < 0) return;
        if (team == TEAM_RADIANT) teamHeroKills[0] = kills;
        else if (team == TEAM_DIRE) teamHeroKills[1] = kills;
    }

    /**
     * Valve's own scoreboard (CDOTA_PlayerResource, indexed by player 0-9): kills, deaths,
     * assists, teamfight participation, selected hero id and pick order. Frozen at game end
     * (W6). Kill and assist
     * counter increments are remembered so buildOutput can credit them to the right death:
     * the combat-log assist list is incomplete and a kill can be credited without a last hit.
     */
    private void updatePlayerResource(Entity e) {
        if (gameEnded) return;
        for (int i = 0; i < 10; i++) {
            String[] props = PLAYER_RESOURCE_PROPS[i];
            PlayerState ps = playerStates[i];
            // Only trust the entry when player i sits on the side our 0-4 / 5-9 layout expects.
            Integer team = intProp(e, props[5]);
            if (team == null || team != teamOfIdx(i)) continue;

            Integer heroId = intProp(e, props[3]);
            if (heroId != null && heroId > 0) {
                if (ps.heroId <= 0) ps.heroPickTick = currentTick;
                ps.heroId = heroId;
            }
            Integer pickOrder = intProp(e, props[4]);
            if (pickOrder != null && pickOrder > 0) ps.pickOrder = pickOrder;
            Float participation = floatProp(e, props[6]);
            if (participation != null) ps.teamfightParticipation = participation;

            Integer kills = intProp(e, props[0]);
            Integer deaths = intProp(e, props[1]);
            Integer assists = intProp(e, props[2]);
            if (kills == null || deaths == null || assists == null) continue;
            int[] last = lastScoreboard[i];
            if (last != null) {
                for (int n = last[0]; n < kills; n++) scoreIncrements.add(new int[]{currentTick, i, 0});
                for (int n = last[1]; n < assists; n++) scoreIncrements.add(new int[]{currentTick, i, 1});
            }
            lastScoreboard[i] = new int[]{kills, assists};
            ps.sbKills = kills;
            ps.sbDeaths = deaths;
            ps.sbAssists = assists;
        }
    }

    // ======== Wards (J2, J3) ========

    private void trackWardCreated(Entity e, String dtName) {
        boolean isObs = dtName.equals("CDOTA_NPC_Observer_Ward");
        WardRecord w = new WardRecord(e.getIndex(), isObs ? "observer" : "sentry", nowRaw());
        w.ehandle = e.getHandle();
        float[] pos = worldPos(e);
        if (pos != null) {
            w.x = pos[0];
            w.y = pos[1];
            w.hasPos = true;
        }
        w.team = intProp(e, "m_iTeamNum");

        int owner = resolveOwnerIdx(e, w.team);
        if (owner >= 0) {
            w.ownerSource = "entity";
        } else {
            // Last resort: correlate with a recent ward purchase from the ward's team.
            int min = 0, max = 9;
            if (w.team != null && w.team == TEAM_RADIANT) max = 4;
            if (w.team != null && w.team == TEAM_DIRE) min = 5;
            owner = findWardPurchaser(min, max, w.type);
            if (owner >= 0) w.ownerSource = "purchase";
        }
        w.ownerIdx = owner;
        wardRecords.add(w);
        wardsByEntityIndex.put(e.getIndex(), w);

        if (DEBUG && wardRecords.size() <= 5) {
            System.err.println("Ward: type=" + w.type + " x=" + w.x + " y=" + w.y + " team=" + w.team
                    + " owner_idx=" + owner + " via " + w.ownerSource);
        }
    }

    /**
     * Ward owner (J3). The ward's own m_nPlayerOwnerID uses the hero m_iPlayerID scheme and is
     * set for both ward types. m_hOwnerEntity is the fallback: it is the hero for sentries but a
     * CDOTAPlayerController for observers.
     */
    private int resolveOwnerIdx(Entity ward, Integer wardTeam) {
        Integer ownerPid = intProp(ward, "m_nPlayerOwnerID");
        if (ownerPid != null && ownerPid >= 0) {
            int idx = entityPidToIdx(ownerPid, wardTeam);
            if (idx >= 0) return idx;
        }
        Entity owner = entityByHandle(intProp(ward, "m_hOwnerEntity"));
        if (owner == null) return -1;
        Integer idx = heroEntityIndexToIdx.get(owner.getIndex());
        if (idx == null) {
            String cls;
            try {
                cls = owner.getDtClass().getDtName();
            } catch (Exception ex) {
                return -1;
            }
            if (!cls.startsWith("CDOTA_Unit_Hero_") && !cls.equals("CDOTAPlayerController")) return -1;
            Integer pid = intProp(owner, "m_iPlayerID", "m_nPlayerID");
            if (pid == null) return -1;
            Integer ownerTeam = intProp(owner, "m_iTeamNum");
            int resolved = entityPidToIdx(pid, ownerTeam != null ? ownerTeam : wardTeam);
            if (resolved < 0) return -1;
            idx = resolved;
        }
        if (wardTeam != null && (wardTeam == TEAM_RADIANT || wardTeam == TEAM_DIRE) && teamOfIdx(idx) != wardTeam) {
            return -1;
        }
        return idx;
    }

    /**
     * Most recent unconsumed ward purchase (within 60s) by a player in [minIdx, maxIdx].
     * Consumes the purchase entry so it's not reused.
     */
    private int findWardPurchaser(int minIdx, int maxIdx, String wardType) {
        float currentTime = nowRaw();
        int bestIdx = -1;
        int bestPurchaseIdx = -1;
        float bestTimeDiff = Float.MAX_VALUE;

        for (int i = recentWardPurchases.size() - 1; i >= 0; i--) {
            WardPurchase wp = recentWardPurchases.get(i);
            if (wp.consumed) continue;
            if (wp.playerIdx < minIdx || wp.playerIdx > maxIdx) continue;
            if (!wp.wardType.equals(wardType)) continue;

            float timeDiff = Math.abs(currentTime - wp.combatLogTime);
            if (timeDiff < bestTimeDiff && timeDiff < 60.0f) {
                bestTimeDiff = timeDiff;
                bestIdx = wp.playerIdx;
                bestPurchaseIdx = i;
            }
        }
        if (bestPurchaseIdx >= 0) {
            recentWardPurchases.get(bestPurchaseIdx).consumed = true;
        }
        return bestIdx;
    }

    // ======== Buildings (J12) ========

    private void trackBuilding(Entity e, String dtClass) {
        BuildingRecord b = new BuildingRecord();
        b.entityIndex = e.getIndex();
        b.entityHandle = e.getHandle();
        b.type = dtClass.endsWith("Tower") ? "tower" : dtClass.endsWith("Barracks") ? "barracks" : "fort";
        Integer team = intProp(e, "m_iTeamNum");
        b.team = team != null ? teamName(team) : null;

        String rawName = entityName(e);
        String canonical = canonicalBuildingName(rawName);
        b.name = canonical != null ? canonical : (rawName != null ? rawName : "");
        if (b.team == null && canonical != null) {
            b.team = canonical.contains("goodguys") ? "radiant" : "dire";
        }
        float[] pos = worldPos(e);
        if (pos != null) {
            b.x = pos[0];
            b.y = pos[1];
            b.hasPos = true;
        }
        buildingsByEntityIndex.put(e.getIndex(), b);
    }

    /**
     * "dota_goodguys_tower1_top" / "npc_dota_goodguys_tower1_top" -> "npc_dota_goodguys_tower1_top".
     * Barracks map entities are named differently: "good_rax_melee_top" ->
     * "npc_dota_goodguys_melee_rax_top" (the combat-log unit name).
     */
    static String canonicalBuildingName(String raw) {
        if (raw == null) return null;
        String lower = raw.toLowerCase(Locale.ROOT);
        Matcher m = BUILDING_RE.matcher(lower);
        if (m.find()) return "npc_dota_" + m.group(0);
        Matcher rax = BARRACKS_ENTITY_RE.matcher(lower);
        if (rax.find()) {
            return "npc_dota_" + rax.group(1) + "guys_" + rax.group(2) + "_rax_" + rax.group(3);
        }
        return null;
    }

    // ======== Team data / snapshots ========

    private void readTeamData(Context ctx) {
        try {
            Entities ents = ctx.getProcessor(Entities.class);
            Entity radiant = ents.getByDtName("CDOTA_DataRadiant");
            if (radiant != null) updatePlayerData(radiant, 0);
            Entity dire = ents.getByDtName("CDOTA_DataDire");
            if (dire != null) updatePlayerData(dire, 5);
            Iterator<Entity> teams = ents.getAllByPredicate(e -> "CDOTATeam".equals(e.getDtClass().getDtName()));
            while (teams.hasNext()) updateTeamScore(teams.next());
        } catch (Exception ex) {
            recordError("team_data.read", ex);
        }
    }

    private void recordMinuteSnapshot(int minute) {
        // Minute 0: the sample taken when the frame clock reached the rounded horn time, if that
        // was before the horn (see rememberPreHornSample).
        int[][] preHorn = minute == 0 ? preHornSamples.get(Math.round(gameStartTime)) : null;
        for (int i = 0; i < 10; i++) {
            List<Integer> gl = goldTimeline.computeIfAbsent(i, k -> new ArrayList<>());
            List<Integer> xl = xpTimeline.computeIfAbsent(i, k -> new ArrayList<>());
            List<Integer> ll = lhTimeline.computeIfAbsent(i, k -> new ArrayList<>());
            List<Integer> dl = denyTimeline.computeIfAbsent(i, k -> new ArrayList<>());

            // Fill any gaps with last known value
            int lastGold = gl.isEmpty() ? 0 : gl.get(gl.size() - 1);
            int lastXp = xl.isEmpty() ? 0 : xl.get(xl.size() - 1);
            int lastLh = ll.isEmpty() ? 0 : ll.get(ll.size() - 1);
            int lastDn = dl.isEmpty() ? 0 : dl.get(dl.size() - 1);
            while (gl.size() < minute) gl.add(lastGold);
            while (xl.size() < minute) xl.add(lastXp);
            while (ll.size() < minute) ll.add(lastLh);
            while (dl.size() < minute) dl.add(lastDn);

            PlayerState ps = playerStates[i];
            gl.add(preHorn != null ? preHorn[i][0] : ps.totalGold);
            xl.add(preHorn != null ? preHorn[i][1] : ps.totalXp);
            ll.add(preHorn != null ? preHorn[i][2] : ps.lastHits);
            dl.add(preHorn != null ? preHorn[i][3] : ps.denies);
        }
    }

    private void recordLaningSnapshot(int gameSecond) {
        for (int i = 0; i < 10; i++) {
            PlayerState ps = playerStates[i];
            if (ps.heroEntity == null) continue;

            Entity e = ps.heroEntity;
            Map<String, Object> snap = new LinkedHashMap<>();
            snap.put("time", gameSecond);
            snap.put("x", (double) ps.lastX);
            snap.put("y", (double) ps.lastY);

            Integer hp = intProp(e, "m_iHealth");
            Integer maxHp = intProp(e, "m_iMaxHealth");
            Float mana = floatProp(e, "m_flMana");
            Float maxMana = floatProp(e, "m_flMaxMana");

            snap.put("hp", hp != null ? hp : 0);
            snap.put("max_hp", maxHp != null ? maxHp : 0);
            snap.put("mana", mana != null ? Math.round(mana) : 0);
            snap.put("max_mana", maxMana != null ? Math.round(maxMana) : 0);

            snap.put("level", ps.level);
            snap.put("last_hits", ps.lastHits);
            snap.put("denies", ps.denies);
            snap.put("net_worth", ps.netWorth > 0 ? ps.netWorth : ps.totalGold);

            ps.laningSnapshots.add(snap);
        }
        lastLaneSnapshotSecond = gameSecond;
    }

    // ======== Inventory (J6) ========

    private void captureAllInventories(boolean freeze) {
        for (PlayerState ps : playerStates) {
            if (ps.inventoryFrozen) continue;
            try {
                captureInventory(ps);
            } catch (Exception ex) {
                recordError("inventory", ex);
            }
            if (freeze && ps.inventoryCaptured) ps.inventoryFrozen = true;
        }
    }

    private void captureInventory(PlayerState ps) {
        Entity hero = ps.heroEntity;
        if (hero == null || ps.heroEntityDeleted) return;
        boolean sawSlot = false;
        int occupied = 0, resolved = 0;
        List<String> main = new ArrayList<>();
        List<String> backpack = new ArrayList<>();
        for (int slot = 0; slot <= 8; slot++) {
            Integer h = intProp(hero, ITEM_SLOT_PROPS[slot]);
            if (h == null) continue;
            sawSlot = true;
            if (h < 0 || h == 0xFFFFFF) continue;  // empty slot
            occupied++;
            String name = itemNameForHandle(h);
            if (name == null || name.isEmpty()) continue;
            resolved++;
            if (slot < 6) main.add(name);
            else backpack.add(name);
        }
        // Skip torn-down states (item entities already deleted) so a good snapshot survives.
        if (!sawSlot || (occupied > 0 && resolved == 0)) return;
        String neutral = itemNameForHandle(intProp(hero, ITEM_SLOT_PROPS[NEUTRAL_ITEM_SLOT]));
        ps.finalInventory = main;
        ps.finalBackpack = backpack;
        ps.neutralItem = neutral != null ? neutral : "";
        ps.inventoryCaptured = true;
    }

    // ======== Skill build (J7) ========

    @OnTickStart
    public void onTickStart(Context ctx, boolean synthetic) {
        lastCtx = ctx;
        if (synthetic) return;
        try {
            currentTick = ctx.getTick();
            float mpt = ctx.getMillisPerTick();
            if (mpt > 0) secondsPerTick = mpt / 1000f;
        } catch (Exception ex) {
            recordError("tick.clock", ex);
        }
        if (!gameStarted && rulesStartTick >= 0 && currentTick - rulesStartTick > GAME_STATE_FALLBACK_TICKS) {
            System.err.println("[warn] No combat log GAME_STATE for the horn; using the game rules state change");
            markGameStarted(rulesStartRaw);
        }
        if (gameStarted && !gameEnded && rulesEndTick >= 0 && currentTick - rulesEndTick > GAME_STATE_FALLBACK_TICKS) {
            markGameEnded(rulesEndRaw);
        }
        if (!draftCollected) collectDraftFromGameRules();
        if (!spawnItemsDone && preGameTick >= 0) collectSpawnItems();
        // Skill build: polled right after a point was spent or a talent changed (so the order
        // of events is kept), and periodically as a safety net.
        if (abilityPollRequested || currentTick - lastAbilityPollTick >= ABILITY_POLL_TICKS) {
            abilityPollRequested = false;
            lastAbilityPollTick = currentTick;
            pollAbilityLevels();
        }
        if (!gameEnded) {
            float now = nowRaw();
            recordMinuteSeries(lastTickRaw);
            lastTickRaw = now;
            recordLaningSnapshots(now);
            // One hero sample per GAME second, so that position_log entries sit on the second
            // they are labelled with; before the horn the scheduled horn time gives the grid.
            float origin = gameStarted ? gameStartTime : preGameStartRaw > 0 ? preGameStartRaw + PRE_GAME_S : 0f;
            if (origin != sampleOrigin) {
                sampleOrigin = origin;
                lastSampleSecond = Integer.MIN_VALUE;
            }
            int second = (int) Math.floor(now - origin);
            if (second > lastSampleSecond) {
                lastSampleSecond = second;
                sampleHeroes(now);
            }
        }
    }

    /**
     * One sample per second and hero: {raw time, x, y, hp %}. Feeds position_log (W2), which
     * takes every fifth one once the horn time is final. hp % is 0 for a dead hero and -1 when
     * it cannot be read.
     */
    private void sampleHeroes(float now) {
        for (PlayerState ps : playerStates) {
            if (ps.heroEntity == null || !ps.hasPosition) continue;
            float hpPct = -1;
            if (!ps.heroEntityDeleted) {
                Entity e = ps.heroEntity;
                Integer life = intProp(e, "m_lifeState");
                Integer hp = intProp(e, "m_iHealth");
                Integer maxHp = intProp(e, "m_iMaxHealth");
                if (life != null && life != 0) {
                    hpPct = 0;
                } else if (hp != null && maxHp != null && maxHp > 0) {
                    hpPct = hp <= 0 ? 0 : Math.max(1, Math.min(100, Math.round(hp * 100f / maxHp)));
                }
            }
            ps.samples.add(new float[]{now, ps.lastX, ps.lastY, hpPct});
        }
    }

    /**
     * Poll each primary hero's ability entities for m_iLevel increases. Sub-abilities that
     * level together with their parent, Invoker's invoked spells and generic abilities are
     * filtered; see pollPlayerAbilities for the rule that decides what is a skill-up.
     */
    private void pollAbilityLevels() {
        if (lastCtx == null) return;
        for (int i = 0; i < 10; i++) {
            PlayerState ps = playerStates[i];
            if (ps.heroEntity == null || ps.heroEntityDeleted) continue;
            try {
                pollPlayerAbilities(ps);
            } catch (Exception ex) {
                recordError("abilities.poll", ex);
            }
        }
    }

    /**
     * Skill build rule (reproduces Valve's ability_upgrades order):
     *  - A talent (special_bonus_* except special_bonus_attributes) is free: every 0 -> 1 is a pick.
     *  - Anything else, special_bonus_attributes included, is a skill-up only against an observed
     *    decrease of the hero's m_iAbilityPoints (tracked in updateHeroState). Levels 15-17 and
     *    19-22 grant no point and auto-level special_bonus_attributes; innates that scale with
     *    the ultimate and Aghanim-granted levels rise without a point too.
     *  - When several abilities rose for one point, the one the player ordered to train takes
     *    it: sub-abilities and granted abilities level with their parent (Activate Fire Remnant
     *    with Fire Remnant, Minefield Sign with Proximity Mines). Without such an order,
     *    abilities first seen at level 0 (skillable) come before abilities first seen at level
     *    1+ (innates), then by slot.
     *  - One level that the player ordered to train while no point was spent is a skill-up too:
     *    Winter Wyvern's Eldwurm's Edda (7.41) raises the level and the maximum level of a basic
     *    ability by one, through a train order and without a skill point (Arctic Burn 3 -> 4 at
     *    11:34 of match 9037747302). Valve's list has that entry.
     * Without m_iAbilityPoints (older replays) the total is capped by hero level instead.
     */
    private void pollPlayerAbilities(PlayerState ps) {
        List<AbilityTrack> increased = new ArrayList<>();
        List<Integer> newLevels = new ArrayList<>();
        for (int slot = 0; slot < MAX_ABILITY_SLOTS; slot++) {
            Integer h = intProp(ps.heroEntity, ABILITY_SLOT_PROPS[slot]);
            if (h == null) continue;
            AbilityTrack track = ps.abilityByHandle.get(h);
            Entity ab = entityByHandle(h);
            if (ab == null) continue;
            Integer lvl = intProp(ab, "m_iLevel");
            if (lvl == null) continue;
            if (track == null) {
                String name = entityName(ab);
                if (name == null) name = abilityNameFromClass(ab);
                // First sight is the baseline: innate / granted / stolen abilities produce no event.
                ps.abilityByHandle.put(h, new AbilityTrack(name, lvl, slot, ab.getIndex()));
                ps.abilityTrackingWorked = true;
                continue;
            }
            if (lvl > track.level) {
                increased.add(track);
                newLevels.add(lvl);
            } else if (lvl < track.level) {
                track.level = lvl;
            }
        }
        // The point and the level change arrive in the same server tick; an allowance that found
        // no level-up within two polls is stale and must not admit a later auto-level.
        if (ps.pendingSpentPoints > 0 && currentTick - ps.pendingSpentTick > 2 * ABILITY_POLL_TICKS) {
            unattributedSkillPoints += ps.pendingSpentPoints;
            ps.pendingSpentPoints = 0;
        }
        if (!increased.isEmpty()) recordSkillUps(ps, increased, newLevels, nowRaw());
    }

    /**
     * Apply the skill build rule (see pollPlayerAbilities) to the abilities whose level rose
     * since the last poll; newLevels.get(k) is the level increased.get(k) has now.
     */
    void recordSkillUps(PlayerState ps, List<AbilityTrack> increased, List<Integer> newLevels, float now) {
        boolean pointsKnown = ps.lastAbilityPoints >= 0;
        int available = pointsKnown ? ps.pendingSpentPoints : Math.max(0, ps.level - ps.entityUpgradeCount);
        List<Integer> paid = new ArrayList<>();
        for (int k = 0; k < increased.size(); k++) {
            AbilityTrack t = increased.get(k);
            if (isTalent(t.name)) {
                if (t.level == 0) addAbilityUpgrade(ps, t.name, 1, now);
                continue;
            }
            if (!isSkillPointAbility(t.name, ps.heroName)) continue;
            boolean derived = false;
            for (AbilityTrack other : increased) {
                if (other != t && other.name != null && t.name.startsWith(other.name + "_")) {
                    derived = true;
                    break;
                }
            }
            if (!derived) paid.add(k);
        }
        // Stable sort: the trained ability, then skillable abilities (first seen at level 0)
        // before innates, then slot order.
        paid.sort(Comparator.<Integer>comparingInt(k -> wasJustTrained(increased.get(k)) ? 0 : 1)
                .thenComparingInt(k -> increased.get(k).baseLevel > 0 ? 1 : 0));
        for (int k : paid) {
            AbilityTrack t = increased.get(k);
            if (pointsKnown && available <= 0 && newLevels.get(k) == t.level + 1 && wasJustTrained(t)) {
                // Trained without a point: the level Eldwurm's Edda gives. No allowance is used,
                // and the order is spent: it explains this one level.
                addAbilityUpgrade(ps, t.name, t.level + 1, now);
                trainOrderTicks.remove(t.entityIndex);
                continue;
            }
            for (int lvl = t.level + 1; lvl <= newLevels.get(k) && available > 0; lvl++) {
                addAbilityUpgrade(ps, t.name, lvl, now);
                ps.entityUpgradeCount++;
                available--;
            }
        }
        for (int k = 0; k < increased.size(); k++) increased.get(k).level = newLevels.get(k);
        if (pointsKnown) ps.pendingSpentPoints = available;
    }

    /** The player's order to train this ability arrived with (just before) its level-up. */
    private boolean wasJustTrained(AbilityTrack track) {
        Integer orderTick = trainOrderTicks.get(track.entityIndex);
        return orderTick != null && orderTick <= currentTick && currentTick - orderTick <= TRAIN_ORDER_MATCH_TICKS;
    }

    private static void addAbilityUpgrade(PlayerState ps, String name, int level, float rawTime) {
        Map<String, Object> up = new LinkedHashMap<>();
        up.put("name", name);
        up.put("level", level);
        up.put("raw_time", rawTime);
        ps.entityAbilityUpgrades.add(up);
    }

    /** A talent pick: any special_bonus_* ability except the attribute bonus (which costs a point). */
    static boolean isTalent(String name) {
        return name != null && name.startsWith("special_bonus_") && !name.equals("special_bonus_attributes");
    }

    private static boolean isSkillPointAbility(String name, String heroName) {
        if (name == null || name.isEmpty()) return false;
        for (String prefix : GENERIC_ABILITY_PREFIXES) {
            if (name.startsWith(prefix)) return false;
        }
        if (name.endsWith("_release") || name.endsWith("_stop") || name.endsWith("_cancel")) return false;
        if (name.startsWith("invoker_")) {
            // Invoked spells mirror the orb levels; only the orbs (and Invoke) take points.
            return name.equals("invoker_quas") || name.equals("invoker_wex")
                    || name.equals("invoker_exort") || name.equals("invoker_invoke");
        }
        return true;
    }

    /** "CDOTA_Ability_DrowRanger_FrostArrows" -> "drow_ranger_frost_arrows" (fallback only). */
    private static String abilityNameFromClass(Entity ab) {
        try {
            String cls = ab.getDtClass().getDtName();
            String prefix = "CDOTA_Ability_";
            if (!cls.startsWith(prefix)) return null;
            String suffix = cls.substring(prefix.length());
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < suffix.length(); i++) {
                char c = suffix.charAt(i);
                if (Character.isUpperCase(c) && i > 0 && suffix.charAt(i - 1) != '_'
                        && !Character.isUpperCase(suffix.charAt(i - 1))) {
                    sb.append('_');
                }
                sb.append(Character.toLowerCase(c));
            }
            return sb.toString();
        } catch (Exception ex) {
            return null;
        }
    }

    // ======== Teamfight Detection ========

    private List<KillEvent> realDeathsSorted() {
        return killLog.stream()
                .sorted(Comparator.comparingDouble(k -> k.time))
                .collect(Collectors.toList());
    }

    /**
     * Teamfights: chains of hero deaths less than TEAMFIGHT_GAP_S apart with at least 3 real
     * deaths. The gap is measured like OpenDota measures it, in whole seconds of its rounded
     * clock (deaths at 347.5 and 362.3 are 14 s apart there and chain). A reincarnation death
     * keeps the chain alive but is not counted.
     */
    List<Map<String, Object>> detectTeamfights() {
        List<Map<String, Object>> teamfights = new ArrayList<>();
        // {raw time, index into the sorted real deaths or -1 for a reincarnation}
        List<KillEvent> deaths = realDeathsSorted();
        List<float[]> events = new ArrayList<>();
        for (int i = 0; i < deaths.size(); i++) events.add(new float[]{deaths.get(i).time, i});
        for (float t : reincarnationRaw) events.add(new float[]{t, -1});
        events.sort((a, b) -> Float.compare(a[0], b[0]));

        List<KillEvent> current = new ArrayList<>();
        float chainStart = 0, chainEnd = 0;
        boolean open = false;
        for (float[] ev : events) {
            if (open && toRoundedGameTime(ev[0]) - toRoundedGameTime(chainEnd) >= TEAMFIGHT_GAP_S) {
                if (current.size() >= 3) teamfights.add(buildTeamfight(current, chainStart, chainEnd));
                current = new ArrayList<>();
                open = false;
            }
            if (!open) {
                open = true;
                chainStart = ev[0];
            }
            chainEnd = ev[0];
            if (ev[1] >= 0) current.add(deaths.get((int) ev[1]));
        }
        if (current.size() >= 3) teamfights.add(buildTeamfight(current, chainStart, chainEnd));
        return teamfights;
    }

    /**
     * One teamfight. The window is padded like OpenDota's (TEAMFIGHT_PAD_S before the first and
     * after the last death of the chain) so that the approach and the clean-up are part of it;
     * every time_offset is relative to the padded start.
     */
    private Map<String, Object> buildTeamfight(List<KillEvent> kills, float chainStart, float chainEnd) {
        float lastKnown = gameEnded ? gameEndTime : lastCombatLogTime;
        int startGame = toGameTime(chainStart - TEAMFIGHT_PAD_S);
        int endGame = toGameTime(Math.min(chainEnd + TEAMFIGHT_PAD_S, Math.max(lastKnown, chainEnd)));
        // The window covers the whole seconds startGame..endGame, as the emitted bounds say.
        float origin = clockAnchored ? gameStartTime : 0f;
        float start = origin + startGame;
        float end = origin + endGame + 1;

        int radiantDeaths = 0, direDeaths = 0;
        for (KillEvent k : kills) {
            if (k.targetIdx < 5) radiantDeaths++;
            else direDeaths++;
        }

        int radiantGold = 0, direGold = 0, radiantXp = 0, direXp = 0;
        List<Map<String, Object>> tfPlayers = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            PlayerState ps = playerStates[i];
            Map<String, Object> tp = new LinkedHashMap<>();
            tp.put("player_slot", ps.playerSlot);
            tp.put("hero_name", ps.heroName != null ? ps.heroName : "");

            int killsInFight = 0, deathsInFight = 0;
            float deathTime = -1;
            List<Map<String, Object>> killTargets = new ArrayList<>();
            for (KillEvent k : kills) {
                if (k.killerIdx == i && !k.deny) {
                    killsInFight++;
                    Map<String, Object> kt = new LinkedHashMap<>();
                    kt.put("target_slot", playerStates[k.targetIdx].playerSlot);
                    kt.put("time_offset", Math.round((k.time - start) * 10.0) / 10.0);
                    killTargets.add(kt);
                }
                if (k.targetIdx == i) {
                    deathsInFight++;
                    if (deathTime < 0) deathTime = k.time;
                }
            }

            List<Map<String, Object>> positions = new ArrayList<>();
            float firstX = 0, firstY = 0;
            boolean foundFirst = false;
            for (float[] pos : ps.positionHistory) {
                if (pos[0] >= end) break;  // history is chronological
                if (pos[0] >= start) {
                    Map<String, Object> posSnap = new LinkedHashMap<>();
                    posSnap.put("x", (double) pos[1]);
                    posSnap.put("y", (double) pos[2]);
                    posSnap.put("time_offset", Math.round((pos[0] - start) * 10.0) / 10.0);
                    positions.add(posSnap);
                    if (!foundFirst) {
                        firstX = pos[1];
                        firstY = pos[2];
                        foundFirst = true;
                    }
                }
            }
            if (!foundFirst) {
                float[] near = findPositionAtTime(ps, start);
                firstX = near != null ? near[0] : ps.lastX;
                firstY = near != null ? near[1] : ps.lastY;
                Map<String, Object> posSnap = new LinkedHashMap<>();
                posSnap.put("x", (double) firstX);
                posSnap.put("y", (double) firstY);
                posSnap.put("time_offset", 0.0);
                positions.add(posSnap);
            }

            List<Map<String, Object>> abilityCasts = new ArrayList<>();
            for (Map<String, Object> cast : ps.abilityCasts) {
                Object rt = cast.get("_raw");
                if (!(rt instanceof Number)) continue;
                float castRaw = ((Number) rt).floatValue();
                if (castRaw >= start && castRaw < end) {
                    float castX = firstX, castY = firstY;
                    float[] castPos = findPositionAtTime(ps, castRaw);
                    if (castPos != null) {
                        castX = castPos[0];
                        castY = castPos[1];
                    }
                    Map<String, Object> ac = new LinkedHashMap<>();
                    ac.put("name", cast.get("name"));
                    ac.put("key", cast.get("name"));
                    ac.put("time_offset", Math.round((castRaw - start) * 10.0) / 10.0);
                    ac.put("x", (double) castX);
                    ac.put("y", (double) castY);
                    abilityCasts.add(ac);
                }
            }

            int damageDealt = 0, damageTaken = 0, healing = 0;
            for (float[] dmg : ps.heroDamageDealtLog) {
                if (dmg[0] >= start && dmg[0] < end) damageDealt += (int) dmg[1];
            }
            for (float[] dmg : ps.heroDamageTakenLog) {
                if (dmg[0] >= start && dmg[0] < end) damageTaken += (int) dmg[1];
            }
            for (float[] heal : ps.heroHealLog) {
                if (heal[0] >= start && heal[0] < end) healing += (int) heal[1];
            }

            // Gold / XP swing as OpenDota defines it: the sum of the combat-log gold and XP rows
            // in the window (kills, creeps, assists and the gold lost on death; no passive income).
            int goldDelta = 0, xpDelta = 0;
            for (float[] g : ps.goldLog) {
                if (g[0] >= start && g[0] < end) goldDelta += (int) g[1];
            }
            for (float[] x : ps.xpLog) {
                if (x[0] >= start && x[0] < end) xpDelta += (int) x[1];
            }
            if (i < 5) {
                radiantGold += goldDelta;
                radiantXp += xpDelta;
            } else {
                direGold += goldDelta;
                direXp += xpDelta;
            }

            int buybacks = 0;
            for (float t : ps.buybackRaw) {
                if (t >= start && t < end) buybacks++;
            }
            Map<String, Integer> itemUses = new LinkedHashMap<>();
            for (TimedKey use : ps.itemUseLog) {
                if (use.rawTime >= start && use.rawTime < end) itemUses.merge(use.key, 1, Integer::sum);
            }

            tp.put("x", (double) firstX);
            tp.put("y", (double) firstY);
            tp.put("kills", killsInFight);
            tp.put("deaths", deathsInFight);
            tp.put("damage_dealt", damageDealt);
            tp.put("damage_taken", damageTaken);
            tp.put("healing", healing);
            tp.put("gold_delta", goldDelta);
            tp.put("xp_delta", xpDelta);
            tp.put("buybacks", buybacks);
            tp.put("item_uses", itemUses);
            tp.put("kill_targets", killTargets);
            tp.put("positions", positions);
            tp.put("ability_casts", abilityCasts);
            if (deathsInFight > 0 && deathTime >= 0) {
                tp.put("death_time_offset", Math.round((deathTime - start) * 10.0) / 10.0);
            }
            tp.put("abilities_used", abilityCasts.stream()
                    .map(a -> (String) a.get("name"))
                    .distinct()
                    .collect(Collectors.toList()));
            tfPlayers.add(tp);
        }

        Map<String, Object> tf = new LinkedHashMap<>();
        tf.put("start", startGame);
        tf.put("end", endGame);
        tf.put("first_death", toGameTime(kills.get(0).time));
        tf.put("last_death", toGameTime(kills.get(kills.size() - 1).time));
        tf.put("radiant_deaths", radiantDeaths);
        tf.put("dire_deaths", direDeaths);
        tf.put("gold_delta", radiantGold - direGold);
        tf.put("xp_delta", radiantXp - direXp);
        tf.put("players", tfPlayers);
        return tf;
    }

    /** Closest recorded position for a player at a raw combat log time ([x, y]) or null. */
    private float[] findPositionAtTime(PlayerState ps, float rawTime) {
        float[] best = null;
        float bestDiff = Float.MAX_VALUE;
        for (float[] pos : ps.positionHistory) {
            float diff = Math.abs(pos[0] - rawTime);
            if (diff < bestDiff) {
                bestDiff = diff;
                best = pos;
            }
            if (pos[0] > rawTime + 5.0f) break;  // history is chronological
        }
        return best != null ? new float[]{best[1], best[2]} : null;
    }

    // ======== Smokes (J10) ========

    private List<Map<String, Object>> buildSmokes() {
        List<Map<String, Object>> out = new ArrayList<>();
        float lastKnown = gameEnded ? gameEndTime : lastCombatLogTime;
        for (SmokeGroup g : smokeGroups) {
            float endRaw = g.endRaw != null ? g.endRaw : Math.min(g.startRaw + 45f, lastKnown);
            PlayerState creator = playerStates[g.creatorIdx];

            List<List<Float>> path = new ArrayList<>();
            for (float[] pos : creator.positionHistory) {
                if (pos[0] >= g.startRaw && pos[0] <= endRaw) {
                    path.add(Arrays.asList(pos[1], pos[2]));
                }
            }
            float[] endPos = findPositionAtTime(creator, endRaw);
            if (endPos == null) endPos = new float[]{creator.lastX, creator.lastY};
            path.add(Arrays.asList(endPos[0], endPos[1]));

            int killsGained = 0, deathsSuffered = 0;
            for (KillEvent k : killLog) {
                if (k.time < endRaw - 5 || k.time > endRaw + 30) continue;
                if (!nearSmokeGroup(g, path, k)) continue;
                boolean victimOnSmokeTeam = (k.targetIdx < 5) == g.team.equals("radiant");
                if (victimOnSmokeTeam) deathsSuffered++;
                else if (!k.deny) killsGained++;
            }

            String objective = smokeObjective(g, path, endRaw);
            String result;
            if (killsGained > 0 && deathsSuffered > 0) result = "traded";
            else if (killsGained > 0) result = "kill";
            else if (objective != null) result = "objective";
            else if (deathsSuffered > 0) result = "caught";
            else result = "no_contact";

            List<Integer> slots = new ArrayList<>();
            for (int idx : g.members) slots.add(playerStates[idx].playerSlot);

            Map<String, Object> smoke = new LinkedHashMap<>();
            smoke.put("time", toGameTime(g.startRaw));
            smoke.put("end_time", toGameTime(endRaw));
            smoke.put("team", g.team);
            smoke.put("x", (double) g.startX);
            smoke.put("y", (double) g.startY);
            smoke.put("path", path);
            smoke.put("result", result);
            smoke.put("kills_gained", killsGained);
            smoke.put("deaths_suffered", deathsSuffered);
            smoke.put("objective_taken", objective);
            smoke.put("player_slot", creator.playerSlot);
            smoke.put("player_slots", slots);
            out.add(smoke);
        }
        return out;
    }

    private boolean nearSmokeGroup(SmokeGroup g, List<List<Float>> path, KillEvent k) {
        if (!k.hasPos) return false;
        for (List<Float> p : path) {
            if (dist(p.get(0), p.get(1), k.x, k.y) <= SMOKE_KILL_RADIUS) return true;
        }
        for (int idx : g.members) {
            float[] pos = findPositionAtTime(playerStates[idx], k.time);
            if (pos != null && dist(pos[0], pos[1], k.x, k.y) <= SMOKE_KILL_RADIUS) return true;
        }
        return false;
    }

    private String smokeObjective(SmokeGroup g, List<List<Float>> path, float endRaw) {
        for (Map<String, Object> obj : objectiveEvents) {
            Object rt = obj.get("_raw");
            if (!(rt instanceof Number)) continue;
            float t = ((Number) rt).floatValue();
            if (t < endRaw - 5 || t > endRaw + 60) continue;
            if (!g.team.equals(obj.get("team")) || Boolean.TRUE.equals(obj.get("denied"))) continue;
            String type = String.valueOf(obj.get("type"));
            float[] where = null;
            if (type.equals("roshan_kill") && obj.get("_x") instanceof Number) {
                where = new float[]{((Number) obj.get("_x")).floatValue(), ((Number) obj.get("_y")).floatValue()};
            } else if (!type.equals("roshan_kill")) {
                // "building" is the entity that died (see linkObjectivesToBuildings); the bare
                // combat-log key cannot tell the two tier-4 towers apart.
                Object linked = obj.get("building");
                BuildingRecord b = buildingByCanonicalName(linked != null ? (String) linked : "npc_dota_" + obj.get("key"));
                if (b != null && b.hasPos) where = new float[]{b.x, b.y};
            }
            if (where == null) continue;
            for (List<Float> p : path) {
                if (dist(p.get(0), p.get(1), where[0], where[1]) <= SMOKE_OBJECTIVE_RADIUS) {
                    return type.equals("roshan_kill") ? "roshan" : type.equals("tower_kill") ? "tower"
                            : type.equals("barracks_kill") ? "barracks" : "ancient";
                }
            }
        }
        return null;
    }

    private BuildingRecord buildingByCanonicalName(String name) {
        for (BuildingRecord b : buildingsByEntityIndex.values()) {
            if (name.equals(b.name)) return b;
        }
        return null;
    }

    private static float dist(float ax, float ay, float bx, float by) {
        float dx = ax - bx, dy = ay - by;
        return (float) Math.sqrt(dx * dx + dy * dy);
    }

    // ======== Final Items / Skills fallbacks ========

    /** Last 6 non-consumable purchases (fallback when the inventory could not be read). */
    private List<String> computeFinalItemsFromPurchases(PlayerState ps) {
        List<String> finalItems = new ArrayList<>();
        for (int i = ps.purchaseLog.size() - 1; i >= 0 && finalItems.size() < 6; i--) {
            String item = (String) ps.purchaseLog.get(i).get("key");
            if (item != null && !item.isEmpty() && !CONSUMABLE_ITEMS.contains(item)
                    && !item.startsWith("recipe_")) {
                finalItems.add(item);
            }
        }
        Collections.reverse(finalItems);
        return finalItems;
    }

    /**
     * Last-resort skill build: correlate hero level-ups with the next ability cast.
     * Only used when neither ability entities nor cast levels are available.
     */
    private List<Map<String, Object>> computeAbilityUpgradesFromLevelUps(PlayerState ps) {
        List<Map<String, Object>> upgrades = new ArrayList<>();
        int castSearchStart = 0;
        for (LevelUpEvent lu : ps.levelUpTimes) {
            Map<String, Object> upgrade = new LinkedHashMap<>();
            upgrade.put("time", toGameTime(lu.rawTime));
            upgrade.put("level", lu.level);
            String abilityName = "";
            for (int c = castSearchStart; c < ps.abilityCasts.size(); c++) {
                Object rt = ps.abilityCasts.get(c).get("_raw");
                if (!(rt instanceof Number)) continue;
                float castRaw = ((Number) rt).floatValue();
                if (castRaw >= lu.rawTime - 2 && castRaw <= lu.rawTime + 30) {
                    abilityName = (String) ps.abilityCasts.get(c).get("name");
                    castSearchStart = c + 1;
                    break;
                }
                if (castRaw > lu.rawTime + 30) break;
            }
            upgrade.put("name", abilityName);
            upgrades.add(upgrade);
        }
        return upgrades;
    }

    // ======== Draft (J11) ========

    /**
     * Draft source for replays whose CDemoFileInfo has no picks/bans: every all-pick replay,
     * and truncated ones. Bans come from here, picks from the player resource (buildOutput).
     */
    private void collectDraftFromGameRules() {
        try {
            Entity grp = gameRulesEntity;
            if (grp == null) return;
            Integer gameState = intProp(grp, "m_pGameRules.m_nGameState");
            // Wait until strategy time (3) .. post game (6); 7+ are not draft-related states.
            if (gameState == null || gameState < 3 || gameState > 6) return;
            draftCollected = true;

            for (int i = 0; i < 24; i++) {
                Integer heroId = intProp(grp, "m_pGameRules.m_vecNewBannedHeroes." + String.format("%04d", i));
                if (heroId != null && heroId > 0) {
                    Map<String, Object> entry = new LinkedHashMap<>();
                    entry.put("is_pick", false);
                    entry.put("hero_id", heroId);
                    // The vector does not say who nominated the ban: no team (W3).
                    entry.put("team", null);
                    entry.put("order", draftOrder++);
                    draftEvents.add(entry);
                }
            }
            for (int i = 0; i < 24; i++) {
                Integer heroId = intProp(grp, "m_pGameRules.m_SelectedHeroes." + String.format("%04d", i));
                if (heroId != null && heroId > 0) {
                    Map<String, Object> entry = new LinkedHashMap<>();
                    entry.put("is_pick", true);
                    entry.put("hero_id", heroId);
                    entry.put("team", i < 5 ? 0 : 1);
                    entry.put("order", draftOrder++);
                    draftEvents.add(entry);
                }
            }
            debug("Draft collected from game rules: " + draftEvents.size() + " entries");
        } catch (Exception e) {
            recordError("draft.gamerules", e);
        }
    }

    private List<Map<String, Object>> draftFromFileInfo() {
        List<Map<String, Object>> out = new ArrayList<>();
        if (fileInfo == null || !fileInfo.hasGameInfo() || !fileInfo.getGameInfo().hasDota()) return out;
        CGameInfo.CDotaGameInfo dota = fileInfo.getGameInfo().getDota();
        for (int i = 0; i < dota.getPicksBansCount(); i++) {
            CGameInfo.CDotaGameInfo.CHeroSelectEvent ev = dota.getPicksBans(i);
            if (!ev.hasHeroId() || ev.getHeroId() <= 0) continue;
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("is_pick", ev.getIsPick());
            entry.put("hero_id", ev.getHeroId());
            entry.put("team", ev.getTeam() == TEAM_DIRE ? 1 : 0);
            entry.put("order", out.size());
            out.add(entry);
        }
        return out;
    }

    // ======== Chat ========

    @OnMessage(CDOTAUserMsg_ChatMessage.class)
    public void onChatMessage(Context ctx, CDOTAUserMsg_ChatMessage message) {
        try {
            Map<String, Object> chat = new LinkedHashMap<>();
            chat.put("raw_time", nowRaw());
            // channelType 11 = all chat
            chat.put("type", message.getChannelType() == 11 ? "chat" : "chatwheel");
            chat.put("_pid", message.getSourcePlayerId());
            chat.put("key", message.getMessageText());
            chatMessages.add(chat);
        } catch (Exception e) {
            recordError("message.chat", e);
        }
    }

    @OnMessage(CDOTAUserMsg_ChatWheel.class)
    public void onChatWheel(Context ctx, CDOTAUserMsg_ChatWheel message) {
        try {
            Map<String, Object> chat = new LinkedHashMap<>();
            chat.put("raw_time", nowRaw());
            chat.put("type", "chatwheel");
            chat.put("_pid", message.getPlayerId());
            chat.put("key", "chatwheel_" + message.getChatMessageId());
            chatMessages.add(chat);
        } catch (Exception e) {
            recordError("message.chatwheel", e);
        }
    }

    @OnMessage(CUserMessageSayText2.class)
    public void onSayText2(Context ctx, CUserMessageSayText2 message) {
        try {
            Map<String, Object> chat = new LinkedHashMap<>();
            chat.put("raw_time", nowRaw());
            chat.put("type", "chat");
            chat.put("key", message.hasParam2() ? message.getParam2() : "");
            chat.put("player_name", message.hasParam1() ? message.getParam1() : "");
            chatMessages.add(chat);
        } catch (Exception e) {
            recordError("message.saytext", e);
        }
    }

    @OnMessage(CDOTAUserMsg_ChatEvent.class)
    public void onChatEvent(Context ctx, CDOTAUserMsg_ChatEvent message) {
        try {
            if (message.getType() == null) return;
            String type = message.getType().name();
            if (type.equals("CHAT_MESSAGE_RUNE_PICKUP") || type.equals("CHAT_MESSAGE_FIRSTBLOOD")
                    || type.equals("CHAT_MESSAGE_AEGIS") || type.equals("CHAT_MESSAGE_AEGIS_STOLEN")
                    || type.equals("CHAT_MESSAGE_DENIED_AEGIS") || type.equals("CHAT_MESSAGE_COURIER_LOST")) {
                chatEvents.add(new RawChatEvent(type, message.getValue(),
                        message.getPlayerid1(), message.getPlayerid2(), nowRaw()));
            }
        } catch (Exception e) {
            recordError("message.chat_event", e);
        }
    }

    /** Every order a player gives to a unit: the action count behind actions_per_min. */
    @OnMessage(CDOTAUserMsg_SpectatorPlayerUnitOrders.class)
    public void onUnitOrder(Context ctx, CDOTAUserMsg_SpectatorPlayerUnitOrders message) {
        try {
            Entity controller = ctx.getProcessor(Entities.class).getByIndex(message.getEntindex());
            Integer pid = intProp(controller, "m_nPlayerID", "m_iPlayerID");
            if (pid != null) unitOrders.merge(pid, 1, Integer::sum);
            // Which ability a skill point is spent on (the order names the ability entity).
            if (message.getOrderType() == ORDER_TRAIN_ABILITY && message.hasAbilityId()) {
                trainOrderTicks.put(message.getAbilityId(), currentTick);
            }
        } catch (Exception e) {
            recordError("message.unit_order", e);
        }
    }

    /** Server tick of the frame being read (OpenDota's sampling clock, see frameClockSeconds). */
    @OnMessage(CNETMsg_Tick.class)
    public void onNetTick(Context ctx, CNETMsg_Tick message) {
        if (message.hasTick()) serverTick = message.getTick();
    }

    /**
     * Valve's post-game match record, stored at the very end of a complete replay: the numbers
     * the post-game screen and the Web API show. Hero healing, GPM and XPM are taken from it
     * (see the class comment); a player whose hero does not match is left on the counters.
     */
    @OnMessage(CMsgDOTAMatch.class)
    public void onMatchRecord(Context ctx, CMsgDOTAMatch match) {
        try {
            for (CMsgDOTAMatch.Player p : match.getPlayersList()) {
                int slot = p.hasPlayerSlot() ? p.getPlayerSlot() : -1;
                int idx = slot >= 0 && slot < 5 ? slot : slot >= 128 && slot < 133 ? slot - 123 : -1;
                if (idx < 0) continue;
                PlayerState ps = playerStates[idx];
                if (ps.heroId > 0 && p.hasHeroId() && p.getHeroId() != ps.heroId) continue;
                if (p.hasHeroHealing()) ps.recordHealing = p.getHeroHealing();
                if (p.hasGoldPerMin()) ps.recordGpm = p.getGoldPerMin();
                if (p.hasXpPerMin()) ps.recordXpm = p.getXpPerMin();
                matchRecordSeen = true;
            }
        } catch (Exception e) {
            recordError("message.match_record", e);
        }
    }

    // ======== Output ========

    private void convertRawTimes(List<Map<String, Object>> events) {
        for (Map<String, Object> ev : events) {
            Object raw = ev.remove("raw_time");
            if (raw instanceof Number) {
                float r = ((Number) raw).floatValue();
                ev.put("_raw", r);
                ev.put("time", toGameTime(r));
            }
        }
    }

    private static void stripInternalKeys(List<Map<String, Object>> events) {
        for (Map<String, Object> ev : events) {
            ev.keySet().removeIf(k -> k.startsWith("_"));
        }
    }

    /** {radiant, dire} score counted from the kill log (fallback for the team counters). */
    int[] killLogScore() {
        int[] score = new int[2];
        for (KillEvent k : killLog) {
            if (scoresForEnemy(k)) score[k.targetIdx < 5 ? 1 : 0]++;
        }
        return score;
    }

    boolean hasUsableData() {
        if (gameStarted) return true;
        for (PlayerState ps : playerStates) {
            if (ps.heroName != null && !ps.heroName.isEmpty()) return true;
        }
        return false;
    }

    /** Fill player names / steam ids / missing hero names from CDemoFileInfo (J12). */
    private void applyFileInfo() {
        if (fileInfo == null || !fileInfo.hasGameInfo() || !fileInfo.getGameInfo().hasDota()) return;
        CGameInfo.CDotaGameInfo dota = fileInfo.getGameInfo().getDota();
        if (dota.hasMatchId() && dota.getMatchId() > 0) {
            // Authoritative when present; the game rules carry the same id (see updateGameRules).
            matchId = dota.getMatchId();
            matchIdSource = "file_info";
        }
        if (gameMode <= 0 && dota.hasGameMode()) gameMode = dota.getGameMode();
        if (!winnerKnown && dota.hasGameWinner()) {
            if (dota.getGameWinner() == TEAM_RADIANT) {
                radiantWin = true;
                winnerKnown = true;
            } else if (dota.getGameWinner() == TEAM_DIRE) {
                radiantWin = false;
                winnerKnown = true;
            }
        }
        if (dota.hasEndTime() && dota.getEndTime() > 0) {
            // end_time is when the recording stopped. The server started serverStartTick ticks
            // before the first recorded tick, and that moment is the match start_time Valve reports.
            double recorded = fileInfo.hasPlaybackTime() ? fileInfo.getPlaybackTime() : 0;
            double tick = fileInfo.hasPlaybackTicks() && fileInfo.getPlaybackTicks() > 0 && recorded > 0
                    ? recorded / fileInfo.getPlaybackTicks() : secondsPerTick;
            if (recorded <= 0) recorded = lastCombatLogTime;
            startTime = Math.max(0, Math.round(dota.getEndTime() - recorded - serverStartTick * tick));
        }

        int radiantOrdinal = 0, direOrdinal = 0;
        boolean[] assigned = new boolean[10];
        List<CGameInfo.CDotaGameInfo.CPlayerInfo> unmatched = new ArrayList<>();
        for (int i = 0; i < dota.getPlayerInfoCount(); i++) {
            CGameInfo.CDotaGameInfo.CPlayerInfo pi = dota.getPlayerInfo(i);
            int idx = -1;
            String npc = pi.hasHeroName() ? pi.getHeroName() : "";
            String norm = normalizeHeroKey(npc);
            if (!norm.isEmpty()) {
                for (int k = 0; k < 10; k++) {
                    if (!assigned[k] && norm.equals(normalizeHeroKey(playerStates[k].heroName))) {
                        idx = k;
                        break;
                    }
                }
            }
            if (idx < 0) {
                unmatched.add(pi);
                continue;
            }
            assigned[idx] = true;
            applyPlayerInfo(playerStates[idx], pi);
        }
        // Fallback for unmatched entries: order within team.
        for (CGameInfo.CDotaGameInfo.CPlayerInfo pi : unmatched) {
            int team = pi.hasGameTeam() ? pi.getGameTeam() : 0;
            int idx = -1;
            if (team == TEAM_RADIANT) {
                while (radiantOrdinal < 5 && assigned[radiantOrdinal]) radiantOrdinal++;
                if (radiantOrdinal < 5) idx = radiantOrdinal;
            } else if (team == TEAM_DIRE) {
                while (direOrdinal < 5 && assigned[5 + direOrdinal]) direOrdinal++;
                if (direOrdinal < 5) idx = 5 + direOrdinal;
            }
            if (idx < 0) continue;
            assigned[idx] = true;
            applyPlayerInfo(playerStates[idx], pi);
        }
    }

    private void applyPlayerInfo(PlayerState ps, CGameInfo.CDotaGameInfo.CPlayerInfo pi) {
        if (pi.hasPlayerName() && !pi.getPlayerName().isEmpty()) ps.playerName = pi.getPlayerName();
        if (pi.hasSteamid()) ps.steamId = pi.getSteamid();
        String npc = pi.hasHeroName() ? pi.getHeroName() : "";
        if (npc.startsWith("npc_dota_hero_")) {
            // J14: prefer the npc internal name over the class-name conversion.
            ps.heroNpcName = npc;
            ps.heroName = npc.substring("npc_dota_hero_".length());
        }
    }

    static String normalizeHeroKey(String name) {
        if (name == null) return "";
        String n = name.toLowerCase(Locale.ROOT);
        if (n.startsWith("npc_dota_hero_")) n = n.substring("npc_dota_hero_".length());
        return n.replaceAll("[^a-z0-9]", "");
    }

    /**
     * Credit the scoreboard's kill and assist counters to individual deaths (W1).
     * The combat log names the unit that landed the kill and lists some assisting players;
     * Valve's counters can credit a kill without a hero last hit and count assists the list
     * omits. Each counter increment is matched to the nearest death of an enemy hero.
     */
    void resolveKillCredits() {
        for (KillEvent k : killLog) {
            k.killerIdx = k.attackerIdx;
        }
        // Kill increments: the death this player landed, else one without a hero killer, else any.
        List<int[]> open = new ArrayList<>();
        for (int[] inc : scoreIncrements) {
            if (inc[2] == 0) open.add(inc);
        }
        for (int pass = 0; pass < 3 && !open.isEmpty(); pass++) {
            Iterator<int[]> it = open.iterator();
            while (it.hasNext()) {
                int[] inc = it.next();
                KillEvent best = null;
                int bestDt = SCORE_MATCH_TICKS + 1;
                for (KillEvent k : killLog) {
                    if (k.deny || k.killCredited || (k.targetIdx < 5) == (inc[1] < 5)) continue;
                    if (pass == 0 && k.attackerIdx != inc[1]) continue;
                    if (pass == 1 && k.attackerIdx >= 0) continue;
                    int dt = Math.abs(k.tick - inc[0]);
                    if (dt < bestDt) {
                        bestDt = dt;
                        best = k;
                    }
                }
                if (best != null) {
                    if (DEBUG && (pass > 0 || bestDt > 1)) {
                        debug("Kill credit: player " + inc[1] + " tick " + inc[0] + " -> death of " + best.targetIdx
                                + " (pass " + pass + ", " + bestDt + " ticks away, landed by " + best.attackerIdx + ")");
                    }
                    best.killCredited = true;
                    best.killerIdx = inc[1];
                    it.remove();
                }
            }
        }
        unmatchedScoreIncrements = open.size();

        // Assists listed by the combat log (also the fallback assist count). The log lists who
        // had damaged the victim even when neutral creeps or Roshan made the kill; the game then
        // credits nobody, so the list only counts for a death that scores for the other side.
        for (KillEvent k : killLog) {
            if (k.assistRaw == null || !scoresForEnemy(k)) continue;
            for (int raw : k.assistRaw) {
                int idx = eventPidToIdx(raw);
                if (idx < 0 || idx == k.attackerIdx || idx == k.targetIdx) continue;
                if ((idx < 5) == (k.targetIdx < 5)) continue;  // assists come from the killer's team
                if (!k.assistIdx.add(idx)) continue;
                playerStates[idx].assists++;
            }
            k.assistIdx.remove(k.killerIdx);
        }
        // Assist increments: confirm a listed assist, else add the player to the nearest death.
        for (int[] inc : scoreIncrements) {
            if (inc[2] != 1) continue;
            KillEvent listed = null, unlisted = null;
            int listedDt = SCORE_MATCH_TICKS + 1, unlistedDt = SCORE_MATCH_TICKS + 1;
            for (KillEvent k : killLog) {
                if (k.deny || (k.targetIdx < 5) == (inc[1] < 5) || k.assistConfirmed.contains(inc[1])) continue;
                int dt = Math.abs(k.tick - inc[0]);
                if (k.assistIdx.contains(inc[1])) {
                    if (dt < listedDt) {
                        listedDt = dt;
                        listed = k;
                    }
                } else if (k.killerIdx != inc[1] && dt < unlistedDt) {
                    unlistedDt = dt;
                    unlisted = k;
                }
            }
            KillEvent k = listed != null ? listed : unlisted;
            if (k != null) {
                if (DEBUG && Math.abs(k.tick - inc[0]) > 1) {
                    debug("Assist credit: player " + inc[1] + " tick " + inc[0] + " -> death of " + k.targetIdx
                            + " (" + Math.abs(k.tick - inc[0]) + " ticks away, " + (listed != null ? "listed" : "added") + ")");
                }
                k.assistIdx.add(inc[1]);
                k.assistConfirmed.add(inc[1]);
            } else {
                unmatchedScoreIncrements++;
            }
        }
        if (unmatchedScoreIncrements > 0) {
            System.err.println("[warn] " + unmatchedScoreIncrements + " scoreboard kill/assist increments match no death in kills_log");
        }

        // killed / killed_by follow the credited killer, so they agree with kills_log and the
        // scoreboard. Summon kills are keyed by the owning hero; other units by their name.
        for (KillEvent k : killLog) {
            if (k.deny) continue;
            PlayerState victim = playerStates[k.targetIdx];
            if (k.killerIdx >= 0) {
                PlayerState killer = playerStates[k.killerIdx];
                killer.killed.merge(victim.heroName, 1, Integer::sum);
                victim.killedBy.merge(killer.heroName, 1, Integer::sum);
            } else if (!k.killerUnit.isEmpty()) {
                victim.killedBy.merge(cleanName(k.killerUnit), 1, Integer::sum);
            }
        }
    }

    /**
     * Whether a death scores for the other side: not a deny, and either a player is credited
     * with it or a unit of that side (lane creep, tower) made the kill. A hero that dies to
     * neutral creeps, Roshan or the Tormentor with no credited killer gives no kill and no
     * assists.
     */
    static boolean scoresForEnemy(KillEvent k) {
        if (k.deny) return false;
        if (k.killerIdx >= 0) return true;
        return k.attackerTeam == (k.targetIdx < 5 ? TEAM_DIRE : TEAM_RADIANT);
    }

    /** Combat-log unit name of a building record: both tier-4 towers are "..._tower4". */
    static String combatLogBuildingName(String canonical) {
        if (canonical == null) return "";
        return canonical.replaceFirst("_tower4_(top|bot)$", "_tower4");
    }

    /**
     * Death time per building. The entity's own death decides WHICH building died; the
     * combat-log time (the exact moment, shared with the objective) is attached to it at most
     * once, so one tier-4 death can never mark both tier-4 towers.
     */
    void resolveBuildingDeaths() {
        Map<String, List<Float>> unused = new HashMap<>();
        for (Map.Entry<String, List<Float>> e : buildingDeathRaw.entrySet()) {
            unused.put(e.getKey(), new ArrayList<>(e.getValue()));
        }
        List<BuildingRecord> records = new ArrayList<>(buildingsByEntityIndex.values());
        for (BuildingRecord b : records) {
            if (b.destroyedRaw == null || (gameEnded && b.destroyedRaw > gameEndTime + 1f)) continue;
            b.deathRaw = b.destroyedRaw;
            List<Float> times = unused.get(combatLogBuildingName(b.name));
            if (times == null) continue;
            int best = -1;
            for (int i = 0; i < times.size(); i++) {
                if (Math.abs(times.get(i) - b.destroyedRaw) > 5f) continue;
                if (best < 0 || Math.abs(times.get(i) - b.destroyedRaw) < Math.abs(times.get(best) - b.destroyedRaw)) best = i;
            }
            if (best >= 0) b.deathRaw = times.remove(best);
        }
        // No entity evidence (life state unreadable): fall back to an unused combat-log time.
        for (BuildingRecord b : records) {
            if (b.deathRaw != null || b.name == null || b.name.isEmpty()) continue;
            List<Float> times = unused.get(combatLogBuildingName(b.name));
            if (times != null && !times.isEmpty()) b.deathRaw = times.remove(0);
        }
    }

    /** Add "building" (the entity that died) to building-kill objectives; needs resolveBuildingDeaths. */
    void linkObjectivesToBuildings() {
        Set<BuildingRecord> taken = new HashSet<>();
        for (Map<String, Object> obj : objectiveEvents) {
            Object raw = obj.get("raw_time");
            Object key = obj.get("key");
            if (!(raw instanceof Number) || !(key instanceof String) || "roshan_kill".equals(obj.get("type"))) continue;
            float t = ((Number) raw).floatValue();
            BuildingRecord best = null;
            for (BuildingRecord b : buildingsByEntityIndex.values()) {
                if (b.deathRaw == null || taken.contains(b)) continue;
                if (!combatLogBuildingName(b.name).equals("npc_dota_" + key)) continue;
                if (best == null || Math.abs(b.deathRaw - t) < Math.abs(best.deathRaw - t)) best = b;
            }
            if (best != null) {
                taken.add(best);
                obj.put("building", best.name);
            }
        }
    }

    /**
     * Attach each combat-log ward death to its ward entity (W4): same type and team, life state
     * changed in the same tick (or, without a life state, entity deleted a few seconds later).
     */
    void matchWardDeaths() {
        for (WardDeath d : wardDeaths) {
            WardRecord best = null;
            int bestScore = Integer.MAX_VALUE;
            for (WardRecord w : wardRecords) {
                if (w.deathMatched || w.destroyedRaw == null || !w.type.equals(d.type)) continue;
                if (w.team != null && d.wardTeam != 0 && w.team != d.wardTeam) continue;
                int dt = w.destroyedTick - d.tick;
                if (w.diedByLifeState ? Math.abs(dt) > WARD_DEATH_MATCH_TICKS : (dt < 0 || dt > 10 * 30)) continue;
                int score = Math.abs(dt);
                if (d.ownerIdx >= 0 && w.ownerIdx >= 0 && d.ownerIdx != w.ownerIdx) score += 100000;
                if (score < bestScore) {
                    bestScore = score;
                    best = w;
                }
            }
            if (best == null) continue;
            best.deathMatched = true;
            best.dewarded = d.byEnemy;
            best.killerIdx = d.killerIdx;
            if (!best.diedByLifeState) best.destroyedRaw = d.rawTime;
        }
    }

    /** Per-minute series of one player, never longer than minute 0 .. duration / 60 (W6). */
    private List<Integer> minuteSeries(Map<Integer, List<Integer>> timeline, int idx) {
        List<Integer> series = timeline.get(idx);
        if (series == null) return new ArrayList<>();
        int max = duration / 60 + 1;
        return series.size() > max ? new ArrayList<>(series.subList(0, max)) : series;
    }

    /** position_log (W2): [time, x, y, hp %] every POSITION_LOG_INTERVAL_S of game time. */
    private List<int[]> buildPositionLog(PlayerState ps, float endRaw) {
        List<int[]> out = new ArrayList<>();
        List<float[]> samples = ps.samples;
        if (samples.isEmpty()) return out;
        int step = POSITION_LOG_INTERVAL_S;
        float origin = clockAnchored ? gameStartTime : 0f;
        int t = (int) Math.ceil((samples.get(0)[0] - origin) / step) * step;
        int last = (int) Math.floor((endRaw - origin) / step) * step;
        int i = 0;
        for (; t <= last; t += step) {
            float raw = origin + t;
            while (i + 1 < samples.size() && samples.get(i + 1)[0] <= raw) i++;
            float[] sm = samples.get(i);
            if (i + 1 < samples.size() && Math.abs(samples.get(i + 1)[0] - raw) < Math.abs(sm[0] - raw)) {
                sm = samples.get(i + 1);
            }
            out.add(new int[]{t, Math.round(sm[1]), Math.round(sm[2]), (int) sm[3]});
        }
        return out;
    }

    private Map<String, Object> buildOutput() {
        Map<String, Object> result = new LinkedHashMap<>();

        if (gameStarted && gameEnded && gameEndTime > gameStartTime) {
            duration = (int) (gameEndTime - gameStartTime);
        } else if (gameStarted && lastCombatLogTime > gameStartTime) {
            duration = (int) (lastCombatLogTime - gameStartTime);
        } else {
            // The replay ends before the horn: no game was played. Anchor the clock at the
            // scheduled horn (or at the end of the data) so that every event time is <= 0.
            duration = 0;
            gameStartTime = preGameStartRaw > 0
                    ? Math.max(preGameStartRaw + PRE_GAME_S, lastCombatLogTime) : lastCombatLogTime;
            clockAnchored = true;
        }
        applyFileInfo();
        replayPendingHeroRows();

        // ---- Resolve deferred player-id based events ----
        resolveKillCredits();
        for (float[] bb : pendingBuybacks) {
            int idx = eventPidToIdx((int) bb[0]);
            if (idx >= 0) playerStates[idx].buybackRaw.add(bb[1]);
        }
        for (PlayerState ps : playerStates) Collections.sort(ps.buybackRaw);
        resolveBuildingDeaths();
        linkObjectivesToBuildings();
        matchWardDeaths();

        for (PlayerState ps : playerStates) {
            convertRawTimes(ps.purchaseLog);
            convertRawTimes(ps.abilityCasts);
            convertRawTimes(ps.abilityUpgrades);
            convertRawTimes(ps.entityAbilityUpgrades);
        }
        convertRawTimes(objectiveEvents);
        convertRawTimes(chatMessages);

        List<Map<String, Object>> teamfights = detectTeamfights();
        List<Map<String, Object>> smokes = buildSmokes();

        // Chat events: runes, first blood, aegis
        Float firstBloodRaw = null;
        for (RawChatEvent ev : chatEvents) {
            int time = toGameTime(ev.rawTime);
            int idx1 = eventPidToIdx(ev.pid1);
            if (ev.type.equals("CHAT_MESSAGE_RUNE_PICKUP")) {
                if (idx1 < 0) continue;
                Map<String, Object> r = new LinkedHashMap<>();
                r.put("time", time);
                r.put("key", String.valueOf(ev.value));
                playerStates[idx1].runesLog.add(r);
                playerStates[idx1].runes.merge(String.valueOf(ev.value), 1, Integer::sum);
            } else {
                Map<String, Object> obj = new LinkedHashMap<>();
                obj.put("time", time);
                obj.put("type", ev.type);
                if (idx1 >= 0) {
                    obj.put("slot", idx1);
                    obj.put("player_slot", playerStates[idx1].playerSlot);
                }
                if (ev.type.equals("CHAT_MESSAGE_FIRSTBLOOD")) {
                    if (firstBloodRaw == null || ev.rawTime < firstBloodRaw) firstBloodRaw = ev.rawTime;
                    int victim = eventPidToIdx(ev.pid2);
                    if (victim >= 0) {
                        obj.put("key", playerStates[victim].heroName);
                        obj.put("victim_slot", playerStates[victim].playerSlot);
                    }
                } else if (ev.type.equals("CHAT_MESSAGE_COURIER_LOST")) {
                    // playerid_1 = the killer (above; -1 when no hero did it), playerid_2 = the
                    // team that LOST the courier, value = the gold bounty.
                    String loser = teamName(ev.pid2);
                    if (loser != null) obj.put("team", loser);
                    obj.put("value", ev.value);
                }
                objectiveEvents.add(obj);
            }
        }
        objectiveEvents.sort(Comparator.comparingInt(o -> ((Number) o.getOrDefault("time", 0)).intValue()));

        // Chat player slots
        for (Map<String, Object> chat : chatMessages) {
            int slot = -1;
            Object pid = chat.remove("_pid");
            if (pid instanceof Number) {
                int idx = eventPidToIdx(((Number) pid).intValue());
                if (idx >= 0) slot = playerStates[idx].playerSlot;
            } else if (chat.get("player_name") instanceof String) {
                String name = (String) chat.get("player_name");
                for (PlayerState ps : playerStates) {
                    if (!name.isEmpty() && name.equals(ps.playerName)) {
                        slot = ps.playerSlot;
                        break;
                    }
                }
            }
            chat.put("player_slot", slot);
        }

        // ---- Score (J9): Valve's team kill counters; without them, the deaths that score ----
        int[] logScore = killLogScore();
        int radiantScore = teamHeroKills[0] >= 0 ? teamHeroKills[0] : logScore[0];
        int direScore = teamHeroKills[1] >= 0 ? teamHeroKills[1] : logScore[1];

        result.put("schema_version", SCHEMA_VERSION);
        result.put("analyzer_version", analyzerVersion());
        result.put("coord_space", "world");
        result.put("partial", partial);
        result.put("match_id", matchId);
        result.put("match_id_source", matchId > 0 ? matchIdSource : "none");
        result.put("game_started", gameStarted);
        result.put("duration", duration);
        // W5: an unknown winner (truncated replay) is null, never a Dire win.
        result.put("radiant_win", winnerKnown ? radiantWin : null);
        result.put("game_mode", gameMode);
        result.put("start_time", startTime);
        result.put("radiant_score", radiantScore);
        result.put("dire_score", direScore);
        result.put("score_source", teamHeroKills[0] >= 0 && teamHeroKills[1] >= 0 ? "scoreboard" : "kills_log");
        result.put("match_record", matchRecordSeen);

        float endRaw = gameEnded ? gameEndTime : lastCombatLogTime;
        // GPM / XPM use the unrounded game length, like Valve's scoreboard.
        float dur = gameStarted && endRaw > gameStartTime ? (endRaw - gameStartTime) / 60.0f : 1;
        // Actions per minute as OpenDota defines it: every order of the recording (pre-game
        // and post-game included) over the game length in whole seconds.
        int[] orders = new int[10];
        for (Map.Entry<Integer, Integer> en : unitOrders.entrySet()) {
            int idx = entityPidScheme != PID_UNKNOWN ? entityPidToIdx(en.getKey(), null) : -1;
            if (idx >= 0) orders[idx] += en.getValue();
        }
        List<Map<String, Object>> playersOut = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            PlayerState ps = playerStates[i];
            Map<String, Object> p = new LinkedHashMap<>();
            boolean scoreboardKda = ps.sbKills >= 0;
            boolean scoreboardDamage = ps.sbHeroDamage >= 0;

            p.put("player_slot", ps.playerSlot);
            p.put("hero_id", ps.heroId);
            p.put("hero_name", ps.heroName != null && !ps.heroName.isEmpty() ? ps.heroName : "unknown_" + i);
            p.put("hero_npc_name", ps.heroNpcName != null ? ps.heroNpcName : "");
            // Empty when unknown: the localized placeholder is the consumer's job.
            p.put("player_name", ps.playerName != null ? ps.playerName : "");
            p.put("steam_id", ps.steamId);
            p.put("kills", scoreboardKda ? ps.sbKills : ps.kills);
            p.put("deaths", scoreboardKda ? ps.sbDeaths : ps.deaths);
            p.put("assists", scoreboardKda ? ps.sbAssists : ps.assists);
            p.put("kda_source", scoreboardKda ? "scoreboard" : "combat_log");
            p.put("level", ps.level);
            p.put("last_hits", ps.lastHits);
            p.put("denies", ps.denies);
            p.put("gold_per_min", ps.recordGpm >= 0 ? ps.recordGpm : (int) (ps.totalGold / dur));
            p.put("xp_per_min", ps.recordXpm >= 0 ? ps.recordXpm : (int) (ps.totalXp / dur));
            p.put("hero_damage", scoreboardDamage ? ps.sbHeroDamage : ps.heroDamage);
            p.put("tower_damage", scoreboardDamage ? ps.sbTowerDamage : ps.towerDamage);
            p.put("damage_source", scoreboardDamage ? "scoreboard" : "combat_log");
            p.put("hero_healing", ps.recordHealing >= 0 ? ps.recordHealing
                    : scoreboardDamage ? ps.sbHealing : ps.heroHealing);
            p.put("healing_source", ps.recordHealing >= 0 ? "match_record"
                    : scoreboardDamage ? "scoreboard" : "combat_log");
            p.put("net_worth", ps.netWorth > 0 ? ps.netWorth : ps.totalGold);
            p.put("net_worth_source", ps.netWorth > 0 ? "entity" : "earned_gold");
            p.put("total_gold", ps.totalGold);
            p.put("total_xp", ps.totalXp);
            p.put("stuns", ps.stuns);
            p.put("teamfight_participation", ps.teamfightParticipation);
            p.put("actions_per_min", duration > 0 ? (int) (orders[i] * 60L / duration) : 0);

            stripInternalKeys(ps.purchaseLog);
            p.put("purchase_log", ps.purchaseLog);
            p.put("ability_casts", ps.abilityCasts);
            p.put("starting_items", ps.startingItems);

            if (ps.inventoryCaptured) {
                p.put("final_items", ps.finalInventory);
                p.put("backpack", ps.finalBackpack);
                p.put("neutral_item", ps.neutralItem);
                p.put("final_items_source", "inventory");
            } else {
                p.put("final_items", computeFinalItemsFromPurchases(ps));
                p.put("backpack", new ArrayList<>());
                p.put("neutral_item", "");
                p.put("final_items_source", "purchases");
            }

            if (!ps.entityAbilityUpgrades.isEmpty()) {
                stripInternalKeys(ps.entityAbilityUpgrades);
                p.put("ability_upgrades", ps.entityAbilityUpgrades);
                p.put("ability_upgrades_source", "entity");
            } else if (!ps.abilityUpgrades.isEmpty()) {
                stripInternalKeys(ps.abilityUpgrades);
                p.put("ability_upgrades", ps.abilityUpgrades);
                p.put("ability_upgrades_source", "casts");
            } else {
                p.put("ability_upgrades", computeAbilityUpgradesFromLevelUps(ps));
                p.put("ability_upgrades_source", "level_ups");
            }

            p.put("gold_t", minuteSeries(goldTimeline, i));
            p.put("xp_t", minuteSeries(xpTimeline, i));
            p.put("lh_t", minuteSeries(lhTimeline, i));
            p.put("dn_t", minuteSeries(denyTimeline, i));
            p.put("position_log", buildPositionLog(ps, endRaw));

            p.put("killed", ps.killed);
            p.put("killed_by", ps.killedBy);
            p.put("damage", ps.damageDealt);
            p.put("damage_taken", ps.damageTaken);
            p.put("damage_inflictor", ps.damageInflictor);

            p.put("gold_reasons", ps.goldReasons);
            p.put("xp_reasons", ps.xpReasons);

            List<Map<String, Object>> buybackLog = new ArrayList<>();
            for (float t : ps.buybackRaw) {
                Map<String, Object> bb = new LinkedHashMap<>();
                bb.put("time", toGameTime(t));
                buybackLog.add(bb);
            }
            p.put("buyback_count", ps.buybackRaw.size());
            p.put("buyback_log", buybackLog);
            p.put("multi_kills", ps.multiKills);
            p.put("kill_streaks", ps.killStreaks);
            p.put("roshans_killed", ps.sbRoshanKills >= 0 ? ps.sbRoshanKills : ps.roshansKilled);
            p.put("courier_kills", ps.courierKills);
            p.put("observer_kills", ps.observerKills);
            p.put("sentry_kills", ps.sentryKills);
            p.put("tower_kills", ps.sbTowerKills >= 0 ? ps.sbTowerKills : ps.towerKills);
            p.put("lane_kills", ps.laneKills);
            p.put("neutral_kills", ps.neutralKills);
            p.put("ancient_kills", ps.ancientKills);
            p.put("runes_log", ps.runesLog);
            p.put("runes", ps.runes);
            // Lane / role are inferred by the Python side from laning_snapshots.
            p.put("lane", 0);
            p.put("lane_role", 0);

            p.put("laning_snapshots", ps.laningSnapshots);
            p.put("laning_damage_events", ps.laningDamageEvents);
            p.put("laning_regen_events", ps.laningRegenEvents);
            p.put("laning_creep_kills", ps.laningCreepKills);
            p.put("laning_denies", ps.laningDenies);

            playersOut.add(p);
        }
        // Ability casts carry an internal raw time used by teamfights; drop it now.
        for (PlayerState ps : playerStates) stripInternalKeys(ps.abilityCasts);
        result.put("players", playersOut);

        // ---- Kill log (W1): every real hero death ----
        List<Map<String, Object>> killsOut = new ArrayList<>();
        for (KillEvent k : realDeathsSorted()) {
            Map<String, Object> ko = new LinkedHashMap<>();
            ko.put("time", toGameTimeTenths(k.time));
            ko.put("victim_slot", playerStates[k.targetIdx].playerSlot);
            ko.put("killer_slot", k.killerIdx >= 0 ? playerStates[k.killerIdx].playerSlot : null);
            ko.put("killer_unit", k.killerUnit);
            ko.put("killer_team", unitTeamName(k.attackerTeam));
            ko.put("x", k.hasPos ? k.x : null);
            ko.put("y", k.hasPos ? k.y : null);
            List<Integer> assistSlots = new ArrayList<>();
            for (int idx : new TreeSet<>(k.assistIdx)) assistSlots.add(playerStates[idx].playerSlot);
            ko.put("assist_slots", assistSlots);
            ko.put("deny", k.deny);
            killsOut.add(ko);
        }
        result.put("kills_log", killsOut);

        // ---- Wards ----
        List<Map<String, Object>> wardsOut = new ArrayList<>();
        for (WardRecord w : wardRecords) {
            if (!w.hasPos) continue;
            Map<String, Object> ward = new LinkedHashMap<>();
            ward.put("type", w.type);
            ward.put("x", w.x);
            ward.put("y", w.y);
            ward.put("time", toGameTime(w.placedRaw));
            ward.put("player_slot", w.ownerIdx >= 0 ? playerStates[w.ownerIdx].playerSlot : -1);
            String team = w.team != null ? teamName(w.team) : (w.ownerIdx >= 0 ? teamName(teamOfIdx(w.ownerIdx)) : null);
            if (team != null) ward.put("team", team);
            boolean died = w.destroyedRaw != null && !(gameEnded && w.destroyedRaw > gameEndTime + 1f);
            ward.put("destroyed_time", died ? toGameTime(w.destroyedRaw) : null);
            ward.put("ehandle", w.ehandle);
            ward.put("owner_source", w.ownerSource);
            // W4: absent when the ward died but no combat-log death tells how.
            if (!died) {
                ward.put("dewarded", false);
            } else if (w.deathMatched) {
                ward.put("dewarded", w.dewarded);
                if (w.dewarded && w.killerIdx >= 0) ward.put("killer_slot", playerStates[w.killerIdx].playerSlot);
            }
            wardsOut.add(ward);
        }
        result.put("wards", wardsOut);
        result.put("smokes", smokes);
        result.put("teamfights", teamfights);
        stripInternalKeys(objectiveEvents);
        result.put("objectives", objectiveEvents);

        // ---- Buildings ----
        List<Map<String, Object>> buildingsOut = new ArrayList<>();
        for (BuildingRecord b : buildingsByEntityIndex.values()) {
            if (!b.hasPos) continue;
            Map<String, Object> bo = new LinkedHashMap<>();
            bo.put("name", b.name);
            bo.put("team", b.team);
            bo.put("type", b.type);
            bo.put("x", b.x);
            bo.put("y", b.y);
            bo.put("destroyed_time", b.deathRaw != null ? toGameTime(b.deathRaw) : null);
            buildingsOut.add(bo);
        }
        result.put("buildings", buildingsOut);

        // ---- Gold/XP advantage ----
        List<Integer> goldAdv = new ArrayList<>();
        List<Integer> xpAdv = new ArrayList<>();
        int minLen = Integer.MAX_VALUE;
        for (int i = 0; i < 10; i++) {
            int sz = minuteSeries(goldTimeline, i).size();
            if (sz > 0) minLen = Math.min(minLen, sz);
        }
        if (minLen < Integer.MAX_VALUE) {
            for (int m = 0; m < minLen; m++) {
                int rGold = 0, dGold = 0, rXp = 0, dXp = 0;
                for (int i = 0; i < 10; i++) {
                    List<Integer> gl = minuteSeries(goldTimeline, i);
                    List<Integer> xl = minuteSeries(xpTimeline, i);
                    int g = gl.size() > m ? gl.get(m) : 0;
                    int x = xl.size() > m ? xl.get(m) : 0;
                    if (i < 5) {
                        rGold += g;
                        rXp += x;
                    } else {
                        dGold += g;
                        dXp += x;
                    }
                }
                goldAdv.add(rGold - dGold);
                xpAdv.add(rXp - dXp);
            }
        }
        result.put("radiant_gold_adv", goldAdv);
        result.put("radiant_xp_adv", xpAdv);

        stripInternalKeys(chatMessages);
        result.put("chat", chatMessages);

        // ---- Draft ----
        List<Map<String, Object>> draft = draftFromFileInfo();
        String draftSource = "file_info";
        if (draft.isEmpty()) {
            draftSource = "game_rules";
            draft = buildDraftFromGameState();
        }
        result.put("picks_bans", draft);
        result.put("draft_source", draftSource);

        // First blood: the game's own FIRSTBLOOD event; else the first death with an enemy killer.
        if (firstBloodRaw == null) {
            for (KillEvent k : realDeathsSorted()) {
                int enemyTeam = k.targetIdx < 5 ? TEAM_DIRE : TEAM_RADIANT;
                if (!k.deny && (k.killerIdx >= 0 || k.attackerTeam == enemyTeam)) {
                    firstBloodRaw = k.time;
                    break;
                }
            }
        }
        result.put("first_blood_time", firstBloodRaw != null ? Math.max(0, toGameTime(firstBloodRaw)) : 0);
        result.put("cluster", cluster);
        result.put("region", CLUSTER_REGION.getOrDefault(cluster, 0));
        result.put("game_build", gameBuild);
        result.put("patch", "");
        if (entityNameHits == 0 && entityNameMisses > 0) {
            // Every lookup failed: the name-index property was probably renamed again.
            errorCounts.put("entity_names", entityNameMisses);
            System.err.println("[warn] EntityNames never resolved (" + entityNameMisses + " lookups): item, ability"
                    + " and building names fall back to class names; check m_pEntity.m_nameStringTableIndex");
        }
        if (unmatchedScoreIncrements > 0) {
            // The scoreboard credited kills / assists that belong to no death of kills_log.
            errorCounts.put("score_credits", unmatchedScoreIncrements);
        }
        for (PlayerState ps : playerStates) unattributedSkillPoints += ps.pendingSpentPoints;
        if (unattributedSkillPoints > 0) {
            // A skill point was spent and no ability was seen to level: ability_upgrades lacks it.
            errorCounts.put("skill_points", unattributedSkillPoints);
        }
        result.put("parse_errors", errorCounts);
        return result;
    }

    /**
     * Draft without CDemoFileInfo picks/bans: picks from the players (hero id and pick order from
     * the player resource), then the game-rules bans, which belong to no team.
     */
    private List<Map<String, Object>> buildDraftFromGameState() {
        boolean hasPicks = draftEvents.stream().anyMatch(e -> Boolean.TRUE.equals(e.get("is_pick")));
        if (hasPicks) return draftEvents;

        List<PlayerState> picked = new ArrayList<>();
        Set<Integer> orders = new HashSet<>();
        boolean orderKnown = true;
        for (PlayerState ps : playerStates) {
            if (ps.heroId <= 0 && (ps.heroName == null || ps.heroName.isEmpty())) continue;
            picked.add(ps);
            if (ps.pickOrder <= 0 || !orders.add(ps.pickOrder)) orderKnown = false;
        }
        // All pick locks heroes in simultaneous phases: order by the moment the hero appeared,
        // then by the game's own pick order.
        if (orderKnown) {
            picked.sort(Comparator.<PlayerState>comparingInt(ps -> ps.heroPickTick).thenComparingInt(ps -> ps.pickOrder));
        }

        List<Map<String, Object>> draft = new ArrayList<>();
        for (PlayerState ps : picked) {
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("is_pick", true);
            entry.put("hero_id", ps.heroId);
            entry.put("hero_name", ps.heroName);
            entry.put("team", ps.playerIdx < 5 ? 0 : 1);
            entry.put("order", draft.size());
            entry.put("player_slot", ps.playerSlot);
            draft.add(entry);
        }
        for (Map<String, Object> ban : draftEvents) {
            ban.put("order", draft.size());
            draft.add(ban);
        }
        return draft;
    }

    // ======== Name Normalization ========

    /**
     * Convert entity class name to a hero name (fallback when EntityNames is unavailable).
     * "CDOTA_Unit_Hero_DrowRanger" -> "drow_ranger". The Python side normalizes these
     * against heroes.json slugs (e.g. "queen_of_pain" -> "queenofpain").
     */
    static String entityClassToHeroName(String dtClassName) {
        if (dtClassName == null) return "";
        String suffix = dtClassName.replace("CDOTA_Unit_Hero_", "");
        if (suffix.equals(dtClassName)) return "";

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < suffix.length(); i++) {
            char c = suffix.charAt(i);
            if (i > 0 && Character.isUpperCase(c)) {
                char prev = suffix.charAt(i - 1);
                if (Character.isLowerCase(prev) || Character.isDigit(prev)) {
                    sb.append('_');
                }
            }
            sb.append(c);
        }
        return sb.toString().toLowerCase(Locale.ROOT);
    }

    /** Clean a combat log hero name: strip "npc_dota_hero_" prefix. */
    static String cleanHeroName(String name) {
        if (name == null) return "";
        if (name.startsWith("npc_dota_hero_")) {
            return name.substring("npc_dota_hero_".length());
        }
        return name;
    }

    /** Clean a generic combat log name: strip common prefixes. */
    static String cleanName(String name) {
        if (name == null) return "";
        return name.replace("npc_dota_hero_", "")
                .replace("npc_dota_", "")
                .replace("item_", "");
    }

    /** Map a hero name (from combat log) to player index (0-9). */
    private int heroNameToPlayerIdx(String name) {
        if (name == null || name.isEmpty()) return -1;

        Integer cached = heroNameMap.get(name);
        if (cached != null) return cached;

        String lower = name.toLowerCase(Locale.ROOT);
        for (int i = 0; i < 10; i++) {
            String hn = playerStates[i].heroName;
            if (hn != null && !hn.isEmpty() && hn.equals(lower)) {
                heroNameMap.put(name, i);
                return i;
            }
        }
        String lowerNoUnderscore = lower.replace("_", "");
        for (int i = 0; i < 10; i++) {
            String hn = playerStates[i].heroName;
            if (hn != null && !hn.isEmpty() && hn.replace("_", "").equals(lowerNoUnderscore)) {
                heroNameMap.put(name, i);
                return i;
            }
        }
        return -1;
    }

    // ======== Inner Classes ========

    static class PlayerState {
        final int playerIdx;
        final int playerSlot;
        String heroName = "";
        String heroNpcName = "";
        int heroId = 0;
        String playerName = "";
        long steamId = 0;
        int pickOrder = 0;                              // 1-10 from the player resource, 0 = unknown
        int heroPickTick = 0;                           // tick at which the selected hero id appeared
        int kills = 0, deaths = 0, assists = 0;         // combat-log counts (fallback)
        int sbKills = -1, sbDeaths = -1, sbAssists = -1; // scoreboard (player resource), -1 = unread
        int level = 1;
        int lastHits = 0, denies = 0;
        int totalGold = 0, totalXp = 0, netWorth = 0;
        int heroDamage = 0, towerDamage = 0, heroHealing = 0;     // combat-log sums (fallback)
        int sbHeroDamage = -1, sbTowerDamage = -1, sbHealing = -1; // scoreboard (team data), -1 = unread
        int recordHealing = -1, recordGpm = -1, recordXpm = -1;    // Valve's post-game match record, -1 = none
        Float stuns = null;                                        // scoreboard: seconds of stun dealt
        Float teamfightParticipation = null;                       // scoreboard: 0..1; null = unread
        int roshansKilled = 0, courierKills = 0;
        int observerKills = 0, sentryKills = 0, towerKills = 0;   // combat-log counts
        int sbTowerKills = -1, sbRoshanKills = -1;                 // scoreboard (team data), -1 = unread
        int laneKills = 0, neutralKills = 0, ancientKills = 0;
        float lastX = 0, lastY = 0;
        boolean hasPosition = false;
        Entity heroEntity = null;
        int heroEntityIndex = -1;
        boolean heroEntityDeleted = false;
        int registeredTick = -1;             // tick at which the first hero entity was assigned
        boolean spawnItemsCollected = false; // see collectSpawnItems

        // Position history for smoke paths and teamfight positions: [rawCombatLogTime, worldX, worldY]
        List<float[]> positionHistory = new ArrayList<>();
        float lastPositionRecordTime = -999;
        // One per second: [rawTime, worldX, worldY, hpPct] (see sampleHeroes)
        List<float[]> samples = new ArrayList<>();
        // Combat-log gold / XP rows for teamfight deltas: [rawCombatLogTime, value]
        List<float[]> goldLog = new ArrayList<>();
        List<float[]> xpLog = new ArrayList<>();
        List<Float> buybackRaw = new ArrayList<>();
        List<TimedKey> itemUseLog = new ArrayList<>();

        // Damage/heal logs for teamfight damage computation: [rawCombatLogTime, value]
        List<float[]> heroDamageDealtLog = new ArrayList<>();
        List<float[]> heroDamageTakenLog = new ArrayList<>();
        List<float[]> heroHealLog = new ArrayList<>();

        Map<String, Integer> killed = new LinkedHashMap<>();
        Map<String, Integer> killedBy = new LinkedHashMap<>();
        Map<String, Integer> damageDealt = new LinkedHashMap<>();
        Map<String, Integer> damageTaken = new LinkedHashMap<>();
        Map<String, Integer> damageInflictor = new LinkedHashMap<>();
        Map<String, Integer> goldReasons = new LinkedHashMap<>();
        Map<String, Integer> xpReasons = new LinkedHashMap<>();
        Map<String, Integer> runes = new LinkedHashMap<>();
        List<Map<String, Object>> runesLog = new ArrayList<>();
        Map<String, Integer> multiKills = new LinkedHashMap<>();   // {kills in a row: times}
        Map<String, Integer> killStreaks = new LinkedHashMap<>();  // {streak length: times}

        List<Map<String, Object>> purchaseLog = new ArrayList<>();
        List<Map<String, Object>> abilityCasts = new ArrayList<>();
        List<LevelUpEvent> levelUpTimes = new ArrayList<>();
        List<Map<String, Object>> abilityUpgrades = new ArrayList<>();       // from cast levels
        Map<String, Integer> abilityLevels = new LinkedHashMap<>();
        List<Map<String, Object>> entityAbilityUpgrades = new ArrayList<>(); // from m_iLevel
        Map<Integer, AbilityTrack> abilityByHandle = new HashMap<>();
        int entityUpgradeCount = 0;
        int lastAbilityPoints = -1;
        int pendingSpentPoints = 0;     // observed m_iAbilityPoints decreases not yet matched to a level-up
        int pendingSpentTick = 0;
        boolean abilityTrackingWorked = false;

        List<Map<String, Object>> laningSnapshots = new ArrayList<>();
        List<Map<String, Object>> laningDamageEvents = new ArrayList<>();
        List<Map<String, Object>> laningRegenEvents = new ArrayList<>();
        List<Map<String, Object>> laningCreepKills = new ArrayList<>();
        List<Map<String, Object>> laningDenies = new ArrayList<>();
        List<String> startingItems = new ArrayList<>();

        List<String> finalInventory = new ArrayList<>();
        List<String> finalBackpack = new ArrayList<>();
        String neutralItem = "";
        boolean inventoryCaptured = false;
        boolean inventoryFrozen = false;

        PlayerState(int idx) {
            this.playerIdx = idx;
            this.playerSlot = idx < 5 ? idx : 128 + (idx - 5);
        }
    }

    static class KillEvent {
        final float time;
        final int tick;
        final int attackerIdx;          // hero (or owner of the summon) that landed the kill, -1 = none
        final int targetIdx;
        final boolean deny;
        String killerUnit = "";         // raw npc name of the unit that landed the kill
        int attackerTeam = 0;
        float x, y;
        boolean hasPos = false;
        List<Integer> assistRaw = null;
        // Filled by resolveKillCredits():
        int killerIdx = -1;             // player credited with the kill (scoreboard first)
        boolean killCredited = false;   // a scoreboard kill increment was matched to this death
        final Set<Integer> assistIdx = new HashSet<>();
        final Set<Integer> assistConfirmed = new HashSet<>();

        KillEvent(float time, int tick, int attackerIdx, int targetIdx, boolean deny) {
            this.time = time;
            this.tick = tick;
            this.attackerIdx = attackerIdx;
            this.targetIdx = targetIdx;
            this.deny = deny;
        }
    }

    static class WardRecord {
        final int entityIndex;
        final String type;
        final float placedRaw;
        int ehandle = -1;
        float x, y;
        boolean hasPos = false;
        Integer team = null;
        int ownerIdx = -1;
        String ownerSource = "unknown";
        Float destroyedRaw = null;
        int destroyedTick = -1;
        boolean diedByLifeState = false;  // destroyedRaw is the death itself, not the entity deletion
        boolean deathMatched = false;     // a combat-log death was attached (dewarded / killerIdx valid)
        boolean dewarded = false;
        int killerIdx = -1;

        WardRecord(int entityIndex, String type, float placedRaw) {
            this.entityIndex = entityIndex;
            this.type = type;
            this.placedRaw = placedRaw;
        }
    }

    static class BuildingRecord {
        int entityIndex;
        int entityHandle;
        String name = "";
        String team;
        String type;
        float x, y;
        boolean hasPos = false;
        Float destroyedRaw = null;  // entity evidence: life state change or deletion
        Float deathRaw = null;      // resolved death time (see resolveBuildingDeaths)
    }

    /** Combat-log death of a ward: who ended it, and when. */
    static class WardDeath {
        final int tick;
        final float rawTime;
        final String type;
        final int wardTeam;
        final boolean byEnemy;
        final int killerIdx;
        final int ownerIdx;

        WardDeath(int tick, float rawTime, String type, int wardTeam, boolean byEnemy, int killerIdx, int ownerIdx) {
            this.tick = tick;
            this.rawTime = rawTime;
            this.type = type;
            this.wardTeam = wardTeam;
            this.byEnemy = byEnemy;
            this.killerIdx = killerIdx;
            this.ownerIdx = ownerIdx;
        }
    }

    private static class TimedKey {
        final float rawTime;
        final String key;

        TimedKey(float rawTime, String key) {
            this.rawTime = rawTime;
            this.key = key;
        }
    }

    /** A combat-log row for a hero that had no player index yet (see deferHeroRow). */
    private static class PendingHeroRow {
        static final int GOLD = 0, XP = 1, PURCHASE = 2;
        final int kind;
        final String hero;
        final float time;
        final int value;
        final String key;   // gold / xp reason, or the purchased item

        PendingHeroRow(int kind, String hero, float time, int value, String key) {
            this.kind = kind;
            this.hero = hero;
            this.time = time;
            this.value = value;
            this.key = key;
        }
    }

    private static class SmokeGroup {
        final String team;
        final float startRaw;
        final int creatorIdx;
        float startX, startY;
        Float endRaw = null;
        final List<Integer> members = new ArrayList<>();

        SmokeGroup(String team, float startRaw, int creatorIdx) {
            this.team = team;
            this.startRaw = startRaw;
            this.creatorIdx = creatorIdx;
        }
    }

    static class AbilityTrack {
        final String name;
        final int baseLevel;  // level when first seen: 0 = skillable, 1+ = innate / granted
        int level;
        final int slot;
        final int entityIndex;  // what a train order refers to

        AbilityTrack(String name, int level, int slot, int entityIndex) {
            this.name = name;
            this.baseLevel = level;
            this.level = level;
            this.slot = slot;
            this.entityIndex = entityIndex;
        }
    }

    private static class RawChatEvent {
        final String type;
        final int value;
        final int pid1;
        final int pid2;
        final float rawTime;

        RawChatEvent(String type, int value, int pid1, int pid2, float rawTime) {
            this.type = type;
            this.value = value;
            this.pid1 = pid1;
            this.pid2 = pid2;
            this.rawTime = rawTime;
        }
    }

    private static class WardPurchase {
        final int playerIdx;
        final float combatLogTime;
        final String wardType;  // "observer" or "sentry"
        boolean consumed = false;

        WardPurchase(int playerIdx, float combatLogTime, String wardType) {
            this.playerIdx = playerIdx;
            this.combatLogTime = combatLogTime;
            this.wardType = wardType;
        }
    }

    private static class LevelUpEvent {
        final float rawTime;
        final int level;

        LevelUpEvent(float rawTime, int level) {
            this.rawTime = rawTime;
            this.level = level;
        }
    }

    // ======== Main ========

    /**
     * Clarity (3.1.3 and 4.0.3) has no decoder for the field type of m_unMatchID64 ("MatchID_t") and falls
     * back to a 32-bit varint, which drops the upper bits of every match id above 2^32. Both
     * read the same bytes, so registering the 64-bit varint decoder for that type is safe. The
     * decoder table is private: if Clarity's internals change this does nothing and the id
     * from the game rules stays 32-bit (reported as match_id_source "game_rules_32bit").
     */
    @SuppressWarnings("unchecked")
    static boolean registerMatchIdDecoder() {
        try {
            Field table = S2DecoderFactory.class.getDeclaredField("DECODERS");
            table.setAccessible(true);
            ((Map<String, Decoder>) table.get(null)).putIfAbsent("MatchID_t", new LongVarUnsignedDecoder());
            return true;
        } catch (ReflectiveOperationException | RuntimeException e) {
            debug("64-bit match id decoder not registered: " + e);
            return false;
        }
    }

    private static String shortMessage(Throwable t) {
        String msg = t.getMessage();
        String s = t.getClass().getSimpleName() + (msg != null ? ": " + msg : "");
        return s.length() > 300 ? s.substring(0, 300) + "..." : s;
    }

    private void printSummary(Map<String, Object> output) {
        System.err.println("=== Parse Summary ===");
        System.err.println("match_id=" + matchId + " (" + matchIdSource + ") duration=" + duration + "s radiant_win="
                + (winnerKnown ? String.valueOf(radiantWin) : "unknown")
                + " partial=" + partial + " cluster=" + cluster + " entity_pid_scheme=" + entityPidScheme
                + " event_pid_scheme=" + eventPidScheme);
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> players = (List<Map<String, Object>>) output.get("players");
        for (int i = 0; i < 10; i++) {
            PlayerState ps = playerStates[i];
            Map<String, Object> p = players.get(i);
            System.err.println("  [" + i + "] " + ps.heroName + " " + p.get("kills") + "/" + p.get("deaths") + "/"
                    + p.get("assists") + " (" + p.get("kda_source") + ") LH:" + ps.lastHits + " NW:" + ps.netWorth
                    + " upgrades:" + ps.entityAbilityUpgrades.size()
                    + " inventory:" + (ps.inventoryCaptured ? "yes" : "no"));
        }
        if (DEBUG) {
            // Fallback sums next to the scoreboard values, to keep the combat-log rules honest.
            for (PlayerState ps : playerStates) {
                System.err.println("  combat-log vs scoreboard " + ps.heroName + ": K/D/A " + ps.kills + "/" + ps.deaths + "/"
                        + ps.assists + " vs " + ps.sbKills + "/" + ps.sbDeaths + "/" + ps.sbAssists
                        + ", hero damage " + ps.heroDamage + " vs " + ps.sbHeroDamage
                        + ", tower damage " + ps.towerDamage + " vs " + ps.sbTowerDamage
                        + ", healing " + ps.heroHealing + " vs " + ps.sbHealing + " (record " + ps.recordHealing + ")"
                        + ", tower kills " + ps.towerKills + " vs " + ps.sbTowerKills
                        + ", roshan kills " + ps.roshansKilled + " vs " + ps.sbRoshanKills);
            }
            int[] logScore = killLogScore();
            System.err.println("  kill-log score vs team counters: " + logScore[0] + "-" + logScore[1]
                    + " vs " + teamHeroKills[0] + "-" + teamHeroKills[1]);
        }
        long owned = wardRecords.stream().filter(w -> w.ownerIdx >= 0).count();
        long wardDeathsMatched = wardRecords.stream().filter(w -> w.deathMatched).count();
        System.err.println("wards=" + wardRecords.size() + " (owned " + owned + ", deaths matched " + wardDeathsMatched
                + "/" + wardDeaths.size() + ") smokes=" + smokeGroups.size()
                + " buildings=" + buildingsByEntityIndex.size() + " deaths=" + killLog.size()
                + " objectives=" + objectiveEvents.size()
                + " entity_names=" + entityNameHits + "/" + (entityNameHits + entityNameMisses));
        Map<String, Integer> diagnostics = new LinkedHashMap<>(errorCounts);
        diagnostics.keySet().retainAll(DIAGNOSTIC_KEYS);
        Map<String, Integer> failed = new LinkedHashMap<>(errorCounts);
        failed.keySet().removeAll(DIAGNOSTIC_KEYS);
        if (!failed.isEmpty()) System.err.println("handler errors: " + failed);
        if (!diagnostics.isEmpty()) System.err.println("diagnostics: " + diagnostics);
    }

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("clarity-analyzer " + analyzerVersion());
            System.err.println("Usage: java -jar clarity-analyzer.jar <replay.dem>");
            System.exit(EXIT_USAGE);
        }

        String replayPath = args[0];
        File replayFile = new File(replayPath);
        if (!replayFile.isFile() || !replayFile.canRead()) {
            System.err.println("Replay file not found or not readable: " + replayFile.getName());
            System.exit(EXIT_NO_INPUT);
        }
        if (replayFile.length() < 16) {
            System.err.println("File is too small to be a Dota 2 replay");
            System.exit(EXIT_INVALID_REPLAY);
        }

        registerMatchIdDecoder();
        ReplayAnalyzer analyzer = new ReplayAnalyzer();
        try {
            analyzer.fileInfo = Clarity.infoForFile(replayPath);
            if (!analyzer.draftFromFileInfo().isEmpty()) {
                analyzer.draftCollected = true;  // file info has the authoritative draft
            }
        } catch (Exception | StackOverflowError e) {
            System.err.println("[warn] Replay file info unavailable (truncated replay?): " + shortMessage(e));
        }

        Throwable runError = null;
        MappedFileSource source = null;
        try {
            source = new MappedFileSource(replayPath);
            new SimpleRunner(source).runWith(analyzer);
        } catch (Exception | StackOverflowError e) {
            runError = e;
        } finally {
            if (source != null) {
                try {
                    source.close();
                } catch (Exception ignored) {
                }
            }
        }

        if (runError != null) {
            System.err.println("Replay parsing stopped early: " + shortMessage(runError));
            if (DEBUG) runError.printStackTrace(System.err);
            if (!analyzer.hasUsableData()) {
                System.err.println("No usable match data could be extracted (not a Dota 2 replay or corrupted file)");
                System.exit(EXIT_INVALID_REPLAY);
            }
            analyzer.partial = true;
        } else if (!analyzer.hasUsableData()) {
            System.err.println("No usable match data could be extracted (not a Dota 2 replay or corrupted file)");
            System.exit(EXIT_INVALID_REPLAY);
        }
        if (analyzer.fileInfo == null && !analyzer.gameEnded) {
            analyzer.partial = true;
        }

        Map<String, Object> output;
        try {
            // Final reads for a replay that stops before game end; otherwise everything was
            // frozen at game end (markGameEnded) and must not pick up post-game values.
            if (analyzer.lastCtx != null && !analyzer.gameEnded) {
                analyzer.readTeamData(analyzer.lastCtx);
                if (analyzer.playerResourceEntity != null) analyzer.updatePlayerResource(analyzer.playerResourceEntity);
                analyzer.sampleHeroes(analyzer.nowRaw());
                analyzer.pollAbilityLevels();
                analyzer.captureAllInventories(false);
            }
            output = analyzer.buildOutput();
        } catch (Exception e) {
            System.err.println("Failed to assemble match data: " + shortMessage(e));
            if (DEBUG) e.printStackTrace(System.err);
            System.exit(EXIT_INTERNAL);
            return;
        }

        analyzer.printSummary(output);

        try {
            Gson gson = new GsonBuilder().serializeNulls().serializeSpecialFloatingPointValues().create();
            Writer out = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(FileDescriptor.out), StandardCharsets.UTF_8), 1 << 16);
            gson.toJson(output, out);
            out.write('\n');
            out.flush();
        } catch (Exception e) {
            System.err.println("Failed to write JSON output: " + shortMessage(e));
            System.exit(EXIT_INTERNAL);
        }
        System.exit(EXIT_OK);
    }
}
