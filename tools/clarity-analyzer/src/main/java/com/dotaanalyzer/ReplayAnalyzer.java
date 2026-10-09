package com.dotaanalyzer;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import skadistats.clarity.Clarity;
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
import skadistats.clarity.wire.shared.demo.proto.Demo.CDemoFileInfo;
import skadistats.clarity.wire.shared.demo.proto.Demo.CGameInfo;
import skadistats.clarity.wire.shared.s2.proto.S2UserMessages.CUserMessageSayText2;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Dota 2 Replay Analyzer using the Clarity 3.x library.
 * Parses .dem replay files and prints one JSON document (schema_version 2) to stdout.
 * Diagnostics go to stderr; set CLARITY_DEBUG=1 for verbose output.
 *
 * Notes:
 *  - Positions are emitted in Dota 2 world units: world = cell * 128 + vec - 16384
 *    (CBodyComponent.m_cellX/m_vecX), matching backend/app/geo.py.
 *  - Hero entity m_iPlayerID has been observed both as 0..9 and as 0,2,..,18 (doubled).
 *    The scheme is detected from (player id, team) pairs; combat-log assist / chat player
 *    ids are resolved separately at the end of the run (they use their own scheme).
 *  - Game time comes from combat log timestamps (DOTA_COMBATLOG_GAME_STATE 5 = horn).
 *  - Every entity property read is null-safe: a missing property falls back to the
 *    previous heuristic so older/newer replays still produce output.
 *
 * Exit codes: 0 ok (output may carry "partial": true), 64 usage, 65 not a replay /
 * nothing usable, 66 file missing or unreadable, 70 internal error.
 */
@UsesEntities
@UsesStringTable("EntityNames")
public class ReplayAnalyzer {

    static final int SCHEMA_VERSION = 2;

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
    /** Two hero deaths belong to the same teamfight when they are at most this many seconds apart. */
    static final float TEAMFIGHT_GAP_S = 15f;

    static final int TEAM_RADIANT = 2;
    static final int TEAM_DIRE = 3;

    private static final int PID_UNKNOWN = 0;
    private static final int PID_DIRECT = 1;
    private static final int PID_DOUBLED = 2;

    private static final int MAX_ABILITY_SLOTS = 35;
    private static final int ABILITY_POLL_TICKS = 15;
    private static final int NEUTRAL_ITEM_SLOT = 16;
    private static final int MAX_ERRORS_LOGGED_PER_HANDLER = 3;

    /** Building unit name, e.g. goodguys_tower1_top, badguys_melee_rax_mid, goodguys_fort. */
    static final Pattern BUILDING_RE = Pattern.compile(
            "(goodguys|badguys)_(tower[1-4](?:_(?:top|mid|bot))?|(?:melee|range)_rax_(?:top|mid|bot)|fort)");

    // Pre-formatted property names (avoid String.format on hot paths)
    private static final String[] ITEM_SLOT_PROPS = new String[17];
    private static final String[] ABILITY_SLOT_PROPS = new String[MAX_ABILITY_SLOTS];
    private static final String[][] DATA_TEAM_PROPS = new String[5][];
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
            };
        }
    }

    // ---- Match-level data ----
    private long matchId = 0;
    private int duration = 0;
    private boolean radiantWin = false;
    private boolean winnerKnown = false;
    private int gameMode = 0;
    private long startTime = 0;
    private boolean partial = false;
    private CDemoFileInfo fileInfo = null;

    // ---- Per-player data (indexed 0-9, 0-4 radiant, 5-9 dire) ----
    private final PlayerState[] playerStates = new PlayerState[10];

    // ---- Events ----
    private final List<WardRecord> wardRecords = new ArrayList<>();
    private final Map<Integer, WardRecord> wardsByEntityIndex = new HashMap<>();
    private final List<Map<String, Object>> objectiveEvents = new ArrayList<>();
    private final List<SmokeGroup> smokeGroups = new ArrayList<>();
    private final Map<Integer, SmokeGroup> activeSmokeByPlayer = new HashMap<>();
    private final Map<Integer, BuildingRecord> buildingsByEntityIndex = new LinkedHashMap<>();
    private final Map<String, Float> buildingDeathRaw = new HashMap<>();
    private final List<RawChatEvent> chatEvents = new ArrayList<>();
    private final List<Map<String, Object>> chatMessages = new ArrayList<>();

    // ---- Time-series data ----
    private final Map<Integer, List<Integer>> goldTimeline = new HashMap<>();
    private final Map<Integer, List<Integer>> xpTimeline = new HashMap<>();
    private final Map<Integer, List<Integer>> lhTimeline = new HashMap<>();

    // ---- Real hero deaths (teamfights, smokes, score) ----
    private final List<KillEvent> killLog = new ArrayList<>();

    // ---- Clock ----
    private float gameStartTime = 0;   // combat log timestamp of the horn (state 5)
    private float gameEndTime = 0;     // combat log timestamp of state 6 (post game)
    private boolean gameStarted = false;
    private boolean gameEnded = false;
    private float lastCombatLogTime = 0;
    private int lastCombatLogTick = 0;
    private int currentTick = 0;
    private float secondsPerTick = 1f / 30f;
    private int lastMinuteRecorded = -1;
    private int lastLaneSnapshotSecond = -5;
    private int lastAbilityPollTick = -1000;
    private int rulesStartTick = -1;
    private int rulesEndTick = -1;
    private float rulesStartRaw = 0;
    private float rulesEndRaw = 0;
    private static final int GAME_STATE_FALLBACK_TICKS = 150;

    // ---- Starting items snapshot ----
    private boolean startingItemsCollected = false;

    // ---- Ward purchase tracking (fallback owner attribution) ----
    private final List<WardPurchase> recentWardPurchases = new ArrayList<>();

    // ---- Draft tracking ----
    private boolean draftCollected = false;
    private int draftOrder = 0;
    private final List<Map<String, Object>> draftEvents = new ArrayList<>();

    // ---- Hero bookkeeping ----
    private final Map<String, Integer> heroNameMap = new HashMap<>();
    private final Map<Integer, Integer> heroEntityIndexToIdx = new HashMap<>();
    private final Set<Integer> secondaryHeroEntities = new HashSet<>();
    private int entityPidScheme = PID_UNKNOWN;
    private int firstHeroTick = -1;

    // ---- Context / processors ----
    private Context lastCtx = null;
    private Entity gameRulesEntity = null;

    // ---- Property access cache (J16) ----
    private static final Object MISSING = new Object();
    private final Map<DTClass, Map<String, Object>> fieldPathCache = new HashMap<>();

    // ---- Diagnostics ----
    private final Map<String, Integer> errorCounts = new LinkedHashMap<>();

    // ---- Consumable items to exclude from the purchase-based final items fallback ----
    private static final Set<String> CONSUMABLE_ITEMS = new HashSet<>(Arrays.asList(
            "tango", "flask", "clarity", "enchanted_mango", "faerie_fire",
            "ward_observer", "ward_sentry", "ward_dispenser", "dust",
            "smoke_of_deceit", "tome_of_knowledge", "tpscroll",
            "blood_grenade", "observer_and_sentry_wards"
    ));

    /** Non-skill abilities that never consume an ability point. */
    private static final String[] GENERIC_ABILITY_PREFIXES = {
            "generic_hidden", "ability_", "plus_", "twin_gate_portal_warp", "seasonal_",
            "dota_", "high_five", "abyssal_underlord_portal_warp", "cny_", "frostivus",
    };

    private static final String[] ANCIENT_CREEP_FRAGMENTS = {
            "black_dragon", "black_drake", "granite_golem", "rock_golem", "big_thunder_lizard",
            "small_thunder_lizard", "elder_jungle_stalker", "jungle_stalker", "prowler_acolyte",
            "prowler_shaman", "frostbitten_golem", "ice_shaman",
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

    /** Entity designer name from the EntityNames string table, e.g. "item_tango". */
    private String entityName(Entity e) {
        Integer idx = intProp(e, "m_pEntity.m_nameStringableIndex");
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
        if (lastCombatLogTick <= 0 || currentTick <= lastCombatLogTick) return lastCombatLogTime;
        float extra = (currentTick - lastCombatLogTick) * secondsPerTick;
        return lastCombatLogTime + Math.min(extra, 3.0f);
    }

    private int toGameTime(float raw) {
        return (int) (gameStarted ? raw - gameStartTime : raw);
    }

    private static int teamOfIdx(int idx) {
        return idx < 5 ? TEAM_RADIANT : TEAM_DIRE;
    }

    private static String teamName(int team) {
        return team == TEAM_RADIANT ? "radiant" : team == TEAM_DIRE ? "dire" : null;
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

        if (gameStarted) {
            try {
                float gameTime = time - gameStartTime;
                int minute = (int) (gameTime / 60);
                if (minute > lastMinuteRecorded && minute >= 0 && minute < 300) {
                    readTeamData(ctx);
                    lastMinuteRecorded = minute;
                    recordMinuteSnapshot(minute);
                    if (!gameEnded) captureAllInventories(false);
                }
                int gameSec = (int) gameTime;
                if (gameSec >= 0 && gameSec <= 600 && gameSec >= lastLaneSnapshotSecond + 5) {
                    recordLaningSnapshot(gameSec);
                }
            } catch (Exception e) {
                recordError("combatlog.snapshots", e);
            }
        }

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
                default:
                    break;
            }
        } catch (Exception e) {
            recordError("combatlog." + typeName, e);
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
        gameStartTime = rawTime;
        debug("Game started at combat log time " + rawTime);
        snapshotStartingItems();
    }

    private void markGameEnded(float rawTime) {
        gameEnded = true;
        gameEndTime = rawTime;
        debug("Game ended at combat log time " + rawTime);
        if (lastCtx != null) readTeamData(lastCtx);
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

    // Entity class suffix -> item key for names that don't follow CamelCase-to-snake_case
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
        Entity item = entityByHandle(handle);
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
        if (attackerTeam != TEAM_RADIANT && attackerTeam != TEAM_DIRE && attackerTeam != 4 && attackerIdx >= 0) {
            attackerTeam = teamOfIdx(attackerIdx);
        }

        // ---- Real hero deaths (no illusions, no reincarnation/aegis deaths) ----
        if (targetIsHero && !targetIsIllusion && targetIdx >= 0 && !reincarnating) {
            int targetTeam = teamOfIdx(targetIdx);
            boolean deny = attackerTeam == targetTeam;
            PlayerState victim = playerStates[targetIdx];
            victim.deaths++;

            KillEvent ke = new KillEvent(time, attackerIdx, targetIdx, deny);
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

            if (!deny && attackerIdx >= 0 && attackerIdx != targetIdx) {
                PlayerState killer = playerStates[attackerIdx];
                killer.kills++;
                killer.killed.merge(target, 1, Integer::sum);
                // Summon kills are credited to (and keyed by) the owning hero.
                boolean viaSummon = heroNameToPlayerIdx(attacker) != attackerIdx;
                victim.killedBy.merge(viaSummon && !killer.heroName.isEmpty() ? killer.heroName : attacker, 1, Integer::sum);
            } else if (!deny && !attacker.isEmpty()) {
                victim.killedBy.merge(cleanName(attackerRaw), 1, Integer::sum);
            }

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
            buildingDeathRaw.putIfAbsent("npc_dota_" + key, time);
            if (!denied && attackerIdx >= 0 && type.equals("tower_kill")) playerStates[attackerIdx].towerKills++;
        } else if (targetClean.contains("roshan") && !targetIsHero) {
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
        if (targetClean.contains("melee")) creepType = "melee";
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

    private void handleDamage(CombatLogEntry cle, float time) {
        String targetRaw = cle.getTargetName();
        String target = cleanHeroName(targetRaw);
        String attacker = cleanHeroName(cle.getAttackerName());
        int value = cle.getValue();
        String inflictor = cleanName(cle.getInflictorName());

        int attackerIdx = heroNameToPlayerIdx(attacker);
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

        if (attackerIdx >= 0 && targetIdx >= 0 && targetIsHero && !targetIsIllusion) {
            playerStates[attackerIdx].heroDamage += value;
            playerStates[attackerIdx].damageDealt.merge(target, value, Integer::sum);
            playerStates[targetIdx].damageTaken.merge(attacker, value, Integer::sum);
            if (!inflictor.isEmpty()) {
                playerStates[attackerIdx].damageInflictor.merge(inflictor, value, Integer::sum);
            }

            playerStates[attackerIdx].heroDamageDealtLog.add(new float[]{time, value});
            playerStates[targetIdx].heroDamageTakenLog.add(new float[]{time, value});

            if (gameStarted) {
                float gameTime = time - gameStartTime;
                if (gameTime >= 0 && gameTime <= 600) {
                    Map<String, Object> evt = new LinkedHashMap<>();
                    evt.put("time", Math.round(gameTime));
                    evt.put("target_idx", targetIdx);
                    evt.put("damage", value);
                    if (!inflictor.isEmpty()) {
                        evt.put("inflictor", inflictor);
                    }
                    playerStates[attackerIdx].laningDamageEvents.add(evt);
                }
            }
        }

        if (attackerIdx >= 0 && BUILDING_RE.matcher(cleanName(targetRaw).toLowerCase(Locale.ROOT)).find()) {
            playerStates[attackerIdx].towerDamage += value;
        }
    }

    private void handleHeal(CombatLogEntry cle, float time) {
        String source = cleanHeroName(cle.getAttackerName());
        int value = cle.getValue();
        int srcIdx = heroNameToPlayerIdx(source);

        boolean targetIsHero;
        try {
            targetIsHero = cle.isTargetHero();
        } catch (Exception e) {
            targetIsHero = cle.getTargetName() != null && cle.getTargetName().contains("npc_dota_hero_");
        }

        if (srcIdx >= 0 && targetIsHero) {
            playerStates[srcIdx].heroHealing += value;
            playerStates[srcIdx].heroHealLog.add(new float[]{time, value});
        }
    }

    private void handlePurchase(CombatLogEntry cle, float time) {
        String buyer = cleanHeroName(cle.getTargetName());
        String item = cleanName(cle.getValueName());
        int buyerIdx = heroNameToPlayerIdx(buyer);
        if (buyerIdx >= 0) {
            Map<String, Object> purchase = new LinkedHashMap<>();
            purchase.put("raw_time", time);  // converted to game time in buildOutput()
            purchase.put("key", item);
            playerStates[buyerIdx].purchaseLog.add(purchase);

            if (item.equals("ward_observer") || item.equals("ward_dispenser")) {
                recentWardPurchases.add(new WardPurchase(buyerIdx, time, "observer"));
            }
            if (item.equals("ward_sentry") || item.equals("ward_dispenser")) {
                recentWardPurchases.add(new WardPurchase(buyerIdx, time, "sentry"));
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
        if (!gameStarted) return;
        float gameTime = time - gameStartTime;
        if (gameTime < 0 || gameTime > 600) return;

        int casterIdx = heroNameToPlayerIdx(cleanHeroName(cle.getAttackerName()));
        if (casterIdx < 0) return;

        String item = cleanName(cle.getInflictorName());
        if (isRegenItem(item)) {
            Map<String, Object> evt = new LinkedHashMap<>();
            evt.put("time", Math.round(gameTime));
            evt.put("item", item);
            playerStates[casterIdx].laningRegenEvents.add(evt);
        }
    }

    private void handleGold(CombatLogEntry cle, float time) {
        int idx = heroNameToPlayerIdx(cleanHeroName(cle.getTargetName()));
        if (idx >= 0) {
            int reason = cle.hasGoldReason() ? cle.getGoldReason() : 0;
            playerStates[idx].goldReasons.merge(String.valueOf(reason), cle.getValue(), Integer::sum);
        }
    }

    private void handleXp(CombatLogEntry cle, float time) {
        int idx = heroNameToPlayerIdx(cleanHeroName(cle.getTargetName()));
        if (idx >= 0) {
            int reason = cle.hasXpReason() ? cle.getXpReason() : 0;
            playerStates[idx].xpReasons.merge(String.valueOf(reason), cle.getValue(), Integer::sum);
        }
    }

    private void handleBuyback(CombatLogEntry cle, float time) {
        int idx = heroNameToPlayerIdx(cleanHeroName(cle.getTargetName()));
        if (idx >= 0) {
            playerStates[idx].buybackCount++;
        } else if (cle.hasValue()) {
            // Buyback entries usually carry the player id in "value" rather than a target name.
            pendingBuybackIds.add(cle.getValue());
        }
    }

    private final List<Integer> pendingBuybackIds = new ArrayList<>();

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
            group = new SmokeGroup(team, time, idx);
            group.startX = ps.lastX;
            group.startY = ps.lastY;
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
            }
        } catch (Exception ex) {
            recordError("entity.created", ex);
        }
    }

    @OnEntityDeleted
    public void onEntityDeleted(Context ctx, Entity e) {
        lastCtx = ctx;
        try {
            int index = e.getIndex();
            WardRecord ward = wardsByEntityIndex.remove(index);
            if (ward != null && ward.destroyedRaw == null) {
                ward.destroyedRaw = nowRaw();
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
            if (w.ownerIdx < 0) {
                int owner = resolveOwnerIdx(e, w.team);
                if (owner >= 0) {
                    w.ownerIdx = owner;
                    w.ownerSource = "entity";
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
        Long mid = longProp(e, "m_pGameRules.m_unMatchID64");
        if (mid != null && mid > 0) matchId = mid;

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
        Integer heroId = intProp(e, "m_iHeroID", "m_nHeroID");
        if (heroId != null && heroId > 0) ps.heroId = heroId;
        debug("Primary hero: " + ps.heroName + " (" + ps.heroNpcName + ") -> idx=" + idx);
    }

    private void updateHeroState(PlayerState ps, Entity e) {
        Integer lvl = intProp(e, "m_iCurrentLevel");
        if (lvl != null && lvl > ps.level) ps.level = lvl;

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

    private void updatePlayerData(Entity e, int offset) {
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
            }
        } catch (Exception ex) {
            recordError("entity.team_data", ex);
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
            // Fallback: correlate with a recent ward purchase from the ward's team.
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

    /** Ward owner via m_hOwnerEntity -> hero entity -> player index (J3). */
    private int resolveOwnerIdx(Entity ward, Integer wardTeam) {
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
            if (!cls.startsWith("CDOTA_Unit_Hero_")) return -1;
            Integer pid = intProp(owner, "m_iPlayerID", "m_nPlayerID");
            if (pid == null) return -1;
            int resolved = entityPidToIdx(pid, intProp(owner, "m_iTeamNum"));
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

    /** "dota_goodguys_tower1_top" / "npc_dota_goodguys_tower1_top" -> "npc_dota_goodguys_tower1_top". */
    static String canonicalBuildingName(String raw) {
        if (raw == null) return null;
        Matcher m = BUILDING_RE.matcher(raw.toLowerCase(Locale.ROOT));
        return m.find() ? "npc_dota_" + m.group(0) : null;
    }

    // ======== Team data / snapshots ========

    private void readTeamData(Context ctx) {
        try {
            Entities ents = ctx.getProcessor(Entities.class);
            Entity radiant = ents.getByDtName("CDOTA_DataRadiant");
            if (radiant != null) updatePlayerData(radiant, 0);
            Entity dire = ents.getByDtName("CDOTA_DataDire");
            if (dire != null) updatePlayerData(dire, 5);
        } catch (Exception ex) {
            recordError("team_data.read", ex);
        }
    }

    private void recordMinuteSnapshot(int minute) {
        for (int i = 0; i < 10; i++) {
            List<Integer> gl = goldTimeline.computeIfAbsent(i, k -> new ArrayList<>());
            List<Integer> xl = xpTimeline.computeIfAbsent(i, k -> new ArrayList<>());
            List<Integer> ll = lhTimeline.computeIfAbsent(i, k -> new ArrayList<>());

            // Fill any gaps with last known value
            int lastGold = gl.isEmpty() ? 0 : gl.get(gl.size() - 1);
            int lastXp = xl.isEmpty() ? 0 : xl.get(xl.size() - 1);
            int lastLh = ll.isEmpty() ? 0 : ll.get(ll.size() - 1);
            while (gl.size() < minute) gl.add(lastGold);
            while (xl.size() < minute) xl.add(lastXp);
            while (ll.size() < minute) ll.add(lastLh);

            gl.add(playerStates[i].totalGold);
            xl.add(playerStates[i].totalXp);
            ll.add(playerStates[i].lastHits);
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
        if (currentTick - lastAbilityPollTick >= ABILITY_POLL_TICKS) {
            lastAbilityPollTick = currentTick;
            pollAbilityLevels();
        }
    }

    /**
     * Poll each primary hero's ability entities and record every m_iLevel increase (includes
     * passives and talents). Sub-abilities that level together with their parent, Invoker's
     * invoked spells and generic abilities are filtered; the total is capped by hero level.
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

    private void pollPlayerAbilities(PlayerState ps) {
        // Skill points actually spent since the last poll (when the hero exposes them).
        // A level-up grants at most one point, so this never undercounts real upgrades.
        Integer pointsLeft = intProp(ps.heroEntity, "m_iAbilityPoints");
        if (pointsLeft != null && pointsLeft >= 0) {
            if (ps.lastAbilityPoints >= 0) {
                int gained = Math.max(0, ps.level - ps.lastPolledLevel);
                ps.pendingSpentPoints += Math.max(0, ps.lastAbilityPoints + gained - pointsLeft);
            }
            ps.lastAbilityPoints = pointsLeft;
        }
        ps.lastPolledLevel = ps.level;

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
                ps.abilityByHandle.put(h, new AbilityTrack(name, lvl, slot));
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
        if (increased.isEmpty()) {
            // Points and levels change in the same server tick; keep at most one unmatched
            // point for the next poll so stale allowance cannot admit later auto-levels.
            ps.pendingSpentPoints = Math.min(ps.pendingSpentPoints, 1);
            return;
        }

        float now = nowRaw();
        List<Integer> accepted = new ArrayList<>();
        for (int k = 0; k < increased.size(); k++) {
            AbilityTrack t = increased.get(k);
            if (!isSkillPointAbility(t.name, ps.heroName)) continue;
            boolean derived = false;
            for (AbilityTrack other : increased) {
                if (other != t && other.name != null && t.name != null && t.name.startsWith(other.name + "_")) {
                    derived = true;
                    break;
                }
            }
            if (!derived) accepted.add(k);
        }
        // Increases are in slot order (main abilities first). Without m_iAbilityPoints the
        // total is capped by hero level instead.
        boolean pointsKnown = ps.lastAbilityPoints >= 0;
        int available = pointsKnown ? ps.pendingSpentPoints : Math.max(0, ps.level - ps.entityUpgradeCount);
        int availableBefore = available;
        for (int k = 0; k < increased.size(); k++) {
            AbilityTrack t = increased.get(k);
            int newLevel = newLevels.get(k);
            if (accepted.contains(k)) {
                for (int lvl = t.level + 1; lvl <= newLevel && available > 0; lvl++) {
                    Map<String, Object> up = new LinkedHashMap<>();
                    up.put("name", t.name);
                    up.put("level", lvl);
                    up.put("raw_time", now);
                    ps.entityAbilityUpgrades.add(up);
                    ps.entityUpgradeCount++;
                    available--;
                }
            }
            t.level = newLevel;
        }
        if (pointsKnown) {
            ps.pendingSpentPoints = Math.min(ps.pendingSpentPoints - (availableBefore - available), 1);
        }
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

    private List<Map<String, Object>> detectTeamfights() {
        List<Map<String, Object>> teamfights = new ArrayList<>();
        List<KillEvent> sorted = realDeathsSorted();
        if (sorted.isEmpty()) return teamfights;

        List<KillEvent> current = new ArrayList<>();
        current.add(sorted.get(0));
        for (int i = 1; i < sorted.size(); i++) {
            KillEvent k = sorted.get(i);
            KillEvent prev = current.get(current.size() - 1);
            if (k.time - prev.time <= TEAMFIGHT_GAP_S) {
                current.add(k);
            } else {
                if (current.size() >= 3) teamfights.add(buildTeamfight(current));
                current = new ArrayList<>();
                current.add(k);
            }
        }
        if (current.size() >= 3) teamfights.add(buildTeamfight(current));
        return teamfights;
    }

    private Map<String, Object> buildTeamfight(List<KillEvent> kills) {
        float start = kills.get(0).time;
        float end = kills.get(kills.size() - 1).time;

        // Expand window slightly to capture pre-fight positioning and post-fight cleanup
        float windowStart = start - 5.0f;
        float windowEnd = end + 5.0f;

        int startGame = toGameTime(start);
        int endGame = toGameTime(end);

        int radiantDeaths = 0, direDeaths = 0;
        for (KillEvent k : kills) {
            if (k.targetIdx < 5) radiantDeaths++;
            else direDeaths++;
        }

        List<Map<String, Object>> tfPlayers = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            PlayerState ps = playerStates[i];
            Map<String, Object> tp = new LinkedHashMap<>();
            tp.put("player_slot", ps.playerSlot);
            tp.put("hero_name", ps.heroName != null ? ps.heroName : "");

            int killsInFight = 0, deathsInFight = 0;
            float deathTime = -1;
            for (KillEvent k : kills) {
                if (k.attackerIdx == i && !k.deny) killsInFight++;
                if (k.targetIdx == i) {
                    deathsInFight++;
                    deathTime = k.time;
                }
            }

            List<Map<String, Object>> positions = new ArrayList<>();
            float firstX = 0, firstY = 0;
            boolean foundFirst = false;
            for (float[] pos : ps.positionHistory) {
                if (pos[0] >= windowStart && pos[0] <= windowEnd) {
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
                if (castRaw >= start - 3 && castRaw <= end + 5) {
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
                if (dmg[0] >= windowStart && dmg[0] <= windowEnd) damageDealt += (int) dmg[1];
            }
            for (float[] dmg : ps.heroDamageTakenLog) {
                if (dmg[0] >= windowStart && dmg[0] <= windowEnd) damageTaken += (int) dmg[1];
            }
            for (float[] heal : ps.heroHealLog) {
                if (heal[0] >= windowStart && heal[0] <= windowEnd) healing += (int) heal[1];
            }

            tp.put("x", (double) firstX);
            tp.put("y", (double) firstY);
            tp.put("kills", killsInFight);
            tp.put("deaths", deathsInFight);
            tp.put("damage_dealt", damageDealt);
            tp.put("damage_taken", damageTaken);
            tp.put("healing", healing);
            tp.put("gold_delta", 0);
            tp.put("xp_delta", 0);
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
        tf.put("radiant_deaths", radiantDeaths);
        tf.put("dire_deaths", direDeaths);
        tf.put("gold_delta", 0);
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
                BuildingRecord b = buildingByCanonicalName("npc_dota_" + obj.get("key"));
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

    /** Fallback draft source (game rules) used only when CDemoFileInfo has no picks/bans. */
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
                    // No team information in this vector; alternate as a best guess.
                    entry.put("team", i % 2);
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
                    || type.equals("CHAT_MESSAGE_DENIED_AEGIS")) {
                chatEvents.add(new RawChatEvent(type, message.getValue(),
                        message.getPlayerid1(), message.getPlayerid2(), nowRaw()));
            }
        } catch (Exception e) {
            recordError("message.chat_event", e);
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
        if (matchId <= 0 && dota.hasMatchId()) matchId = dota.getMatchId();
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
            startTime = Math.max(0, (long) dota.getEndTime() - duration);
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

    private Map<String, Object> buildOutput() {
        Map<String, Object> result = new LinkedHashMap<>();

        if (gameStarted && gameEnded && gameEndTime > gameStartTime) {
            duration = (int) (gameEndTime - gameStartTime);
        } else if (gameStarted && lastCombatLogTime > gameStartTime) {
            duration = (int) (lastCombatLogTime - gameStartTime);
        } else if (lastCombatLogTime > 0) {
            duration = (int) lastCombatLogTime;
        }
        applyFileInfo();

        // ---- Resolve deferred player-id based events ----
        for (KillEvent k : killLog) {
            if (k.assistRaw == null || k.deny) continue;
            Set<Integer> credited = new HashSet<>();
            for (int raw : k.assistRaw) {
                int idx = eventPidToIdx(raw);
                if (idx < 0 || idx == k.attackerIdx || idx == k.targetIdx || !credited.add(idx)) continue;
                if ((idx < 5) == (k.targetIdx < 5)) continue;  // assists come from the killer's team
                playerStates[idx].assists++;
            }
        }
        for (int raw : pendingBuybackIds) {
            int idx = eventPidToIdx(raw);
            if (idx >= 0) playerStates[idx].buybackCount++;
        }

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
                    int victim = eventPidToIdx(ev.pid2);
                    if (victim >= 0) obj.put("key", playerStates[victim].heroName);
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

        // ---- Score (J9): real hero deaths per team, excluding denies ----
        int radiantScore = 0, direScore = 0;
        for (KillEvent k : killLog) {
            if (k.deny) continue;
            if (k.targetIdx < 5) direScore++;
            else radiantScore++;
        }

        result.put("schema_version", SCHEMA_VERSION);
        result.put("coord_space", "world");
        result.put("partial", partial);
        result.put("match_id", matchId);
        result.put("duration", duration);
        result.put("radiant_win", radiantWin);
        result.put("game_mode", gameMode);
        result.put("start_time", startTime);
        result.put("radiant_score", radiantScore);
        result.put("dire_score", direScore);

        List<Map<String, Object>> playersOut = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            PlayerState ps = playerStates[i];
            Map<String, Object> p = new LinkedHashMap<>();
            float dur = duration > 0 ? duration / 60.0f : 1;

            p.put("player_slot", ps.playerSlot);
            p.put("hero_id", ps.heroId);
            p.put("hero_name", ps.heroName != null && !ps.heroName.isEmpty() ? ps.heroName : "unknown_" + i);
            p.put("hero_npc_name", ps.heroNpcName != null ? ps.heroNpcName : "");
            p.put("player_name", ps.playerName != null && !ps.playerName.isEmpty() ? ps.playerName : "Player " + (i + 1));
            p.put("steam_id", ps.steamId);
            p.put("kills", ps.kills);
            p.put("deaths", ps.deaths);
            p.put("assists", ps.assists);
            p.put("level", ps.level);
            p.put("last_hits", ps.lastHits);
            p.put("denies", ps.denies);
            p.put("gold_per_min", (int) (ps.totalGold / dur));
            p.put("xp_per_min", (int) (ps.totalXp / dur));
            p.put("hero_damage", ps.heroDamage);
            p.put("tower_damage", ps.towerDamage);
            p.put("hero_healing", ps.heroHealing);
            p.put("net_worth", ps.netWorth > 0 ? ps.netWorth : ps.totalGold);
            p.put("net_worth_source", ps.netWorth > 0 ? "entity" : "earned_gold");
            p.put("total_gold", ps.totalGold);

            stripInternalKeys(ps.purchaseLog);
            p.put("purchase_log", ps.purchaseLog);
            p.put("ability_casts", ps.abilityCasts);
            p.put("wards_placed", new ArrayList<>());
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

            p.put("gold_t", goldTimeline.getOrDefault(i, new ArrayList<>()));
            p.put("xp_t", xpTimeline.getOrDefault(i, new ArrayList<>()));
            p.put("lh_t", lhTimeline.getOrDefault(i, new ArrayList<>()));

            p.put("killed", ps.killed);
            p.put("killed_by", ps.killedBy);
            p.put("damage", ps.damageDealt);
            p.put("damage_taken", ps.damageTaken);
            p.put("damage_inflictor", ps.damageInflictor);

            p.put("gold_reasons", ps.goldReasons);
            p.put("xp_reasons", ps.xpReasons);

            p.put("buyback_count", ps.buybackCount);
            p.put("roshans_killed", ps.roshansKilled);
            p.put("courier_kills", ps.courierKills);
            p.put("observer_kills", ps.observerKills);
            p.put("sentry_kills", ps.sentryKills);
            p.put("tower_kills", ps.towerKills);
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
            Integer destroyed = null;
            if (w.destroyedRaw != null && !(gameEnded && w.destroyedRaw > gameEndTime + 1f)) {
                destroyed = toGameTime(w.destroyedRaw);
            }
            ward.put("destroyed_time", destroyed);
            ward.put("ehandle", w.ehandle);
            ward.put("owner_source", w.ownerSource);
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
            Float deathRaw = b.name != null ? buildingDeathRaw.get(b.name) : null;
            if (deathRaw == null && b.destroyedRaw != null
                    && !(gameEnded && b.destroyedRaw > gameEndTime + 1f)) {
                deathRaw = b.destroyedRaw;
            }
            bo.put("destroyed_time", deathRaw != null ? toGameTime(deathRaw) : null);
            buildingsOut.add(bo);
        }
        result.put("buildings", buildingsOut);

        // ---- Gold/XP advantage ----
        List<Integer> goldAdv = new ArrayList<>();
        List<Integer> xpAdv = new ArrayList<>();
        int minLen = Integer.MAX_VALUE;
        for (int i = 0; i < 10; i++) {
            int sz = goldTimeline.getOrDefault(i, Collections.emptyList()).size();
            if (sz > 0) minLen = Math.min(minLen, sz);
        }
        if (minLen < Integer.MAX_VALUE) {
            for (int m = 0; m < minLen; m++) {
                int rGold = 0, dGold = 0, rXp = 0, dXp = 0;
                for (int i = 0; i < 10; i++) {
                    List<Integer> gl = goldTimeline.getOrDefault(i, Collections.emptyList());
                    List<Integer> xl = xpTimeline.getOrDefault(i, Collections.emptyList());
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
        if (draft.isEmpty()) {
            draft = draftEvents;
            boolean hasPicks = draft.stream().anyMatch(e -> Boolean.TRUE.equals(e.get("is_pick")));
            if (!hasPicks) {
                for (int i = 0; i < 10; i++) {
                    PlayerState ps = playerStates[i];
                    if (ps.heroName != null && !ps.heroName.isEmpty()) {
                        Map<String, Object> entry = new LinkedHashMap<>();
                        entry.put("is_pick", true);
                        entry.put("hero_id", ps.heroId);
                        entry.put("hero_name", ps.heroName);
                        entry.put("team", i < 5 ? 0 : 1);
                        entry.put("order", draftOrder++);
                        draft.add(entry);
                    }
                }
            }
        }
        result.put("picks_bans", draft);

        // Building bitmasks are derived by the Python side from "buildings"/"objectives".
        result.put("tower_status_radiant", 0);
        result.put("tower_status_dire", 0);
        result.put("barracks_status_radiant", 0);
        result.put("barracks_status_dire", 0);

        int firstBlood = 0;
        for (KillEvent k : realDeathsSorted()) {
            if (!k.deny) {
                firstBlood = Math.max(0, toGameTime(k.time));
                break;
            }
        }
        result.put("first_blood_time", firstBlood);
        result.put("region", 0);
        result.put("patch", "");
        result.put("parse_errors", errorCounts);
        return result;
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

    private static class PlayerState {
        final int playerIdx;
        final int playerSlot;
        String heroName = "";
        String heroNpcName = "";
        int heroId = 0;
        String playerName = "";
        long steamId = 0;
        int kills = 0, deaths = 0, assists = 0;
        int level = 1;
        int lastHits = 0, denies = 0;
        int totalGold = 0, totalXp = 0, netWorth = 0;
        int heroDamage = 0, towerDamage = 0, heroHealing = 0;
        int buybackCount = 0, roshansKilled = 0, courierKills = 0;
        int observerKills = 0, sentryKills = 0, towerKills = 0;
        int laneKills = 0, neutralKills = 0, ancientKills = 0;
        float lastX = 0, lastY = 0;
        boolean hasPosition = false;
        Entity heroEntity = null;
        int heroEntityIndex = -1;
        boolean heroEntityDeleted = false;

        // Position history for smoke paths and teamfight positions: [rawCombatLogTime, worldX, worldY]
        List<float[]> positionHistory = new ArrayList<>();
        float lastPositionRecordTime = -999;

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

        List<Map<String, Object>> purchaseLog = new ArrayList<>();
        List<Map<String, Object>> abilityCasts = new ArrayList<>();
        List<LevelUpEvent> levelUpTimes = new ArrayList<>();
        List<Map<String, Object>> abilityUpgrades = new ArrayList<>();       // from cast levels
        Map<String, Integer> abilityLevels = new LinkedHashMap<>();
        List<Map<String, Object>> entityAbilityUpgrades = new ArrayList<>(); // from m_iLevel
        Map<Integer, AbilityTrack> abilityByHandle = new HashMap<>();
        int entityUpgradeCount = 0;
        int lastAbilityPoints = -1;
        int lastPolledLevel = 1;
        int pendingSpentPoints = 0;
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

    private static class KillEvent {
        final float time;
        final int attackerIdx;
        final int targetIdx;
        final boolean deny;
        float x, y;
        boolean hasPos = false;
        List<Integer> assistRaw = null;

        KillEvent(float time, int attackerIdx, int targetIdx, boolean deny) {
            this.time = time;
            this.attackerIdx = attackerIdx;
            this.targetIdx = targetIdx;
            this.deny = deny;
        }
    }

    private static class WardRecord {
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

        WardRecord(int entityIndex, String type, float placedRaw) {
            this.entityIndex = entityIndex;
            this.type = type;
            this.placedRaw = placedRaw;
        }
    }

    private static class BuildingRecord {
        int entityIndex;
        int entityHandle;
        String name = "";
        String team;
        String type;
        float x, y;
        boolean hasPos = false;
        Float destroyedRaw = null;
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

    private static class AbilityTrack {
        final String name;
        int level;
        final int slot;

        AbilityTrack(String name, int level, int slot) {
            this.name = name;
            this.level = level;
            this.slot = slot;
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

    private static String shortMessage(Throwable t) {
        String msg = t.getMessage();
        String s = t.getClass().getSimpleName() + (msg != null ? ": " + msg : "");
        return s.length() > 300 ? s.substring(0, 300) + "..." : s;
    }

    private void printSummary(Map<String, Object> output) {
        System.err.println("=== Parse Summary ===");
        System.err.println("match_id=" + matchId + " duration=" + duration + "s radiant_win=" + radiantWin
                + " partial=" + partial + " entity_pid_scheme=" + entityPidScheme
                + " event_pid_scheme=" + eventPidScheme);
        for (int i = 0; i < 10; i++) {
            PlayerState ps = playerStates[i];
            System.err.println("  [" + i + "] " + ps.heroName + " " + ps.kills + "/" + ps.deaths + "/" + ps.assists
                    + " LH:" + ps.lastHits + " NW:" + ps.netWorth + " upgrades:" + ps.entityAbilityUpgrades.size()
                    + " inventory:" + (ps.inventoryCaptured ? "yes" : "no"));
        }
        long owned = wardRecords.stream().filter(w -> w.ownerIdx >= 0).count();
        System.err.println("wards=" + wardRecords.size() + " (owned " + owned + ") smokes=" + smokeGroups.size()
                + " buildings=" + buildingsByEntityIndex.size() + " deaths=" + killLog.size()
                + " objectives=" + objectiveEvents.size());
        if (!errorCounts.isEmpty()) System.err.println("handler errors: " + errorCounts);
    }

    public static void main(String[] args) {
        if (args.length < 1) {
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
            if (analyzer.lastCtx != null) {
                analyzer.readTeamData(analyzer.lastCtx);
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
