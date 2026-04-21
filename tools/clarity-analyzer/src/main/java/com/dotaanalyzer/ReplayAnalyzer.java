package com.dotaanalyzer;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import skadistats.clarity.model.CombatLogEntry;
import skadistats.clarity.model.Entity;
import skadistats.clarity.model.FieldPath;
import skadistats.clarity.processor.entities.Entities;
import skadistats.clarity.processor.entities.OnEntityEntered;
import skadistats.clarity.processor.entities.OnEntityUpdated;
import skadistats.clarity.processor.entities.OnEntityUpdatesCompleted;
import skadistats.clarity.processor.entities.UsesEntities;
import skadistats.clarity.processor.gameevents.OnCombatLogEntry;
import skadistats.clarity.processor.reader.OnMessage;
import skadistats.clarity.processor.reader.OnTickStart;
import skadistats.clarity.processor.runner.Context;
import skadistats.clarity.processor.runner.SimpleRunner;
import skadistats.clarity.source.MappedFileSource;
import skadistats.clarity.wire.dota.common.proto.DOTAUserMessages.CDOTAUserMsg_ChatMessage;
import skadistats.clarity.wire.dota.common.proto.DOTAUserMessages.CDOTAUserMsg_ChatWheel;
import skadistats.clarity.wire.shared.s2.proto.S2UserMessages.CUserMessageSayText2;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Dota 2 Replay Analyzer using the Clarity 3.x library.
 * Parses .dem replay files and outputs comprehensive JSON match data.
 *
 * Key findings from replay analysis:
 *  - Entity m_iPlayerID uses values 0,2,4,6,8 (radiant) and 10,12,14,16,18 (dire)
 *    -> player index = playerID / 2
 *  - Game time is NOT available as an entity property (m_fGameTime/m_flGameTime don't exist)
 *    -> Use combat log timestamps exclusively
 *  - Game state transitions come from DOTA_COMBATLOG_GAME_STATE events:
 *    state 5 = game in progress, state 6 = post-game
 *  - m_iLastHitCount is on team data entities (CDOTA_DataRadiant/Dire), not hero entities
 *  - Entity class "CDOTA_Unit_Hero_DrowRanger" -> hero name "drow_ranger" (CamelCase to snake_case)
 */
@UsesEntities
public class ReplayAnalyzer {

    // ---- Match-level data ----
    private long matchId = 0;
    private int duration = 0;
    private boolean radiantWin = false;
    private int gameMode = 0;
    private long startTime = 0;

    // ---- Per-player data (indexed 0-9, 0-4 radiant, 5-9 dire) ----
    private final PlayerState[] playerStates = new PlayerState[10];

    // ---- Events ----
    private final List<Map<String, Object>> wardEvents = new ArrayList<>();
    private final List<Map<String, Object>> objectiveEvents = new ArrayList<>();
    private final List<Map<String, Object>> smokeEvents = new ArrayList<>();

    // ---- Time-series data ----
    private final Map<Integer, List<Integer>> goldTimeline = new HashMap<>();
    private final Map<Integer, List<Integer>> xpTimeline = new HashMap<>();
    private final Map<Integer, List<Integer>> lhTimeline = new HashMap<>();

    // ---- Teamfight detection ----
    private final List<KillEvent> killLog = new ArrayList<>();

    // ---- Game time tracking (from combat log only) ----
    private float gameStartTime = 0;   // combat log timestamp when state=5
    private float gameEndTime = 0;     // combat log timestamp when state=6
    private boolean gameStarted = false;
    private float lastCombatLogTime = 0;
    private int lastMinuteRecorded = -1;
    private int lastLaneSnapshotSecond = -5;

    // ---- Starting items snapshot ----
    private boolean startingItemsCollected = false;

    // ---- Ward dedup ----
    private final Set<String> seenWards = new HashSet<>();

    // ---- Ward purchase tracking: correlate purchases with ward entities ----
    // Each entry: [playerIdx, gameTime, wardType("observer"/"sentry")]
    private final List<WardPurchase> recentWardPurchases = new ArrayList<>();

    // ---- Smoke tracking ----
    // Track active smokes: playerIdx -> start time (combat log time)
    private final Map<Integer, Float> activeSmokeStartTime = new HashMap<>();
    // Track active smoke start position: playerIdx -> [x, y]
    private final Map<Integer, float[]> activeSmokeStartPosition = new HashMap<>();
    // Track active smoke: playerIdx -> players involved (same team, smoked at same time window)
    private final Map<Integer, List<Integer>> activeSmokeGroup = new HashMap<>();

    // ---- Draft tracking ----
    private final int[] draftHeroes = new int[24]; // 0-13 = bans, 14-23 = picks
    private final boolean[] draftOrderProcessed = new boolean[24];
    private int draftOrder = 0;
    private final List<Map<String, Object>> draftEvents = new ArrayList<>();

    // ---- Chat tracking ----
    private final List<Map<String, Object>> chatMessages = new ArrayList<>();

    // ---- Hero name mapping: combat log name -> player index ----
    private final Map<String, Integer> heroNameMap = new HashMap<>();

    // ---- Entities context for lookups ----
    private Context lastCtx = null;

    // ---- Consumable items to exclude from final items ----
    private static final Set<String> CONSUMABLE_ITEMS = new HashSet<>(Arrays.asList(
        "tango", "flask", "clarity", "enchanted_mango", "faerie_fire",
        "ward_observer", "ward_sentry", "ward_dispenser", "dust",
        "smoke_of_deceit", "tome_of_knowledge", "tpscroll",
        "blood_grenade", "observer_and_sentry_wards"
    ));

    public ReplayAnalyzer() {
        for (int i = 0; i < 10; i++) {
            playerStates[i] = new PlayerState(i);
        }
    }

    // ======== Player ID Mapping ========

    /**
     * Convert entity m_iPlayerID to our 0-9 player index.
     * Entity player IDs: 0,2,4,6,8 (radiant), 10,12,14,16,18 (dire)
     * Our index:          0,1,2,3,4 (radiant), 5,6,7,8,9 (dire)
     */
    private static int entityPlayerIdToIndex(int entityPlayerId) {
        if (entityPlayerId < 0) return -1;
        int idx = entityPlayerId / 2;
        if (idx >= 0 && idx < 10) return idx;
        return -1;
    }

    // ======== Combat Log Processing ========

    @OnCombatLogEntry
    public void onCombatLog(Context ctx, CombatLogEntry cle) {
        lastCtx = ctx;
        float time = cle.getTimestamp();
        lastCombatLogTime = Math.max(lastCombatLogTime, time);

        // Track per-minute snapshots from combat log time
        if (gameStarted) {
            float gameTime = time - gameStartTime;
            int minute = (int) (gameTime / 60);
            if (minute > lastMinuteRecorded && minute >= 0 && minute < 200) {
                // Record snapshot before advancing
                readTeamData(ctx);
                lastMinuteRecorded = minute;
                recordMinuteSnapshot(minute);
            }

            // Laning phase snapshots (every 5 seconds, 0-600s)
            int gameSec = (int)(gameTime);
            if (gameSec >= 0 && gameSec <= 600 && gameSec >= lastLaneSnapshotSecond + 5) {
                recordLaningSnapshot(gameSec);
            }
        }

        try {
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
            // Skip malformed combat log entries
        }
    }

    private void handleGameState(CombatLogEntry cle, float time) {
        try {
            int state = cle.getValue();
            System.err.println("Game state: " + state + " at time " + time);

            // State 5 = DOTA_GAMERULES_STATE_GAME_IN_PROGRESS
            if (state == 5 && !gameStarted) {
                gameStarted = true;
                gameStartTime = time;
                System.err.println("Game started at combat log time " + time);
                // Snapshot starting items from hero entity inventory
                snapshotStartingItems();
            }

            // State 6 = DOTA_GAMERULES_STATE_POST_GAME
            if (state == 6 && gameStarted) {
                gameEndTime = time;
                System.err.println("Game ended at combat log time " + time);
            }
        } catch (Exception e) {
            // skip
        }
    }

    /**
     * Snapshot each hero's inventory at game start to capture starting items.
     * Reads m_hItems.NNNN handles from hero entities and resolves them to item names.
     * Slots 0-5 = main inventory, 6-8 = backpack, 9-15 = stash.
     * We read slots 0-8 (inventory + backpack) as starting items.
     */
    private void snapshotStartingItems() {
        if (startingItemsCollected || lastCtx == null) return;
        startingItemsCollected = true;

        Entities entities = lastCtx.getProcessor(Entities.class);

        for (int i = 0; i < 10; i++) {
            PlayerState ps = playerStates[i];
            if (ps == null || ps.heroEntity == null) continue;

            Entity hero = ps.heroEntity;
            for (int slot = 0; slot <= 8; slot++) {
                String prop = "m_hItems." + String.format("%04d", slot);
                try {
                    Integer handle = tryGetInt(hero, prop);
                    if (handle == null || handle == 16777215) continue; // 0xFFFFFF = no item

                    Entity itemEntity = entities.getByHandle(handle);
                    if (itemEntity == null) continue;

                    // Extract item name from entity class: "CDOTA_Item_Tango" -> "tango"
                    String dtName = itemEntity.getDtClass().getDtName();
                    String itemName = entityClassToItemName(dtName);
                    if (itemName != null && !itemName.isEmpty()) {
                        ps.startingItems.add(itemName);
                    }
                } catch (Exception ex) {
                    // skip unreadable slots
                }
            }
            if (!ps.startingItems.isEmpty()) {
                System.err.println("Starting items for player " + i + " (" + ps.heroName + "): " + ps.startingItems);
            }
        }
    }

    // Entity class suffix -> CDN/API item key mapping for names that don't follow
    // simple CamelCase-to-snake_case conversion
    private static final Map<String, String> ITEM_NAME_ALIASES = new HashMap<>();
    static {
        ITEM_NAME_ALIASES.put("IronwoodBranch", "branches");
        ITEM_NAME_ALIASES.put("Ward_Dispenser", "ward_observer");
        ITEM_NAME_ALIASES.put("Ward_Sentry", "ward_sentry");
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
    }

    /**
     * Convert item entity class name to CDN-compatible item key.
     * "CDOTA_Item_Tango" -> "tango"
     * "CDOTA_Item_MagicStick" -> "magic_stick"
     * "CDOTA_Item_Faerie_Fire" -> "faerie_fire"
     * "CDOTA_Item_IronwoodBranch" -> "branches"
     */
    private static String entityClassToItemName(String dtClassName) {
        if (dtClassName == null) return null;
        String prefix = "CDOTA_Item_";
        if (!dtClassName.startsWith(prefix)) return null;
        String suffix = dtClassName.substring(prefix.length());
        if (suffix.isEmpty()) return null;

        // Check alias table first
        String alias = ITEM_NAME_ALIASES.get(suffix);
        if (alias != null) return alias;

        // Convert CamelCase to snake_case, but handle existing underscores
        // e.g. "Faerie_Fire" -> "faerie_fire", not "faerie__fire"
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

    private void handleDeath(CombatLogEntry cle, float time) {
        String targetRaw = cle.getTargetName();
        String attackerRaw = cle.getAttackerName();
        String target = cleanHeroName(targetRaw);
        String attacker = cleanHeroName(attackerRaw);

        // Check if target is a hero (not creep, building, etc.)
        boolean targetIsHero = false;
        try {
            targetIsHero = cle.isTargetHero();
        } catch (Exception e) {
            targetIsHero = targetRaw != null && targetRaw.contains("npc_dota_hero_");
        }

        // Check if target is an illusion
        boolean targetIsIllusion = false;
        try {
            targetIsIllusion = cle.isTargetIllusion();
        } catch (Exception e) {}

        int targetIdx = heroNameToPlayerIdx(target);
        int attackerIdx = heroNameToPlayerIdx(attacker);

        // Only count real hero kills/deaths (not illusions, not creeps)
        if (targetIsHero && !targetIsIllusion && targetIdx >= 0) {
            playerStates[targetIdx].deaths++;

            if (attackerIdx >= 0) {
                playerStates[attackerIdx].kills++;
                playerStates[attackerIdx].killed.merge(target, 1, Integer::sum);
                playerStates[targetIdx].killedBy.merge(attacker, 1, Integer::sum);
                killLog.add(new KillEvent(time, attackerIdx, targetIdx));
            }

            // Track assists from combat log
            // Note: assist player IDs are already 0-9 indices (NOT entity player IDs)
            try {
                if (cle.hasAssistPlayers()) {
                    List<Integer> assistPlayers = cle.getAssistPlayers();
                    if (assistPlayers != null) {
                        for (int assistIdx : assistPlayers) {
                            if (assistIdx >= 0 && assistIdx < 10 && assistIdx != attackerIdx) {
                                playerStates[assistIdx].assists++;
                            }
                        }
                    }
                }
            } catch (Exception e) {
                // Assist tracking not available
            }
        }

        // Track objective kills (buildings, roshan, courier)
        String targetClean = cleanName(targetRaw);
        if (targetClean.contains("tower") || targetClean.contains("rax") || targetClean.contains("fort")) {
            Map<String, Object> obj = new LinkedHashMap<>();
            int gameTime = gameStarted ? (int)(time - gameStartTime) : (int)time;
            obj.put("time", gameTime);
            obj.put("type", targetClean.contains("tower") ? "tower_kill" :
                    targetClean.contains("rax") ? "barracks_kill" : "ancient_kill");
            obj.put("key", targetClean);
            obj.put("team", attackerIdx >= 0 && attackerIdx < 5 ? "radiant" : "dire");
            if (attackerIdx >= 0) obj.put("player_slot", playerStates[attackerIdx].playerSlot);
            objectiveEvents.add(obj);
        }
        if (targetClean.contains("roshan")) {
            Map<String, Object> obj = new LinkedHashMap<>();
            int gameTime = gameStarted ? (int)(time - gameStartTime) : (int)time;
            obj.put("time", gameTime);
            obj.put("type", "roshan_kill");
            obj.put("team", attackerIdx >= 0 && attackerIdx < 5 ? "radiant" : "dire");
            objectiveEvents.add(obj);
            if (attackerIdx >= 0) playerStates[attackerIdx].roshansKilled++;
        }
        if (targetClean.contains("courier")) {
            if (attackerIdx >= 0) playerStates[attackerIdx].courierKills++;
        }

        // Track creep kills during laning phase (0-600s)
        if (!targetIsHero && gameStarted) {
            float gameTime = time - gameStartTime;
            if (gameTime >= 0 && gameTime <= 600) {
                String targetName = cle.getTargetName();

                // Only track lane creeps and neutral creeps
                if (targetName != null && (targetName.contains("creep") || targetName.contains("neutral"))) {
                    int creepAttackerIdx = heroNameToPlayerIdx(attacker);
                    if (creepAttackerIdx >= 0 && creepAttackerIdx < 10) {
                        PlayerState ps = playerStates[creepAttackerIdx];

                        // Determine creep type
                        String creepType = "unknown";
                        if (targetName.contains("melee")) creepType = "melee";
                        else if (targetName.contains("ranged")) creepType = "ranged";
                        else if (targetName.contains("siege")) creepType = "siege";
                        else if (targetName.contains("neutral")) creepType = "neutral";

                        // Check if it's a deny (attacker and target on same team)
                        // Radiant creeps: "goodguys", Dire creeps: "badguys"
                        boolean targetIsRadiant = targetName.contains("goodguys");
                        boolean attackerIsRadiant = creepAttackerIdx < 5;
                        boolean isDeny = (targetIsRadiant == attackerIsRadiant);

                        Map<String, Object> event = new LinkedHashMap<>();
                        event.put("time", Math.round(gameTime));
                        event.put("creep_type", creepType);
                        event.put("gold", (int) cle.getValue());

                        if (isDeny) {
                            ps.laningDenies.add(event);
                        } else {
                            ps.laningCreepKills.add(event);
                        }
                    }
                }
            }
        }
    }

    private void handleDamage(CombatLogEntry cle, float time) {
        String target = cleanHeroName(cle.getTargetName());
        String attacker = cleanHeroName(cle.getAttackerName());
        int value = cle.getValue();
        String inflictor = cleanName(cle.getInflictorName());

        int attackerIdx = heroNameToPlayerIdx(attacker);
        int targetIdx = heroNameToPlayerIdx(target);

        // Only count hero-to-hero damage (not illusions)
        boolean targetIsHero = false;
        try {
            targetIsHero = cle.isTargetHero();
        } catch (Exception e) {
            targetIsHero = cle.getTargetName() != null && cle.getTargetName().contains("npc_dota_hero_");
        }

        boolean targetIsIllusion = false;
        try {
            targetIsIllusion = cle.isTargetIllusion();
        } catch (Exception e) {}

        if (attackerIdx >= 0 && targetIdx >= 0 && targetIsHero && !targetIsIllusion) {
            playerStates[attackerIdx].heroDamage += value;
            playerStates[attackerIdx].damageDealt.merge(target, value, Integer::sum);
            playerStates[targetIdx].damageTaken.merge(attacker, value, Integer::sum);
            if (!inflictor.isEmpty()) {
                playerStates[attackerIdx].damageInflictor.merge(inflictor, value, Integer::sum);
            }

            // Log for teamfight damage computation
            playerStates[attackerIdx].heroDamageDealtLog.add(new float[]{time, value});
            playerStates[targetIdx].heroDamageTakenLog.add(new float[]{time, value});

            // Track individual hero damage events during laning phase
            if (gameStarted) {
                float gameTime = time - gameStartTime;
                if (gameTime >= 0 && gameTime <= 600) {
                    Map<String, Object> evt = new LinkedHashMap<>();
                    evt.put("time", Math.round(gameTime));
                    evt.put("target_idx", targetIdx);
                    evt.put("damage", (int) cle.getValue());
                    if (!inflictor.isEmpty()) {
                        evt.put("inflictor", inflictor);
                    }
                    playerStates[attackerIdx].laningDamageEvents.add(evt);
                }
            }
        }

        String targetClean = cleanName(cle.getTargetName());
        if (attackerIdx >= 0 && (targetClean.contains("tower") || targetClean.contains("fort") || targetClean.contains("rax"))) {
            playerStates[attackerIdx].towerDamage += value;
        }
    }

    private void handleHeal(CombatLogEntry cle, float time) {
        String source = cleanHeroName(cle.getAttackerName());
        int value = cle.getValue();
        int srcIdx = heroNameToPlayerIdx(source);

        boolean targetIsHero = false;
        try {
            targetIsHero = cle.isTargetHero();
        } catch (Exception e) {
            targetIsHero = cle.getTargetName() != null && cle.getTargetName().contains("npc_dota_hero_");
        }

        if (srcIdx >= 0 && targetIsHero) {
            playerStates[srcIdx].heroHealing += value;
            // Log for teamfight healing computation
            playerStates[srcIdx].heroHealLog.add(new float[]{time, value});
        }
    }

    private void handlePurchase(CombatLogEntry cle, float time) {
        String buyer = cleanHeroName(cle.getTargetName());
        String item = cleanName(cle.getValueName());
        int buyerIdx = heroNameToPlayerIdx(buyer);
        if (buyerIdx >= 0) {
            Map<String, Object> purchase = new LinkedHashMap<>();
            // Store raw combat log time; will be converted to game time in buildOutput()
            purchase.put("raw_time", time);
            purchase.put("key", item);
            playerStates[buyerIdx].purchaseLog.add(purchase);

            // Track ward purchases for correlating with ward entities
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
        if (casterIdx >= 0) {
            int gameTime = gameStarted ? (int)(time - gameStartTime) : (int)time;
            Map<String, Object> cast = new LinkedHashMap<>();
            cast.put("name", ability);
            cast.put("time", gameTime);
            playerStates[casterIdx].abilityCasts.add(cast);

            // Track ability levels for ability upgrade detection
            // When we see an ability used at a new (higher) level, record it as an upgrade
            try {
                if (cle.hasAbilityLevel()) {
                    int abilityLevel = cle.getAbilityLevel();
                    if (abilityLevel > 0 && !ability.isEmpty()) {
                        PlayerState ps = playerStates[casterIdx];
                        int prevLevel = ps.abilityLevels.getOrDefault(ability, 0);
                        if (abilityLevel > prevLevel) {
                            // New ability level detected - record all levels from prevLevel+1 to abilityLevel
                            for (int lvl = prevLevel + 1; lvl <= abilityLevel; lvl++) {
                                Map<String, Object> upgrade = new LinkedHashMap<>();
                                upgrade.put("name", ability);
                                upgrade.put("level", lvl);
                                upgrade.put("time", gameTime);
                                ps.abilityUpgrades.add(upgrade);
                            }
                            ps.abilityLevels.put(ability, abilityLevel);
                        }
                    }
                }
            } catch (Exception e) {
                // ability level not available
            }

            // Track regen item usage during laning
            if (gameStarted) {
                float gameTimeLaning = time - gameStartTime;
                if (gameTimeLaning >= 0 && gameTimeLaning <= 600) {
                    String abilityName = ability;
                    if (abilityName != null && (
                        abilityName.contains("tango") ||
                        abilityName.contains("flask") ||
                        abilityName.contains("clarity") ||
                        abilityName.contains("mango") ||
                        abilityName.contains("bottle") ||
                        abilityName.contains("faerie_fire") ||
                        abilityName.contains("healing_salve"))) {
                        Map<String, Object> evt = new LinkedHashMap<>();
                        evt.put("time", Math.round(gameTimeLaning));
                        evt.put("item", abilityName);
                        playerStates[casterIdx].laningRegenEvents.add(evt);
                    }
                }
            }
        }
    }

    private void handleItem(CombatLogEntry cle, float time) {
        // Track regen/consumable item usage during laning phase
        if (!gameStarted) return;
        float gameTime = time - gameStartTime;
        if (gameTime < 0 || gameTime > 600) return;

        String caster = cleanHeroName(cle.getAttackerName());
        int casterIdx = heroNameToPlayerIdx(caster);
        if (casterIdx < 0 || casterIdx >= 10) return;

        String item = cleanName(cle.getInflictorName());
        if (item == null || item.isEmpty()) return;

        if (item.contains("tango") || item.contains("flask") || item.contains("clarity")
                || item.contains("mango") || item.contains("bottle") || item.contains("faerie_fire")
                || item.contains("salve")) {
            Map<String, Object> evt = new LinkedHashMap<>();
            evt.put("time", Math.round(gameTime));
            evt.put("item", item);
            playerStates[casterIdx].laningRegenEvents.add(evt);
        }
    }

    private void handleGold(CombatLogEntry cle, float time) {
        String target = cleanHeroName(cle.getTargetName());
        int value = cle.getValue();
        int idx = heroNameToPlayerIdx(target);
        if (idx >= 0) {
            try {
                int reason = cle.getGoldReason();
                String reasonKey = String.valueOf(reason);
                playerStates[idx].goldReasons.merge(reasonKey, value, Integer::sum);
            } catch (Exception e) {}
        }
    }

    private void handleXp(CombatLogEntry cle, float time) {
        String target = cleanHeroName(cle.getTargetName());
        int value = cle.getValue();
        int idx = heroNameToPlayerIdx(target);
        if (idx >= 0) {
            try {
                int reason = cle.getXpReason();
                String reasonKey = String.valueOf(reason);
                playerStates[idx].xpReasons.merge(reasonKey, value, Integer::sum);
            } catch (Exception e) {}
        }
    }

    private void handleBuyback(CombatLogEntry cle, float time) {
        String target = cleanHeroName(cle.getTargetName());
        int idx = heroNameToPlayerIdx(target);
        if (idx >= 0) {
            playerStates[idx].buybackCount++;
        }
    }

    /**
     * Handle modifier add events - used for smoke detection and ability upgrade tracking.
     */
    private void handleModifierAdd(CombatLogEntry cle, float time) {
        try {
            String inflictorRaw = cle.getInflictorName();
            String inflictor = inflictorRaw != null ? inflictorRaw : "";

            // Smoke detection: modifier_smoke_of_deceit or similar
            if (inflictor.contains("smoke_of_deceit") || inflictor.contains("modifier_smoke")) {
                String targetRaw = cle.getTargetName();
                String target = cleanHeroName(targetRaw);
                int targetIdx = heroNameToPlayerIdx(target);
                if (targetIdx >= 0) {
                    activeSmokeStartTime.put(targetIdx, time);
                    activeSmokeStartPosition.put(targetIdx, new float[]{
                        playerStates[targetIdx].lastX, playerStates[targetIdx].lastY
                    });
                    System.err.println("Smoke applied to " + target + " (idx=" + targetIdx + ") at time " + time +
                        " pos=(" + playerStates[targetIdx].lastX + "," + playerStates[targetIdx].lastY + ")");
                }
            }
        } catch (Exception e) {
            // skip
        }
    }

    /**
     * Handle modifier remove events - used for smoke break detection.
     */
    private void handleModifierRemove(CombatLogEntry cle, float time) {
        try {
            String inflictorRaw = cle.getInflictorName();
            String inflictor = inflictorRaw != null ? inflictorRaw : "";

            // Smoke break detection
            if (inflictor.contains("smoke_of_deceit") || inflictor.contains("modifier_smoke")) {
                String targetRaw = cle.getTargetName();
                String target = cleanHeroName(targetRaw);
                int targetIdx = heroNameToPlayerIdx(target);
                if (targetIdx >= 0 && activeSmokeStartTime.containsKey(targetIdx)) {
                    float smokeStartCombatTime = activeSmokeStartTime.remove(targetIdx);
                    float[] startPos = activeSmokeStartPosition.remove(targetIdx);
                    int smokeStartGameTime = gameStarted ? (int)(smokeStartCombatTime - gameStartTime) : (int)smokeStartCombatTime;
                    int smokeEndGameTime = gameStarted ? (int)(time - gameStartTime) : (int)time;

                    // Start and end positions
                    float startX = startPos != null ? startPos[0] : 0;
                    float startY = startPos != null ? startPos[1] : 0;
                    float endX = playerStates[targetIdx].lastX;
                    float endY = playerStates[targetIdx].lastY;

                    // Determine team
                    String team = targetIdx < 5 ? "radiant" : "dire";
                    int playerSlot = playerStates[targetIdx].playerSlot;

                    // Check if this smoke event is part of an existing group smoke
                    // (multiple heroes smoked at the same time)
                    boolean merged = false;
                    for (Map<String, Object> existingSmoke : smokeEvents) {
                        int existingStart = (int) existingSmoke.get("time");
                        String existingTeam = (String) existingSmoke.get("team");
                        if (existingTeam.equals(team) && Math.abs(existingStart - smokeStartGameTime) <= 2) {
                            // Same team, same smoke usage - add this player
                            @SuppressWarnings("unchecked")
                            List<Integer> slots = (List<Integer>) existingSmoke.get("player_slots");
                            if (!slots.contains(playerSlot)) {
                                slots.add(playerSlot);
                            }
                            merged = true;
                            break;
                        }
                    }

                    if (!merged) {
                        // Build path from position history during smoke duration
                        List<List<Float>> path = new ArrayList<>();
                        for (float[] pos : playerStates[targetIdx].positionHistory) {
                            if (pos[0] >= smokeStartCombatTime && pos[0] <= time) {
                                List<Float> waypoint = new ArrayList<>();
                                waypoint.add(pos[1]);
                                waypoint.add(pos[2]);
                                path.add(waypoint);
                            }
                        }
                        // Add end position as final waypoint
                        List<Float> endWaypoint = new ArrayList<>();
                        endWaypoint.add(endX);
                        endWaypoint.add(endY);
                        path.add(endWaypoint);

                        // Determine smoke result by checking kills within 30s after break
                        String result = "no_contact";
                        int killsGained = 0, deathsSuffered = 0;
                        boolean smokeTeamIsRadiant = team.equals("radiant");
                        for (KillEvent k : killLog) {
                            float killGameTime = gameStarted ? k.time - gameStartTime : k.time;
                            if (killGameTime >= smokeEndGameTime - 5 && killGameTime <= smokeEndGameTime + 30) {
                                boolean attackerIsRadiant = k.attackerIdx < 5;
                                if (attackerIsRadiant == smokeTeamIsRadiant) {
                                    killsGained++;
                                } else {
                                    deathsSuffered++;
                                }
                            }
                        }
                        if (killsGained > 0 && deathsSuffered > 0) {
                            result = "traded";
                        } else if (killsGained > 0) {
                            result = "kill";
                        }

                        Map<String, Object> smoke = new LinkedHashMap<>();
                        smoke.put("time", smokeStartGameTime);
                        smoke.put("end_time", smokeEndGameTime);
                        smoke.put("team", team);
                        smoke.put("x", (double) startX);
                        smoke.put("y", (double) startY);
                        smoke.put("path", path);
                        smoke.put("result", result);
                        smoke.put("kills_gained", killsGained);
                        smoke.put("deaths_suffered", deathsSuffered);
                        smoke.put("player_slot", playerSlot);
                        List<Integer> slots = new ArrayList<>();
                        slots.add(playerSlot);
                        smoke.put("player_slots", slots);
                        smokeEvents.add(smoke);
                    }

                    System.err.println("Smoke ended for " + target + " (idx=" + targetIdx +
                        ") start=" + smokeStartGameTime + " end=" + smokeEndGameTime +
                        " startPos=(" + startX + "," + startY + ") endPos=(" + endX + "," + endY + ")");
                }
            }
        } catch (Exception e) {
            // skip
        }
    }

    /**
     * Handle hero level-up events from combat log.
     * DOTA_COMBATLOG_HERO_LEVELUP fires when a hero gains a level.
     * The attacker is the hero, and we track which ability was leveled
     * using the ability level info from the combat log entry.
     */
    private void handleHeroLevelUp(CombatLogEntry cle, float time) {
        try {
            // The hero who leveled up is in the target name
            String targetRaw = cle.getTargetName();
            String target = cleanHeroName(targetRaw);
            int targetIdx = heroNameToPlayerIdx(target);

            if (targetIdx >= 0) {
                int gameTime = gameStarted ? (int)(time - gameStartTime) : (int)time;
                int newLevel = 0;
                try {
                    newLevel = cle.getValue();
                } catch (Exception e) {}

                if (newLevel <= 0) {
                    try {
                        newLevel = cle.getTargetHeroLevel();
                    } catch (Exception e) {}
                }

                // Record the level-up event. We don't know the exact ability from the
                // combat log HERO_LEVELUP event alone. We'll correlate with ability
                // casts or record the level progression.
                playerStates[targetIdx].levelUpTimes.add(new LevelUpEvent(gameTime, newLevel));

                System.err.println("Hero level up: " + target + " -> level " + newLevel + " at game time " + gameTime);
            }
        } catch (Exception e) {
            // skip
        }
    }

    // ======== Entity Tracking ========

    @OnEntityEntered
    public void onEntityEntered(Context ctx, Entity e) {
        lastCtx = ctx;
        String dtClass = e.getDtClass().getDtName();

        if (dtClass.startsWith("CDOTA_Unit_Hero_")) {
            discoverHero(e);
        }

        if (dtClass.equals("CDOTA_NPC_Observer_Ward") || dtClass.equals("CDOTA_NPC_Observer_Ward_TrueSight")) {
            trackWardEntity(e);
        }
    }

    /**
     * Track game rules updates for match ID, game mode, winner.
     * Game time is NOT available from entities - we use combat log timestamps instead.
     */
    @OnEntityUpdated(classPattern = "CDOTAGamerulesProxy")
    public void onGameRulesUpdated(Context ctx, Entity e, FieldPath[] fps, int count) {
        lastCtx = ctx;
        updateGameRules(e);
    }

    @OnEntityUpdated(classPattern = "CDOTA_Unit_Hero_.*")
    public void onHeroUpdated(Context ctx, Entity e, FieldPath[] fps, int count) {
        lastCtx = ctx;
        discoverHero(e);
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

    private void updateGameRules(Entity e) {
        try {
            // Match ID
            Long mid = tryGetLong(e, "m_pGameRules.m_unMatchID64");
            if (mid != null && mid > 0) {
                matchId = mid;
            }

            // Game mode
            Integer mode = tryGetInt(e, "m_pGameRules.m_iGameMode");
            if (mode != null) {
                gameMode = mode;
            }

            // Winner: only set when value is 2 (radiant) or 3 (dire)
            Integer winner = tryGetInt(e, "m_pGameRules.m_nGameWinner");
            if (winner != null) {
                if (winner == 2) radiantWin = true;
                else if (winner == 3) radiantWin = false;
            }
        } catch (Exception ex) {}
    }

    /**
     * Discover a hero entity: extract name and player ID.
     * Entity player IDs: 0,2,4,6,8 (radiant), 10,12,14,16,18 (dire)
     */
    private void discoverHero(Entity e) {
        try {
            Integer entityPlayerId = tryGetInt(e, "m_iPlayerID");
            if (entityPlayerId == null || entityPlayerId < 0) return;

            int playerIdx = entityPlayerIdToIndex(entityPlayerId);
            if (playerIdx < 0 || playerIdx >= 10) return;

            PlayerState ps = playerStates[playerIdx];

            // Convert entity class name to combat-log-compatible name
            String dtName = e.getDtClass().getDtName();
            String entityHeroName = entityClassToHeroName(dtName);

            if (entityHeroName != null && !entityHeroName.isEmpty()) {
                if (ps.heroName == null || ps.heroName.isEmpty()) {
                    ps.heroName = entityHeroName;
                    heroNameMap.put(entityHeroName, playerIdx);
                    System.err.println("Discovered hero: " + entityHeroName +
                        " entityPID=" + entityPlayerId + " -> idx=" + playerIdx);
                }
            }

            // Store the hero entity handle for final item reading
            ps.heroEntity = e;

            // Update level
            Integer lvl = tryGetInt(e, "m_iCurrentLevel");
            if (lvl != null && lvl > ps.level) ps.level = lvl;

            // Update position using OpenDota-compatible coordinate mapping
            // Source 2 cellX/Y -> game coords: cell * 62.5 - 8000
            // This maps cell 64 -> -4000, cell 190 -> 3875 (matching frontend gameToMap)
            try {
                Object cellX = getProperty(e, "CBodyComponent.m_cellX");
                Object cellY = getProperty(e, "CBodyComponent.m_cellY");
                if (cellX != null && cellY != null) {
                    float x = ((Number) cellX).floatValue() * 62.5f - 8000.0f;
                    float y = ((Number) cellY).floatValue() * 62.5f - 8000.0f;
                    ps.lastX = x;
                    ps.lastY = y;

                    // Record position history (throttled to ~2 second intervals)
                    if (lastCombatLogTime - ps.lastPositionRecordTime >= 2.0f) {
                        ps.positionHistory.add(new float[]{lastCombatLogTime, x, y});
                        ps.lastPositionRecordTime = lastCombatLogTime;
                    }
                }
            } catch (Exception ex) {}

        } catch (Exception ex) {}
    }

    private void updatePlayerData(Entity e, int offset) {
        try {
            for (int i = 0; i < 5; i++) {
                String prefix = String.format("m_vecDataTeam.%04d.", i);

                Integer gold = tryGetInt(e, prefix + "m_iTotalEarnedGold");
                if (gold != null) playerStates[offset + i].totalGold = gold;

                Integer xp = tryGetInt(e, prefix + "m_iTotalEarnedXP");
                if (xp != null) playerStates[offset + i].totalXp = xp;

                // Last hits and denies are on team data entity
                Integer lh = tryGetInt(e, prefix + "m_iLastHitCount");
                if (lh != null) playerStates[offset + i].lastHits = lh;

                Integer dn = tryGetInt(e, prefix + "m_iDenyCount");
                if (dn != null) playerStates[offset + i].denies = dn;
            }
        } catch (Exception ex) {}
    }

    private void trackWardEntity(Entity e) {
        try {
            String dtName = e.getDtClass().getDtName();
            boolean isObs = dtName.equals("CDOTA_NPC_Observer_Ward");

            Object cellX = getProperty(e, "CBodyComponent.m_cellX");
            Object cellY = getProperty(e, "CBodyComponent.m_cellY");

            if (cellX != null && cellY != null) {
                // OpenDota-compatible coordinate mapping: cell * 62.5 - 8000
                // Maps cell 64 -> -4000, cell 190 -> 3875
                float rawCX = ((Number) cellX).floatValue();
                float rawCY = ((Number) cellY).floatValue();

                float x = rawCX * 62.5f - 8000.0f;
                float y = rawCY * 62.5f - 8000.0f;

                // Dedup key using raw cell values to avoid floating point issues
                String key = String.format("%.0f_%.0f_%s", rawCX, rawCY, isObs ? "obs" : "sen");
                if (seenWards.contains(key)) return;
                seenWards.add(key);

                // Fix #3: Ward timing - store raw combat log time now,
                // convert to game time in buildOutput() when gameStartTime is known
                float wardRawTime = lastCombatLogTime;

                // Fix #1: Ward ownership - try entity properties first, then correlate with purchases
                int ownerPlayerSlot = -1;
                int ownerPlayerIdx = -1;

                // Approach 1: Try to read team from entity's m_iTeamNum
                Integer teamNum = tryGetInt(e, "m_iTeamNum");
                if (teamNum != null) {
                    // teamNum 2 = Radiant, teamNum 3 = Dire
                    if (teamNum == 2) {
                        // Radiant team - find closest ward purchase from radiant players
                        ownerPlayerIdx = findWardPurchaser(0, 4, isObs ? "observer" : "sentry");
                        if (ownerPlayerIdx < 0) {
                            // Fallback: assign to team representative (slot 0 for radiant)
                            ownerPlayerIdx = 0;
                        }
                    } else if (teamNum == 3) {
                        // Dire team
                        ownerPlayerIdx = findWardPurchaser(5, 9, isObs ? "observer" : "sentry");
                        if (ownerPlayerIdx < 0) {
                            ownerPlayerIdx = 5;
                        }
                    }
                }

                // Approach 2: If no team from entity, try purchase correlation without team filter
                if (ownerPlayerIdx < 0) {
                    ownerPlayerIdx = findWardPurchaser(0, 9, isObs ? "observer" : "sentry");
                }

                if (ownerPlayerIdx >= 0) {
                    ownerPlayerSlot = playerStates[ownerPlayerIdx].playerSlot;
                }

                Map<String, Object> ward = new LinkedHashMap<>();
                ward.put("type", isObs ? "observer" : "sentry");
                ward.put("x", x);
                ward.put("y", y);
                ward.put("raw_time", wardRawTime); // Store raw combat log time, convert later
                ward.put("time", 0); // Will be computed in buildOutput
                ward.put("player_slot", ownerPlayerSlot);

                // Log first few wards for debugging
                if (wardEvents.size() < 5) {
                    System.err.println("Ward: type=" + (isObs ? "obs" : "sen") +
                        " cellX=" + rawCX + " cellY=" + rawCY +
                        " x=" + x + " y=" + y +
                        " rawTime=" + wardRawTime + " team=" + teamNum +
                        " owner_idx=" + ownerPlayerIdx + " slot=" + ownerPlayerSlot);
                }

                wardEvents.add(ward);
            }
        } catch (Exception ex) {
            System.err.println("Ward tracking error: " + ex.getMessage());
        }
    }

    /**
     * Find the most recent ward purchaser in the given player index range
     * for the given ward type. Consumes the purchase entry so it's not reused.
     */
    private int findWardPurchaser(int minIdx, int maxIdx, String wardType) {
        // Look backwards through recent purchases to find the closest match
        float currentTime = lastCombatLogTime;
        int bestIdx = -1;
        int bestPurchaseIdx = -1;
        float bestTimeDiff = Float.MAX_VALUE;

        for (int i = recentWardPurchases.size() - 1; i >= 0; i--) {
            WardPurchase wp = recentWardPurchases.get(i);
            if (wp.consumed) continue;
            if (wp.playerIdx < minIdx || wp.playerIdx > maxIdx) continue;
            if (!wp.wardType.equals(wardType)) continue;

            float timeDiff = Math.abs(currentTime - wp.combatLogTime);
            // Allow up to 60 seconds between purchase and placement
            if (timeDiff < bestTimeDiff && timeDiff < 60.0f) {
                bestTimeDiff = timeDiff;
                bestIdx = wp.playerIdx;
                bestPurchaseIdx = i;
            }
        }

        // Consume the matched purchase
        if (bestPurchaseIdx >= 0) {
            recentWardPurchases.get(bestPurchaseIdx).consumed = true;
        }

        return bestIdx;
    }

    /**
     * Read team data from entities for timeline snapshots.
     */
    private void readTeamData(Context ctx) {
        try {
            Entities entities = ctx.getProcessor(Entities.class);
            Entity radiant = entities.getByDtName("CDOTA_DataRadiant");
            if (radiant != null) updatePlayerData(radiant, 0);
            Entity dire = entities.getByDtName("CDOTA_DataDire");
            if (dire != null) updatePlayerData(dire, 5);
        } catch (Exception ex) {}
    }

    private void recordMinuteSnapshot(int minute) {
        for (int i = 0; i < 10; i++) {
            goldTimeline.computeIfAbsent(i, k -> new ArrayList<>());
            xpTimeline.computeIfAbsent(i, k -> new ArrayList<>());
            lhTimeline.computeIfAbsent(i, k -> new ArrayList<>());

            // Fill any gaps with last known value
            List<Integer> gl = goldTimeline.get(i);
            List<Integer> xl = xpTimeline.get(i);
            List<Integer> ll = lhTimeline.get(i);
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
            if (ps == null || ps.heroEntity == null) continue;

            Entity e = ps.heroEntity;
            Map<String, Object> snap = new LinkedHashMap<>();
            snap.put("time", gameSecond);

            // Position (already tracked)
            snap.put("x", (double) ps.lastX);
            snap.put("y", (double) ps.lastY);

            // HP/Mana from entity
            Integer hp = tryGetInt(e, "m_iHealth");
            Integer maxHp = tryGetInt(e, "m_iMaxHealth");
            Float mana = tryGetFloat(e, "m_flMana");
            Float maxMana = tryGetFloat(e, "m_flMaxMana");

            snap.put("hp", hp != null ? hp : 0);
            snap.put("max_hp", maxHp != null ? maxHp : 0);
            snap.put("mana", mana != null ? Math.round(mana) : 0);
            snap.put("max_mana", maxMana != null ? Math.round(maxMana) : 0);

            // Level and farm
            snap.put("level", ps.level);
            snap.put("last_hits", ps.lastHits);
            snap.put("denies", ps.denies);
            snap.put("net_worth", ps.totalGold);

            ps.laningSnapshots.add(snap);
        }
        lastLaneSnapshotSecond = gameSecond;
    }

    // ======== Teamfight Detection ========

    private List<Map<String, Object>> detectTeamfights() {
        List<Map<String, Object>> teamfights = new ArrayList<>();
        if (killLog.isEmpty()) return teamfights;

        List<KillEvent> sorted = killLog.stream()
                .sorted(Comparator.comparingDouble(k -> k.time))
                .collect(Collectors.toList());

        float windowStart = sorted.get(0).time;
        List<KillEvent> currentWindow = new ArrayList<>();
        currentWindow.add(sorted.get(0));

        for (int i = 1; i < sorted.size(); i++) {
            KillEvent k = sorted.get(i);
            if (k.time - windowStart <= 20.0f) {
                currentWindow.add(k);
            } else {
                if (currentWindow.size() >= 3) {
                    teamfights.add(buildTeamfight(currentWindow));
                }
                currentWindow.clear();
                currentWindow.add(k);
                windowStart = k.time;
            }
        }
        if (currentWindow.size() >= 3) {
            teamfights.add(buildTeamfight(currentWindow));
        }

        return teamfights;
    }

    private Map<String, Object> buildTeamfight(List<KillEvent> kills) {
        float start = kills.stream().map(k -> k.time).min(Float::compare).orElse(0f);
        float end = kills.stream().map(k -> k.time).max(Float::compare).orElse(0f);

        // Expand window slightly to capture pre-fight positioning and post-fight cleanup
        float windowStart = start - 5.0f;
        float windowEnd = end + 5.0f;

        // Convert to game time
        int startGame = gameStarted ? (int)(start - gameStartTime) : (int)start;
        int endGame = gameStarted ? (int)(end - gameStartTime) : (int)end;
        int fightDuration = Math.max(1, endGame - startGame);

        int radiantDeaths = 0, direDeaths = 0;
        int radiantGoldDelta = 0;
        for (KillEvent k : kills) {
            if (k.targetIdx < 5) radiantDeaths++;
            else direDeaths++;
        }

        List<Map<String, Object>> tfPlayers = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            PlayerState ps = playerStates[i];
            Map<String, Object> tp = new LinkedHashMap<>();
            int slot = i < 5 ? i : 128 + (i - 5);
            tp.put("player_slot", slot);
            tp.put("hero_name", ps.heroName != null ? ps.heroName : "");

            int killsInFight = 0, deathsInFight = 0;
            float deathTime = -1;
            for (KillEvent k : kills) {
                if (k.attackerIdx == i) killsInFight++;
                if (k.targetIdx == i) {
                    deathsInFight++;
                    deathTime = k.time;
                }
            }

            // Extract position snapshots during the fight from positionHistory
            List<Map<String, Object>> positions = new ArrayList<>();
            float firstX = 0, firstY = 0;
            boolean foundFirst = false;
            for (float[] pos : ps.positionHistory) {
                if (pos[0] >= windowStart && pos[0] <= windowEnd) {
                    float timeOffset = pos[0] - start;
                    Map<String, Object> posSnap = new LinkedHashMap<>();
                    posSnap.put("x", (double) pos[1]);
                    posSnap.put("y", (double) pos[2]);
                    posSnap.put("time_offset", Math.round(timeOffset * 10.0) / 10.0);
                    positions.add(posSnap);
                    if (!foundFirst) {
                        firstX = pos[1];
                        firstY = pos[2];
                        foundFirst = true;
                    }
                }
            }
            // If no positions from history, use lastX/lastY as fallback
            if (!foundFirst) {
                firstX = ps.lastX;
                firstY = ps.lastY;
                Map<String, Object> posSnap = new LinkedHashMap<>();
                posSnap.put("x", (double) firstX);
                posSnap.put("y", (double) firstY);
                posSnap.put("time_offset", 0.0);
                positions.add(posSnap);
            }

            // Extract ability casts during fight window
            List<Map<String, Object>> abilityCasts = new ArrayList<>();
            for (Map<String, Object> cast : ps.abilityCasts) {
                int castTime = (int) cast.get("time");
                if (castTime >= startGame - 3 && castTime <= endGame + 5) {
                    float castOffset = castTime - startGame;
                    // Find position at cast time
                    float castX = firstX, castY = firstY;
                    float castRawTime = gameStarted ? castTime + gameStartTime : castTime;
                    float[] castPos = findPositionAtTime(ps, castRawTime);
                    if (castPos != null) {
                        castX = castPos[0];
                        castY = castPos[1];
                    }
                    Map<String, Object> ac = new LinkedHashMap<>();
                    ac.put("name", cast.get("name"));
                    ac.put("key", cast.get("name"));
                    ac.put("time_offset", Math.round(castOffset * 10.0) / 10.0);
                    ac.put("x", (double) castX);
                    ac.put("y", (double) castY);
                    abilityCasts.add(ac);
                }
            }

            // Sum damage dealt/taken/healing during fight window
            int damageDealt = 0, damageTaken = 0, healing = 0;
            for (float[] dmg : ps.heroDamageDealtLog) {
                if (dmg[0] >= windowStart && dmg[0] <= windowEnd) {
                    damageDealt += (int) dmg[1];
                }
            }
            for (float[] dmg : ps.heroDamageTakenLog) {
                if (dmg[0] >= windowStart && dmg[0] <= windowEnd) {
                    damageTaken += (int) dmg[1];
                }
            }
            for (float[] heal : ps.heroHealLog) {
                if (heal[0] >= windowStart && heal[0] <= windowEnd) {
                    healing += (int) heal[1];
                }
            }

            // Death time offset relative to fight start
            Double deathTimeOffset = null;
            if (deathsInFight > 0 && deathTime >= 0) {
                float dtGameTime = gameStarted ? deathTime - gameStartTime : deathTime;
                deathTimeOffset = Math.round((dtGameTime - startGame) * 10.0) / 10.0;
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
            if (deathTimeOffset != null) {
                tp.put("death_time_offset", deathTimeOffset);
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

    /**
     * Find the closest position for a player at a given raw combat log time.
     * Returns [x, y] or null if no position data available.
     */
    private float[] findPositionAtTime(PlayerState ps, float rawTime) {
        float[] best = null;
        float bestDiff = Float.MAX_VALUE;
        for (float[] pos : ps.positionHistory) {
            float diff = Math.abs(pos[0] - rawTime);
            if (diff < bestDiff) {
                bestDiff = diff;
                best = pos;
            }
            // positionHistory is chronological, stop if we've passed the target time
            if (pos[0] > rawTime + 5.0f) break;
        }
        if (best != null) {
            return new float[]{best[1], best[2]};
        }
        return null;
    }

    // ======== Final Items (from purchase log) ========

    /**
     * Compute final items for each player from their purchase log.
     * Takes the last 6 non-consumable items purchased, representing the
     * approximate final inventory. Items are returned newest-first.
     */
    private List<String> computeFinalItems(PlayerState ps) {
        List<String> finalItems = new ArrayList<>();
        // Walk purchase log backwards, collecting non-consumable items
        for (int i = ps.purchaseLog.size() - 1; i >= 0 && finalItems.size() < 6; i--) {
            String item = (String) ps.purchaseLog.get(i).get("key");
            if (item != null && !item.isEmpty() && !CONSUMABLE_ITEMS.contains(item)
                && !item.startsWith("recipe_")) {
                // Avoid duplicates (e.g., buying the same item twice counts, but
                // we want unique final items for display)
                finalItems.add(item);
            }
        }
        // Reverse so they're in chronological order
        Collections.reverse(finalItems);
        return finalItems;
    }

    // ======== Ability Upgrades (from level-up events + ability casts) ========

    /**
     * Reconstruct ability upgrades from level-up times and ability cast log.
     * For each level up, we find the first ability cast after the level-up that
     * hasn't been accounted for at a previous level. This gives us the skill build.
     *
     * If we can't correlate (no nearby cast), we still record the level-up time.
     */
    private List<Map<String, Object>> computeAbilityUpgrades(PlayerState ps) {
        List<Map<String, Object>> upgrades = new ArrayList<>();

        // If no level-up events recorded, return empty
        if (ps.levelUpTimes.isEmpty()) {
            return upgrades;
        }

        // Track which ability casts we've already matched
        int castSearchStart = 0;

        for (LevelUpEvent lu : ps.levelUpTimes) {
            Map<String, Object> upgrade = new LinkedHashMap<>();
            upgrade.put("time", lu.gameTime);
            upgrade.put("level", lu.level);

            // Try to find the ability that was skilled at this level
            // Look for the first ability cast within a short window after this level-up
            String abilityName = "";
            for (int c = castSearchStart; c < ps.abilityCasts.size(); c++) {
                Map<String, Object> cast = ps.abilityCasts.get(c);
                int castTime = (int) cast.get("time");
                // The ability cast should be near or after the level up time
                if (castTime >= lu.gameTime - 2 && castTime <= lu.gameTime + 30) {
                    abilityName = (String) cast.get("name");
                    castSearchStart = c + 1;
                    break;
                }
                if (castTime > lu.gameTime + 30) {
                    break; // No more relevant casts
                }
            }

            upgrade.put("name", abilityName);
            upgrades.add(upgrade);
        }

        return upgrades;
    }

    // ======== Draft Detection (OnTickStart) ========

    private boolean draftCollected = false;

    @OnTickStart
    public void onTickStart(Context ctx, boolean synthetic) {
        if (synthetic) return;
        if (draftCollected) return; // Only collect draft once
        try {
            Entity grp = ctx.getProcessor(Entities.class).getByDtName("CDOTAGamerulesProxy");
            if (grp == null) return;

            Integer gameState = tryGetInt(grp, "m_pGameRules.m_nGameState");
            // Wait until strategy phase (state 3) or later — draft data is fully populated by then
            if (gameState == null || gameState < 3) return;

            draftCollected = true;

            // Read bans from m_vecNewBannedHeroes
            for (int i = 0; i < 24; i++) {
                String prop = "m_pGameRules.m_vecNewBannedHeroes." + String.format("%04d", i);
                Integer heroId = tryGetInt(grp, prop);
                if (heroId != null && heroId > 0) {
                    // In All Pick ranked, bans are not team-specific, alternate team assignment
                    int team = (i % 2 == 0) ? 0 : 1;
                    Map<String, Object> entry = new LinkedHashMap<>();
                    entry.put("is_pick", false);
                    entry.put("hero_id", heroId);
                    entry.put("team", team);
                    entry.put("order", draftOrder++);
                    draftEvents.add(entry);
                }
            }

            // Read picks: each player's hero entity should be known by now
            // Use playerStates which have hero info from entity tracking
            // Also try m_SelectedHeroes for CM-style drafts
            for (int i = 0; i < 24; i++) {
                String prop = "m_pGameRules.m_SelectedHeroes." + String.format("%04d", i);
                Integer heroId = tryGetInt(grp, prop);
                if (heroId != null && heroId > 0) {
                    int team = (i < 5) ? 0 : 1; // slots 0-4 radiant, 5-9 dire (for 10 player slots)
                    Map<String, Object> entry = new LinkedHashMap<>();
                    entry.put("is_pick", true);
                    entry.put("hero_id", heroId);
                    entry.put("team", team);
                    entry.put("order", draftOrder++);
                    draftEvents.add(entry);
                }
            }

            System.err.println("Draft bans collected: " + draftEvents.size() + " entries");
        } catch (Exception e) {
            System.err.println("Draft collection error: " + e.getMessage());
        }
    }

    // ======== Chat Message Handlers ========

    @OnMessage(CDOTAUserMsg_ChatMessage.class)
    public void onChatMessage(Context ctx, CDOTAUserMsg_ChatMessage message) {
        try {
            int playerId = message.getSourcePlayerId();
            String text = message.getMessageText();
            int channelType = message.getChannelType();

            // channelType 11 = all chat, 12 = team chat
            String type = (channelType == 11) ? "chat" : "chatwheel";

            int playerSlot = -1;
            // sourcePlayerId is a direct 0-9 player index
            int playerIdx = playerId;
            if (playerIdx >= 0 && playerIdx < 10) {
                playerSlot = playerStates[playerIdx].playerSlot;
            }

            int gameTime = 0;
            if (gameStarted && lastCombatLogTime > gameStartTime) {
                gameTime = (int)(lastCombatLogTime - gameStartTime);
            }

            Map<String, Object> chat = new LinkedHashMap<>();
            chat.put("time", gameTime);
            chat.put("type", type);
            chat.put("player_slot", playerSlot);
            chat.put("key", text);
            chatMessages.add(chat);
        } catch (Exception e) {
            // skip
        }
    }

    @OnMessage(CDOTAUserMsg_ChatWheel.class)
    public void onChatWheel(Context ctx, CDOTAUserMsg_ChatWheel message) {
        try {
            int playerId = message.getPlayerId();
            int chatMessageId = message.getChatMessageId();

            // playerId from ChatWheel is a direct 0-9 player index (NOT entity player ID)
            int playerIdx = playerId;
            int playerSlot = -1;
            if (playerIdx >= 0 && playerIdx < 10) {
                playerSlot = playerStates[playerIdx].playerSlot;
            }

            int gameTime = 0;
            if (gameStarted && lastCombatLogTime > gameStartTime) {
                gameTime = (int)(lastCombatLogTime - gameStartTime);
            }

            Map<String, Object> chat = new LinkedHashMap<>();
            chat.put("time", gameTime);
            chat.put("type", "chatwheel");
            chat.put("player_slot", playerSlot);
            chat.put("key", "chatwheel_" + chatMessageId);
            chatMessages.add(chat);
        } catch (Exception e) {
            // skip
        }
    }

    @OnMessage(CUserMessageSayText2.class)
    public void onSayText2(Context ctx, CUserMessageSayText2 message) {
        try {
            String playerName = message.hasParam1() ? message.getParam1() : "";
            String text = message.hasParam2() ? message.getParam2() : "";

            int gameTime = 0;
            if (gameStarted && lastCombatLogTime > gameStartTime) {
                gameTime = (int)(lastCombatLogTime - gameStartTime);
            }

            // Try to find player slot from player name
            int playerSlot = -1;
            for (int i = 0; i < 10; i++) {
                // Player names might not match perfectly, but it's our best bet
                if (playerStates[i].heroName != null && !playerStates[i].heroName.isEmpty()) {
                    // SayText2 doesn't give us player index directly, store with -1
                }
            }

            Map<String, Object> chat = new LinkedHashMap<>();
            chat.put("time", gameTime);
            chat.put("type", "chat");
            chat.put("player_slot", playerSlot);
            chat.put("key", text);
            chat.put("player_name", playerName);
            chatMessages.add(chat);
        } catch (Exception e) {
            // skip
        }
    }

    // ======== Output ========

    private Map<String, Object> buildOutput() {
        Map<String, Object> result = new LinkedHashMap<>();

        // Compute duration
        if (gameStarted && gameEndTime > gameStartTime) {
            duration = (int) (gameEndTime - gameStartTime);
        } else if (gameStarted && lastCombatLogTime > gameStartTime) {
            duration = (int) (lastCombatLogTime - gameStartTime);
        } else if (lastCombatLogTime > 0) {
            duration = (int) lastCombatLogTime;
        }

        result.put("match_id", matchId);
        result.put("duration", duration);
        result.put("radiant_win", radiantWin);
        result.put("game_mode", gameMode);
        result.put("start_time", startTime);
        result.put("radiant_score", Arrays.stream(playerStates).limit(5).mapToInt(p -> p.kills).sum());
        result.put("dire_score", Arrays.stream(playerStates, 5, 10).mapToInt(p -> p.kills).sum());

        // Players
        List<Map<String, Object>> playersOut = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            PlayerState ps = playerStates[i];
            Map<String, Object> p = new LinkedHashMap<>();

            int slot = i < 5 ? i : 128 + (i - 5);
            float dur = duration > 0 ? duration / 60.0f : 1;

            p.put("player_slot", slot);
            p.put("hero_id", 0);
            p.put("hero_name", ps.heroName != null && !ps.heroName.isEmpty() ? ps.heroName : "unknown_" + i);
            p.put("player_name", "Player " + (i + 1));
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
            p.put("net_worth", ps.totalGold);

            // Convert purchase times from raw combat log to game-relative time
            for (Map<String, Object> purchase : ps.purchaseLog) {
                Object rawTimeObj = purchase.get("raw_time");
                if (rawTimeObj != null) {
                    float rawTime = ((Number) rawTimeObj).floatValue();
                    int purchaseGameTime;
                    if (gameStartTime > 0) {
                        purchaseGameTime = (int)(rawTime - gameStartTime);
                    } else {
                        purchaseGameTime = (int)rawTime;
                    }
                    purchase.put("time", purchaseGameTime);
                    purchase.remove("raw_time");
                }
            }
            p.put("purchase_log", ps.purchaseLog);
            p.put("ability_casts", ps.abilityCasts);
            p.put("wards_placed", new ArrayList<>());

            // Starting items from entity inventory snapshot at game start
            p.put("starting_items", ps.startingItems);

            // Fix #5: Final items from purchase log
            p.put("final_items", computeFinalItems(ps));
            p.put("backpack", new ArrayList<>());

            // Fix #4: Ability upgrades - prefer direct tracking from ability casts with levels,
            // fall back to level-up event correlation
            if (!ps.abilityUpgrades.isEmpty()) {
                p.put("ability_upgrades", ps.abilityUpgrades);
            } else {
                p.put("ability_upgrades", computeAbilityUpgrades(ps));
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
            p.put("lane", 0);
            p.put("lane_role", 0);

            p.put("laning_snapshots", ps.laningSnapshots);
            p.put("laning_damage_events", ps.laningDamageEvents);
            p.put("laning_regen_events", ps.laningRegenEvents);
            p.put("laning_creep_kills", ps.laningCreepKills);
            p.put("laning_denies", ps.laningDenies);

            playersOut.add(p);
        }
        result.put("players", playersOut);

        // Fix ward times: convert raw combat log timestamps to game time
        for (Map<String, Object> ward : wardEvents) {
            Object rawTimeObj = ward.get("raw_time");
            if (rawTimeObj != null) {
                float rawTime = ((Number) rawTimeObj).floatValue();
                int gameTime;
                if (gameStartTime > 0) {
                    gameTime = (int)(rawTime - gameStartTime);
                } else {
                    gameTime = (int)rawTime;
                }
                ward.put("time", gameTime);
                ward.remove("raw_time"); // Clean up internal field
            }
        }
        result.put("wards", wardEvents);
        // Fix #6: Smoke events
        result.put("smokes", smokeEvents);
        result.put("teamfights", detectTeamfights());
        result.put("objectives", objectiveEvents);

        // Gold/XP advantage
        List<Integer> goldAdv = new ArrayList<>();
        List<Integer> xpAdv = new ArrayList<>();
        int minLen = Integer.MAX_VALUE;
        for (int i = 0; i < 10; i++) {
            int sz = goldTimeline.getOrDefault(i, Collections.emptyList()).size();
            if (sz > 0) minLen = Math.min(minLen, sz);
        }
        if (minLen < Integer.MAX_VALUE && minLen > 0) {
            for (int m = 0; m < minLen; m++) {
                int rGold = 0, dGold = 0, rXp = 0, dXp = 0;
                for (int i = 0; i < 5; i++) {
                    List<Integer> gl = goldTimeline.getOrDefault(i, Collections.emptyList());
                    List<Integer> xl = xpTimeline.getOrDefault(i, Collections.emptyList());
                    if (gl.size() > m) rGold += gl.get(m);
                    if (xl.size() > m) rXp += xl.get(m);
                }
                for (int i = 5; i < 10; i++) {
                    List<Integer> gl = goldTimeline.getOrDefault(i, Collections.emptyList());
                    List<Integer> xl = xpTimeline.getOrDefault(i, Collections.emptyList());
                    if (gl.size() > m) dGold += gl.get(m);
                    if (xl.size() > m) dXp += xl.get(m);
                }
                goldAdv.add(rGold - dGold);
                xpAdv.add(rXp - dXp);
            }
        }
        result.put("radiant_gold_adv", goldAdv);
        result.put("radiant_xp_adv", xpAdv);

        result.put("chat", chatMessages);

        // Add picks to draft data from player hero entities (hero names are finalized by now)
        boolean hasPicks = draftEvents.stream().anyMatch(e -> (boolean) e.get("is_pick"));
        if (!hasPicks) {
            for (int i = 0; i < 10; i++) {
                PlayerState ps = playerStates[i];
                if (ps.heroName != null && !ps.heroName.isEmpty()) {
                    int team = (i < 5) ? 0 : 1;
                    Map<String, Object> entry = new LinkedHashMap<>();
                    entry.put("is_pick", true);
                    entry.put("hero_id", 0);
                    entry.put("hero_name", ps.heroName);
                    entry.put("team", team);
                    entry.put("order", draftOrder++);
                    draftEvents.add(entry);
                }
            }
        }
        System.err.println("Draft total: " + draftEvents.size() + " entries ("
            + draftEvents.stream().filter(e -> !(boolean) e.get("is_pick")).count() + " bans, "
            + draftEvents.stream().filter(e -> (boolean) e.get("is_pick")).count() + " picks)");
        result.put("picks_bans", draftEvents);
        result.put("tower_status_radiant", 0);
        result.put("tower_status_dire", 0);
        result.put("barracks_status_radiant", 0);
        result.put("barracks_status_dire", 0);
        int firstBlood = 0;
        if (!killLog.isEmpty()) {
            float fbRaw = killLog.get(0).time - (gameStarted ? gameStartTime : 0);
            firstBlood = Math.max(0, (int) fbRaw);
        }
        result.put("first_blood_time", firstBlood);
        result.put("region", 0);
        result.put("patch", "");

        return result;
    }

    // ======== Name Normalization ========

    /**
     * Convert entity class name to combat-log-compatible hero name.
     * "CDOTA_Unit_Hero_DrowRanger" -> "drow_ranger"
     * "CDOTA_Unit_Hero_Hoodwink" -> "hoodwink"
     * "CDOTA_Unit_Hero_DragonKnight" -> "dragon_knight"
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
        return sb.toString().toLowerCase();
    }

    /**
     * Clean a combat log hero name: strip "npc_dota_hero_" prefix.
     */
    static String cleanHeroName(String name) {
        if (name == null) return "";
        if (name.startsWith("npc_dota_hero_")) {
            return name.substring("npc_dota_hero_".length());
        }
        return name;
    }

    /**
     * Clean a generic combat log name: strip common prefixes.
     */
    static String cleanName(String name) {
        if (name == null) return "";
        return name.replace("npc_dota_hero_", "")
                .replace("npc_dota_", "")
                .replace("item_", "");
    }

    // ======== Utility ========

    private static Object getProperty(Entity e, String path) {
        try {
            return e.getProperty(path);
        } catch (Exception ex) {
            return null;
        }
    }

    private static Integer tryGetInt(Entity e, String... paths) {
        for (String path : paths) {
            try {
                Object val = e.getProperty(path);
                if (val != null) return ((Number) val).intValue();
            } catch (Exception ex) {}
        }
        return null;
    }

    private static Float tryGetFloat(Entity e, String... paths) {
        for (String path : paths) {
            try {
                Object val = e.getProperty(path);
                if (val != null) return ((Number) val).floatValue();
            } catch (Exception ex) {}
        }
        return null;
    }

    private static Long tryGetLong(Entity e, String... paths) {
        for (String path : paths) {
            try {
                Object val = e.getProperty(path);
                if (val != null) return ((Number) val).longValue();
            } catch (Exception ex) {}
        }
        return null;
    }

    /**
     * Map a hero name (from combat log) to player index (0-9).
     */
    private int heroNameToPlayerIdx(String name) {
        if (name == null || name.isEmpty()) return -1;

        Integer cached = heroNameMap.get(name);
        if (cached != null) return cached;

        String lower = name.toLowerCase();

        // Exact match
        for (int i = 0; i < 10; i++) {
            if (playerStates[i].heroName != null && !playerStates[i].heroName.isEmpty()) {
                if (playerStates[i].heroName.equals(lower)) {
                    heroNameMap.put(name, i);
                    return i;
                }
            }
        }

        // Match ignoring underscores (handles edge cases)
        String lowerNoUnderscore = lower.replace("_", "");
        for (int i = 0; i < 10; i++) {
            if (playerStates[i].heroName != null && !playerStates[i].heroName.isEmpty()) {
                if (playerStates[i].heroName.replace("_", "").equals(lowerNoUnderscore)) {
                    heroNameMap.put(name, i);
                    return i;
                }
            }
        }

        return -1;
    }

    // ======== Inner Classes ========

    private static class PlayerState {
        int playerIdx;
        int playerSlot;
        String heroName = "";
        int kills = 0, deaths = 0, assists = 0;
        int level = 1;
        int lastHits = 0, denies = 0;
        int totalGold = 0, totalXp = 0;
        int heroDamage = 0, towerDamage = 0, heroHealing = 0;
        int buybackCount = 0, roshansKilled = 0, courierKills = 0;
        float lastX = 0, lastY = 0;
        Entity heroEntity = null;  // Reference to hero entity for final item reading

        // Position history for smoke paths and teamfight positions: [rawCombatLogTime, gameX, gameY]
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

        List<Map<String, Object>> purchaseLog = new ArrayList<>();
        List<Map<String, Object>> abilityCasts = new ArrayList<>();
        List<LevelUpEvent> levelUpTimes = new ArrayList<>();
        List<Map<String, Object>> abilityUpgrades = new ArrayList<>();
        Map<String, Integer> abilityLevels = new LinkedHashMap<>();  // ability name -> current known level
        List<Map<String, Object>> laningSnapshots = new ArrayList<>();
        List<Map<String, Object>> laningDamageEvents = new ArrayList<>(); // hero-vs-hero damage during laning
        List<Map<String, Object>> laningRegenEvents = new ArrayList<>(); // regen item/ability uses during laning
        List<Map<String, Object>> laningCreepKills = new ArrayList<>(); // last hits with timestamps
        List<Map<String, Object>> laningDenies = new ArrayList<>();     // denies with timestamps
        List<String> startingItems = new ArrayList<>();  // inventory snapshot at game start

        PlayerState(int idx) {
            this.playerIdx = idx;
            this.playerSlot = idx < 5 ? idx : 128 + (idx - 5);
        }
    }

    private static class KillEvent {
        float time;
        int attackerIdx;
        int targetIdx;

        KillEvent(float time, int attackerIdx, int targetIdx) {
            this.time = time;
            this.attackerIdx = attackerIdx;
            this.targetIdx = targetIdx;
        }
    }

    private static class WardPurchase {
        int playerIdx;
        float combatLogTime;
        String wardType; // "observer" or "sentry"
        boolean consumed = false;

        WardPurchase(int playerIdx, float combatLogTime, String wardType) {
            this.playerIdx = playerIdx;
            this.combatLogTime = combatLogTime;
            this.wardType = wardType;
        }
    }

    private static class LevelUpEvent {
        int gameTime;
        int level;

        LevelUpEvent(int gameTime, int level) {
            this.gameTime = gameTime;
            this.level = level;
        }
    }

    // ======== Main ========

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: java -jar clarity-analyzer.jar <replay.dem>");
            System.exit(1);
        }

        String replayPath = args[0];
        File replayFile = new File(replayPath);
        if (!replayFile.exists()) {
            System.err.println("File not found: " + replayPath);
            System.exit(1);
        }

        ReplayAnalyzer analyzer = new ReplayAnalyzer();

        try {
            new SimpleRunner(new MappedFileSource(replayPath)).runWith(analyzer);
        } catch (Exception e) {
            System.err.println("Error parsing replay: " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(1);
        }

        // Do a final team data read
        if (analyzer.lastCtx != null) {
            analyzer.readTeamData(analyzer.lastCtx);
        }

        Map<String, Object> output = analyzer.buildOutput();

        // Debug summary
        System.err.println("=== Parse Summary ===");
        System.err.println("match_id: " + analyzer.matchId);
        System.err.println("duration: " + analyzer.duration + "s");
        System.err.println("gameStartTime: " + analyzer.gameStartTime);
        System.err.println("gameEndTime: " + analyzer.gameEndTime);
        System.err.println("lastCombatLogTime: " + analyzer.lastCombatLogTime);
        System.err.println("radiantWin: " + analyzer.radiantWin);
        System.err.println("Heroes:");
        for (int i = 0; i < 10; i++) {
            PlayerState ps = analyzer.playerStates[i];
            System.err.println("  [" + i + "] " + ps.heroName + ": " +
                ps.kills + "/" + ps.deaths + "/" + ps.assists +
                " LH:" + ps.lastHits + " DN:" + ps.denies +
                " Gold:" + ps.totalGold + " XP:" + ps.totalXp +
                " LevelUps:" + ps.levelUpTimes.size() +
                " AbilUpgrades:" + ps.abilityUpgrades.size());
        }
        System.err.println("heroNameMap: " + analyzer.heroNameMap);
        System.err.println("wards: " + analyzer.wardEvents.size());
        int wardsWithOwner = 0;
        for (Map<String, Object> w : analyzer.wardEvents) {
            if ((int) w.get("player_slot") >= 0) wardsWithOwner++;
        }
        System.err.println("wards with owner: " + wardsWithOwner);
        System.err.println("smokes: " + analyzer.smokeEvents.size());
        System.err.println("objectives: " + analyzer.objectiveEvents.size());
        System.err.println("kills in log: " + analyzer.killLog.size());
        System.err.println("gold_adv points: " + ((List)output.get("radiant_gold_adv")).size());

        Gson gson = new GsonBuilder().create();
        System.out.println(gson.toJson(output));
    }
}
