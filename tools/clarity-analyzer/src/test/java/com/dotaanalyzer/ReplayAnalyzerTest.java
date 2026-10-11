package com.dotaanalyzer;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for the name helpers, the skill-up rule and the end-of-run resolve steps of
 * ReplayAnalyzer (run with `./gradlew test`). The names and times come from real replays
 * (matches 8762775364 and 8713267082, and the 7.41f matches 9037980041, 9037747302 and
 * 9038377636). Reading the entities themselves needs a replay: verify that part with
 * tools/compare_with_opendota.py.
 */
class ReplayAnalyzerTest {

    @Test
    void towerAndFortEntityNamesBecomeCombatLogNames() {
        assertEquals("npc_dota_goodguys_tower1_top", ReplayAnalyzer.canonicalBuildingName("dota_goodguys_tower1_top"));
        assertEquals("npc_dota_badguys_tower3_mid", ReplayAnalyzer.canonicalBuildingName("npc_dota_badguys_tower3_mid"));
        assertEquals("npc_dota_goodguys_fort", ReplayAnalyzer.canonicalBuildingName("dota_goodguys_fort"));
        assertNull(ReplayAnalyzer.canonicalBuildingName("dota_fountain"));
        assertNull(ReplayAnalyzer.canonicalBuildingName(null));
    }

    @Test
    void barracksMapEntityNamesAreCanonicalised() {
        // The map names barracks good_rax_melee_top; the combat log says goodguys_melee_rax_top.
        assertEquals("npc_dota_goodguys_melee_rax_top", ReplayAnalyzer.canonicalBuildingName("good_rax_melee_top"));
        assertEquals("npc_dota_badguys_range_rax_mid", ReplayAnalyzer.canonicalBuildingName("bad_rax_range_mid"));
        assertEquals("npc_dota_badguys_melee_rax_bot", ReplayAnalyzer.canonicalBuildingName("npc_dota_badguys_melee_rax_bot"));
    }

    @Test
    void tierFourTowersKeepTheirSideButShareOneCombatLogName() {
        assertEquals("npc_dota_goodguys_tower4_top", ReplayAnalyzer.canonicalBuildingName("dota_goodguys_tower4_top"));
        assertEquals("npc_dota_goodguys_tower4_bot", ReplayAnalyzer.canonicalBuildingName("dota_goodguys_tower4_bot"));
        assertEquals("npc_dota_goodguys_tower4", ReplayAnalyzer.combatLogBuildingName("npc_dota_goodguys_tower4_top"));
        assertEquals("npc_dota_goodguys_tower4", ReplayAnalyzer.combatLogBuildingName("npc_dota_goodguys_tower4_bot"));
        assertEquals("npc_dota_badguys_tower3_top", ReplayAnalyzer.combatLogBuildingName("npc_dota_badguys_tower3_top"));
        assertEquals("", ReplayAnalyzer.combatLogBuildingName(null));
    }

    @Test
    void itemClassFallbackKnowsTheKeysThatAreNotSnakeCase() {
        assertEquals("travel_boots", ReplayAnalyzer.entityClassToItemName("CDOTA_Item_BootsOfTravel"));
        assertEquals("dust", ReplayAnalyzer.entityClassToItemName("CDOTA_Item_DustofAppearance"));
        assertEquals("invis_sword", ReplayAnalyzer.entityClassToItemName("CDOTA_Item_InvisibilityEdge"));
        assertEquals("bottle", ReplayAnalyzer.entityClassToItemName("CDOTA_Item_EmptyBottle"));
        assertEquals("recipe_travel_boots", ReplayAnalyzer.entityClassToItemName("CDOTA_Item_Recipe_BootsOfTravel"));
        assertEquals("magic_stick", ReplayAnalyzer.entityClassToItemName("CDOTA_Item_MagicStick"));
        assertEquals("quelling_blade", ReplayAnalyzer.entityClassToItemName("CDOTA_Item_QuellingBlade"));
        // seen in the 7.41f replays
        assertEquals("basher", ReplayAnalyzer.entityClassToItemName("CDOTA_Item_CraniumBasher"));
        assertEquals("greater_crit", ReplayAnalyzer.entityClassToItemName("CDOTA_Item_GreaterCritical"));
        assertEquals("sheepstick", ReplayAnalyzer.entityClassToItemName("CDOTA_Item_SheepStick"));
        assertEquals("refresher_shard", ReplayAnalyzer.entityClassToItemName("CDOTA_Item_RefresherOrb_Shard"));
        assertEquals("recipe_orchid", ReplayAnalyzer.entityClassToItemName("CDOTA_Item_Recipe_OrchidMalevolence"));
        // the items 7.41 added convert by themselves
        assertEquals("crellas_crozier", ReplayAnalyzer.entityClassToItemName("CDOTA_Item_CrellasCrozier"));
        assertEquals("essence_distiller", ReplayAnalyzer.entityClassToItemName("CDOTA_Item_EssenceDistiller"));
        assertNull(ReplayAnalyzer.entityClassToItemName("CDOTA_Ability_Lion_Impale"));
    }

    @Test
    void clusterComesFromTheDemoHeaderServerName() {
        Matcher m = ReplayAnalyzer.SERVER_CLUSTER_RE.matcher("Valve Dota 2 Singapore Server (srcds1006-sgp2.151.19)");
        assertTrue(m.find());
        assertEquals("151", m.group(1));
        m = ReplayAnalyzer.SERVER_CLUSTER_RE.matcher("Valve Dota 2 Singapore Server (srcds2008-sgp2.152.190)");
        assertTrue(m.find());
        assertEquals("152", m.group(1));
        assertFalse(ReplayAnalyzer.SERVER_CLUSTER_RE.matcher("Dota 2 local server").find());
        assertEquals(5, ReplayAnalyzer.CLUSTER_REGION.get(151));
        assertEquals(3, ReplayAnalyzer.CLUSTER_REGION.get(138));
        assertNull(ReplayAnalyzer.CLUSTER_REGION.get(999));
    }

    @Test
    void gameBuildComesFromTheGameDirectory() {
        Matcher m = ReplayAnalyzer.GAME_BUILD_RE.matcher("/opt/srcds/dota/dota_v6762/dota");
        assertTrue(m.find());
        assertEquals("6762", m.group(1));
    }

    @Test
    void talentsAreSpecialBonusAbilitiesExceptTheAttributeBonus() {
        assertTrue(ReplayAnalyzer.isTalent("special_bonus_unique_drow_ranger_2"));
        assertTrue(ReplayAnalyzer.isTalent("special_bonus_attack_range_75"));
        assertFalse(ReplayAnalyzer.isTalent("special_bonus_attributes"));
        assertFalse(ReplayAnalyzer.isTalent("drow_ranger_marksmanship"));
        assertFalse(ReplayAnalyzer.isTalent(null));
    }

    @Test
    void combatLogNamesAreCleaned() {
        assertEquals("skeleton_king", ReplayAnalyzer.cleanHeroName("npc_dota_hero_skeleton_king"));
        assertEquals("creep_badguys_melee", ReplayAnalyzer.cleanName("npc_dota_creep_badguys_melee"));
        assertEquals("ward_observer", ReplayAnalyzer.cleanName("item_ward_observer"));
        assertEquals("drow_ranger", ReplayAnalyzer.entityClassToHeroName("CDOTA_Unit_Hero_DrowRanger"));
        assertEquals("queenofpain", ReplayAnalyzer.normalizeHeroKey("npc_dota_hero_queen_of_pain"));
    }

    @Test
    void onlyRoshanHimselfIsARoshanKill() {
        assertTrue(ReplayAnalyzer.isRoshan("npc_dota_roshan"));
        // Dropped by every second and later Roshan; its expiry is a combat-log death (9037980041).
        assertFalse(ReplayAnalyzer.isRoshan("npc_dota_unit_roshans_banner"));
        assertFalse(ReplayAnalyzer.isRoshan("npc_dota_roshan_halloween"));
        assertFalse(ReplayAnalyzer.isRoshan("npc_dota_mutation_pocket_roshan"));
        assertFalse(ReplayAnalyzer.isRoshan("npc_dota_miniboss"));
        assertFalse(ReplayAnalyzer.isRoshan(null));
    }

    @Test
    void frameClockIsTheRoundedServerTickWithoutPausedTicks() {
        // 30 ticks a second; x.5 rounds up, so second 2420 starts at tick 72585.
        assertEquals(2419, ReplayAnalyzer.frameClockSeconds(72584, 0, -1));
        assertEquals(2420, ReplayAnalyzer.frameClockSeconds(72585, 0, -1));
        assertEquals(2410, ReplayAnalyzer.frameClockSeconds(72585, 300, -1));
        // While paused the clock stands at the tick the pause started.
        assertEquals(2410, ReplayAnalyzer.frameClockSeconds(90000, 300, 72585));
    }

    // ---- Resolve steps (state is filled the way the combat-log / entity handlers fill it) ----

    /** Analyzer whose clock starts at raw time 0, with heroes hero0 .. hero9. */
    private static ReplayAnalyzer analyzer() {
        ReplayAnalyzer a = new ReplayAnalyzer();
        a.gameStarted = true;
        a.clockAnchored = true;
        a.gameStartTime = 0f;
        a.lastCombatLogTime = 10_000f;
        for (int i = 0; i < 10; i++) a.playerStates[i].heroName = "hero" + i;
        return a;
    }

    private static ReplayAnalyzer.KillEvent death(ReplayAnalyzer a, float time, int tick, int attackerIdx,
                                                 int targetIdx, boolean deny, String killerUnit) {
        ReplayAnalyzer.KillEvent k = new ReplayAnalyzer.KillEvent(time, tick, attackerIdx, targetIdx, deny);
        k.killerUnit = killerUnit;
        a.killLog.add(k);
        return k;
    }

    @Test
    void scoreboardKillWithoutHeroLastHitIsCreditedToThePlayer() {
        ReplayAnalyzer a = analyzer();
        // A creep lands the kill on dire hero 5; the scoreboard credits radiant player 2.
        ReplayAnalyzer.KillEvent byCreep = death(a, 100f, 3000, -1, 5, false, "npc_dota_creep_goodguys_melee");
        // Radiant hero 1 kills dire hero 6; the scoreboard also gives player 3 an assist the
        // combat-log list does not have.
        ReplayAnalyzer.KillEvent byHero = death(a, 200f, 6000, 1, 6, false, "npc_dota_hero_hero1");
        // Dire hero 8 is denied by dire hero 7: no kill, no assists.
        ReplayAnalyzer.KillEvent denied = death(a, 300f, 9000, 7, 8, true, "npc_dota_hero_hero7");
        // A neutral kills radiant hero 0 and nobody is credited.
        ReplayAnalyzer.KillEvent byNeutral = death(a, 400f, 12000, -1, 0, false, "npc_dota_neutral_kobold");
        a.scoreIncrements.add(new int[]{3000, 2, 0});
        a.scoreIncrements.add(new int[]{6000, 1, 0});
        a.scoreIncrements.add(new int[]{6000, 3, 1});

        a.resolveKillCredits();

        assertEquals(2, byCreep.killerIdx);
        assertEquals(1, byHero.killerIdx);
        assertEquals(Set.of(3), byHero.assistIdx);
        assertEquals(7, denied.killerIdx);
        assertTrue(denied.assistIdx.isEmpty());
        assertEquals(-1, byNeutral.killerIdx);
        // killed / killed_by follow the credited killer; a deny counts for neither.
        assertEquals(Map.of("hero5", 1), a.playerStates[2].killed);
        assertEquals(Map.of("hero2", 1), a.playerStates[5].killedBy);
        assertEquals(Map.of("neutral_kobold", 1), a.playerStates[0].killedBy);
        assertTrue(a.playerStates[7].killed.isEmpty());
        assertTrue(a.playerStates[8].killedBy.isEmpty());
    }

    @Test
    void deathToNeutralsScoresForNobodyAndCreditsNoAssists() {
        ReplayAnalyzer a = analyzer();
        // Match 9037747302 at 1:44: a neutral creep kills dire hero 7; the combat log still lists
        // radiant players 4 and 2, but the game credits no kill, no assists and no score.
        ReplayAnalyzer.KillEvent byNeutral = death(a, 104.3f, 3130, -1, 7, false, "npc_dota_neutral_fel_beast");
        byNeutral.attackerTeam = 4;
        byNeutral.assistRaw = new ArrayList<>(List.of(4, 2));
        // A radiant tower kills dire hero 8 with nobody near: the radiant side scores, no player does.
        ReplayAnalyzer.KillEvent byTower = death(a, 200f, 6000, -1, 8, false, "npc_dota_goodguys_tower1_mid");
        byTower.attackerTeam = ReplayAnalyzer.TEAM_RADIANT;
        // Radiant hero 1 kills dire hero 5 with the listed assist of player 2 (scoreboard agrees).
        ReplayAnalyzer.KillEvent byHero = death(a, 300f, 9000, 1, 5, false, "npc_dota_hero_hero1");
        byHero.attackerTeam = ReplayAnalyzer.TEAM_RADIANT;
        byHero.assistRaw = new ArrayList<>(List.of(2));
        // Dire hero 6 is denied by dire hero 9.
        ReplayAnalyzer.KillEvent denied = death(a, 400f, 12000, 9, 6, true, "npc_dota_hero_hero9");
        denied.attackerTeam = ReplayAnalyzer.TEAM_DIRE;
        a.scoreIncrements.add(new int[]{9000, 1, 0});
        a.scoreIncrements.add(new int[]{9000, 2, 1});

        a.resolveKillCredits();

        assertTrue(byNeutral.assistIdx.isEmpty());
        assertEquals(Set.of(2), byHero.assistIdx);
        assertEquals(0, a.playerStates[4].assists);
        assertEquals(1, a.playerStates[2].assists);
        assertFalse(ReplayAnalyzer.scoresForEnemy(byNeutral));
        assertTrue(ReplayAnalyzer.scoresForEnemy(byTower));
        assertTrue(ReplayAnalyzer.scoresForEnemy(byHero));
        assertFalse(ReplayAnalyzer.scoresForEnemy(denied));
        assertEquals(2, a.killLogScore()[0]);  // radiant: the tower kill and the hero kill
        assertEquals(0, a.killLogScore()[1]);
    }

    @Test
    void assistIncrementConfirmsAListedAssistBeforeAddingANewOne() {
        ReplayAnalyzer a = analyzer();
        // Two dire heroes die one second apart; the combat log lists player 2 only for the second.
        ReplayAnalyzer.KillEvent first = death(a, 100f, 3000, 1, 5, false, "npc_dota_hero_hero1");
        ReplayAnalyzer.KillEvent second = death(a, 101f, 3030, 1, 6, false, "npc_dota_hero_hero1");
        second.assistRaw = new ArrayList<>(List.of(2));
        a.scoreIncrements.add(new int[]{3000, 2, 1});
        a.scoreIncrements.add(new int[]{3030, 2, 1});

        a.resolveKillCredits();

        assertEquals(Set.of(2), first.assistIdx);
        assertEquals(Set.of(2), second.assistIdx);
        assertEquals(1, a.playerStates[2].assists);  // the combat-log fallback count stays as listed
    }

    private static ReplayAnalyzer.BuildingRecord building(ReplayAnalyzer a, int index, String name, Float destroyedRaw) {
        ReplayAnalyzer.BuildingRecord b = new ReplayAnalyzer.BuildingRecord();
        b.entityIndex = index;
        b.name = name;
        b.destroyedRaw = destroyedRaw;
        a.buildingsByEntityIndex.put(index, b);
        return b;
    }

    @Test
    void oneTierFourDeathNeverMarksBothTierFourTowers() {
        ReplayAnalyzer a = analyzer();
        ReplayAnalyzer.BuildingRecord top = building(a, 1, "npc_dota_goodguys_tower4_top", null);
        ReplayAnalyzer.BuildingRecord bot = building(a, 2, "npc_dota_goodguys_tower4_bot", 1890.3f);
        ReplayAnalyzer.BuildingRecord t3 = building(a, 3, "npc_dota_goodguys_tower3_mid", 1700.2f);
        // The combat log calls both tier-4 towers npc_dota_goodguys_tower4.
        a.buildingDeathRaw.put("npc_dota_goodguys_tower4", new ArrayList<>(List.of(1890f)));
        a.buildingDeathRaw.put("npc_dota_goodguys_tower3_mid", new ArrayList<>(List.of(1700f)));
        Map<String, Object> objective = new LinkedHashMap<>();
        objective.put("raw_time", 1890f);
        objective.put("type", "tower_kill");
        objective.put("key", "goodguys_tower4");
        a.objectiveEvents.add(objective);

        a.resolveBuildingDeaths();
        a.linkObjectivesToBuildings();

        assertNull(top.deathRaw);
        assertEquals(1890f, bot.deathRaw);
        assertEquals(1700f, t3.deathRaw);
        assertEquals("npc_dota_goodguys_tower4_bot", objective.get("building"));
    }

    @Test
    void combatLogBuildingTimeIsUsedAtMostOnceWithoutEntityEvidence() {
        ReplayAnalyzer a = analyzer();
        ReplayAnalyzer.BuildingRecord top = building(a, 1, "npc_dota_badguys_tower4_top", null);
        ReplayAnalyzer.BuildingRecord bot = building(a, 2, "npc_dota_badguys_tower4_bot", null);
        a.buildingDeathRaw.put("npc_dota_badguys_tower4", new ArrayList<>(List.of(2000f)));

        a.resolveBuildingDeaths();

        long destroyed = Arrays.asList(top, bot).stream().filter(b -> b.deathRaw != null).count();
        assertEquals(1, destroyed);
    }

    private static ReplayAnalyzer.WardRecord ward(ReplayAnalyzer a, int index, String type, int team, int ownerIdx,
                                                  int destroyedTick, boolean byLifeState) {
        ReplayAnalyzer.WardRecord w = new ReplayAnalyzer.WardRecord(index, type, 10f);
        w.team = team;
        w.ownerIdx = ownerIdx;
        w.destroyedRaw = destroyedTick / 30f;
        w.destroyedTick = destroyedTick;
        w.diedByLifeState = byLifeState;
        a.wardRecords.add(w);
        return w;
    }

    @Test
    void wardDeathsAreMatchedByTypeTeamTickAndOwner() {
        ReplayAnalyzer a = analyzer();
        ReplayAnalyzer.WardRecord dewarded = ward(a, 1, "sentry", ReplayAnalyzer.TEAM_RADIANT, 1, 3000, true);
        ReplayAnalyzer.WardRecord expired = ward(a, 2, "sentry", ReplayAnalyzer.TEAM_RADIANT, 3, 3000, true);
        ReplayAnalyzer.WardRecord enemySentry = ward(a, 3, "sentry", ReplayAnalyzer.TEAM_DIRE, 6, 3000, true);
        ReplayAnalyzer.WardRecord unexplained = ward(a, 4, "observer", ReplayAnalyzer.TEAM_DIRE, 7, 9000, true);
        // Without a life state the record only has the entity deletion, ~5 s after the death.
        ReplayAnalyzer.WardRecord deleted = ward(a, 5, "observer", ReplayAnalyzer.TEAM_RADIANT, 2, 6150, false);
        // Same tick, same type, same team: the owner tells the two radiant sentries apart.
        a.wardDeaths.add(new ReplayAnalyzer.WardDeath(3000, 100f, "sentry", ReplayAnalyzer.TEAM_RADIANT,
                ReplayAnalyzer.WARD_EXPIRED, ReplayAnalyzer.TEAM_RADIANT, -1, 3));
        a.wardDeaths.add(new ReplayAnalyzer.WardDeath(3000, 100f, "sentry", ReplayAnalyzer.TEAM_RADIANT,
                ReplayAnalyzer.WARD_DEWARDED, ReplayAnalyzer.TEAM_DIRE, 6, 1));
        a.wardDeaths.add(new ReplayAnalyzer.WardDeath(6000, 200f, "observer", ReplayAnalyzer.TEAM_RADIANT,
                ReplayAnalyzer.WARD_DEWARDED, ReplayAnalyzer.TEAM_DIRE, 8, 2));

        a.matchWardDeaths();

        assertTrue(dewarded.deathMatched && dewarded.dewarded && !dewarded.denied);
        assertEquals(6, dewarded.killerIdx);
        assertEquals(ReplayAnalyzer.TEAM_DIRE, dewarded.killerTeam);
        assertTrue(expired.deathMatched && !expired.dewarded && !expired.denied);
        assertEquals(-1, expired.killerIdx);
        assertEquals(ReplayAnalyzer.TEAM_RADIANT, expired.killerTeam);
        assertFalse(enemySentry.deathMatched);
        assertFalse(unexplained.deathMatched);
        assertTrue(deleted.deathMatched && deleted.dewarded);
        assertEquals(200f, deleted.destroyedRaw);  // corrected to the death itself
    }

    @Test
    void aWardDeniedByItsOwnSideIsNeitherDewardedNorExpired() {
        ReplayAnalyzer a = analyzer();
        ReplayAnalyzer.WardRecord denied = ward(a, 1, "sentry", ReplayAnalyzer.TEAM_DIRE, 6, 3000, true);
        a.wardDeaths.add(new ReplayAnalyzer.WardDeath(3000, 100f, "sentry", ReplayAnalyzer.TEAM_DIRE,
                ReplayAnalyzer.WARD_DENIED, ReplayAnalyzer.TEAM_DIRE, 7, 6));

        a.matchWardDeaths();

        assertTrue(denied.deathMatched && denied.denied && !denied.dewarded);
        assertEquals(7, denied.killerIdx);  // the ally that attacked it, not its owner
        assertEquals(ReplayAnalyzer.TEAM_DIRE, denied.killerTeam);
    }

    @Test
    void respawnIsTheFirstSampleAliveAgainAfterADeath() {
        List<float[]> samples = new ArrayList<>();
        // one sample a second: alive until 100, dead from 101, alive again at 131
        for (int t = 95; t <= 140; t++) {
            samples.add(new float[]{t + 0.02f, 0f, 0f, t <= 100 || t >= 131 ? 80f : 0f});
        }
        List<Float> none = new ArrayList<>();
        assertEquals(131.02f, ReplayAnalyzer.respawnRaw(samples, none, 100.4f));
        // Around a pause the sample clock runs ahead: samples after the death can still carry
        // the last health (replay 8762775364, death at 1705.7: 1707.3 = 11 %, 1708.0 = dead).
        samples.set(6, new float[]{101.02f, 0f, 0f, 3f});
        samples.set(7, new float[]{102.3f, 0f, 0f, 3f});
        assertEquals(131.02f, ReplayAnalyzer.respawnRaw(samples, none, 100.4f));
        // Unknown health is no sign of life, and a hero that never comes back has no respawn.
        assertNull(ReplayAnalyzer.respawnRaw(samples.subList(0, 30), none, 100.4f));
        List<float[]> unknown = new ArrayList<>(samples.subList(0, 30));
        unknown.add(new float[]{125.02f, 0f, 0f, -1f});
        assertNull(ReplayAnalyzer.respawnRaw(unknown, none, 100.4f));
    }

    @Test
    void aBuybackIsTheExactRespawnAndBelongsToTheDeathBeforeIt() {
        List<float[]> samples = new ArrayList<>();
        // dead from 101, bought back at 110.6 (alive in the sample of 111), dead again from 121
        for (int t = 95; t <= 140; t++) {
            boolean alive = t <= 100 || (t >= 111 && t <= 120);
            samples.add(new float[]{t + 0.02f, 0f, 0f, alive ? 80f : 0f});
        }
        List<Float> buybacks = Arrays.asList(110.6f);
        assertEquals(110.6f, ReplayAnalyzer.respawnRaw(samples, buybacks, 100.4f));
        assertNull(ReplayAnalyzer.respawnRaw(samples, buybacks, 120.5f));  // not the earlier buyback
        // An instant buyback: no sample ever shows the hero dead.
        List<float[]> never = new ArrayList<>();
        for (int t = 95; t <= 110; t++) never.add(new float[]{t + 0.02f, 0f, 0f, 60f});
        assertEquals(100.9f, ReplayAnalyzer.respawnRaw(never, Arrays.asList(100.9f), 100.4f));
    }

    @Test
    void theAttackerOfAWardDeathTellsDewardDenyAndExpiryApart() {
        int radiant = ReplayAnalyzer.TEAM_RADIANT, dire = ReplayAnalyzer.TEAM_DIRE;
        assertEquals(ReplayAnalyzer.WARD_DEWARDED, ReplayAnalyzer.wardDeathCause("npc_dota_hero_lion", dire, radiant));
        assertEquals(ReplayAnalyzer.WARD_DEWARDED,
                ReplayAnalyzer.wardDeathCause("npc_dota_creep_badguys_ranged", dire, radiant));
        assertEquals(ReplayAnalyzer.WARD_DENIED, ReplayAnalyzer.wardDeathCause("npc_dota_hero_snapfire", dire, dire));
        // A ward that runs out is its own attacker, on its own side: never a deny.
        assertEquals(ReplayAnalyzer.WARD_EXPIRED, ReplayAnalyzer.wardDeathCause("npc_dota_sentry_wards", dire, dire));
        assertEquals(ReplayAnalyzer.WARD_EXPIRED,
                ReplayAnalyzer.wardDeathCause("npc_dota_observer_wards", radiant, radiant));
        assertEquals(ReplayAnalyzer.WARD_OTHER, ReplayAnalyzer.wardDeathCause("npc_dota_neutral_kobold", 4, radiant));
        assertEquals(ReplayAnalyzer.WARD_OTHER, ReplayAnalyzer.wardDeathCause(null, 0, radiant));
        assertEquals(ReplayAnalyzer.WARD_OTHER, ReplayAnalyzer.wardDeathCause("npc_dota_hero_lion", dire, 0));
    }

    @SuppressWarnings("unchecked")
    @Test
    void teamfightsChainUnderFifteenSecondsAndReincarnationsKeepThemAlive() {
        ReplayAnalyzer a = analyzer();
        // Fight 1: two deaths, a reincarnation 10 s later, then a third death: one fight of 3.
        death(a, 100.2f, 3006, 1, 5, false, "npc_dota_hero_hero1");
        death(a, 104.7f, 3141, 6, 1, false, "npc_dota_hero_hero6");
        a.reincarnationRaw.add(114.0f);
        death(a, 124.5f, 3735, 2, 6, false, "npc_dota_hero_hero2");
        // Three deaths exactly 15 whole seconds apart are not a fight (OpenDota's rule).
        death(a, 500.0f, 15000, 1, 5, false, "npc_dota_hero_hero1");
        death(a, 515.0f, 15450, 1, 6, false, "npc_dota_hero_hero1");
        death(a, 530.0f, 15900, 1, 7, false, "npc_dota_hero_hero1");
        a.resolveKillCredits();

        List<Map<String, Object>> fights = a.detectTeamfights();

        assertEquals(1, fights.size());
        Map<String, Object> fight = fights.get(0);
        assertEquals(85, fight.get("start"));         // first death - 15 s
        assertEquals(139, fight.get("end"));          // last death + 15 s
        assertEquals(100, fight.get("first_death"));
        assertEquals(124, fight.get("last_death"));
        assertEquals(1, fight.get("radiant_deaths"));
        assertEquals(2, fight.get("dire_deaths"));
        List<Map<String, Object>> players = (List<Map<String, Object>>) fight.get("players");
        Map<String, Object> hero1 = players.get(1);
        assertEquals(1, hero1.get("kills"));
        assertEquals(1, hero1.get("deaths"));
        assertEquals(19.7, (Double) hero1.get("death_time_offset"), 0.01);
        List<Map<String, Object>> targets = (List<Map<String, Object>>) hero1.get("kill_targets");
        assertEquals(1, targets.size());
        assertEquals(128, targets.get(0).get("target_slot"));
        assertEquals(15.2, (Double) targets.get(0).get("time_offset"), 0.01);
    }

    @Test
    void teamfightGapIsMeasuredOnTheRoundedClock() {
        ReplayAnalyzer a = analyzer();
        // Match 9038377636: deaths at 347.5, 362.3 and 363.3 s. OpenDota's clock rounds them to
        // 348 and 362, 14 s apart: one fight (333-378 there). Truncated they are 15 s apart.
        death(a, 347.5f, 10425, 1, 9, false, "npc_dota_hero_hero1");
        death(a, 362.3f, 10869, 2, 6, false, "npc_dota_hero_hero2");
        death(a, 363.3f, 10899, 0, 7, false, "npc_dota_hero_hero0");
        a.resolveKillCredits();

        List<Map<String, Object>> fights = a.detectTeamfights();

        assertEquals(1, fights.size());
        assertEquals(332, fights.get(0).get("start"));
        assertEquals(378, fights.get(0).get("end"));
        assertEquals(3, fights.get(0).get("dire_deaths"));
    }

    private static List<String> skillBuild(ReplayAnalyzer.PlayerState ps) {
        List<String> build = new ArrayList<>();
        for (Map<String, Object> up : ps.entityAbilityUpgrades) build.add(up.get("name") + " " + up.get("level"));
        return build;
    }

    @Test
    void aLevelTrainedWithoutASkillPointIsASkillUp() {
        ReplayAnalyzer a = analyzer();
        ReplayAnalyzer.PlayerState wyvern = a.playerStates[7];
        wyvern.lastAbilityPoints = 0;  // the replay has m_iAbilityPoints, and no point is left
        ReplayAnalyzer.AbilityTrack arcticBurn = new ReplayAnalyzer.AbilityTrack("winter_wyvern_arctic_burn", 3, 0, 700);
        ReplayAnalyzer.AbilityTrack edda = new ReplayAnalyzer.AbilityTrack("winter_wyvern_eldwurms_edda", 1, 3, 703);

        // Match 9037747302 at 11:34: Winter Wyvern consumes Eldwurm's Edda. The replay has a
        // train order for Arctic Burn and its level 3 -> 4; m_iAbilityPoints does not move.
        a.currentTick = 43000;
        a.trainOrderTicks.put(700, 43000);
        a.recordSkillUps(wyvern, List.of(arcticBurn), List.of(4), 784f);
        assertEquals(List.of("winter_wyvern_arctic_burn 4"), skillBuild(wyvern));
        assertEquals(0, wyvern.pendingSpentPoints);

        // The order explained that one level. A level that rises by itself - no point, no order
        // of its own - stays out, also within the second after the order.
        a.currentTick = 43015;
        a.recordSkillUps(wyvern, List.of(arcticBurn, edda), List.of(5, 2), 784.5f);
        assertEquals(1, wyvern.entityAbilityUpgrades.size());
        assertEquals(5, arcticBurn.level);

        // A point spent later is a skill-up as before, with the level the ability really has.
        ReplayAnalyzer.AbilityTrack coldEmbrace = new ReplayAnalyzer.AbilityTrack("winter_wyvern_cold_embrace", 3, 2, 702);
        a.currentTick = 50000;
        a.trainOrderTicks.put(702, 50000);
        wyvern.pendingSpentPoints = 1;
        a.recordSkillUps(wyvern, List.of(coldEmbrace), List.of(4), 1017f);
        assertEquals(List.of("winter_wyvern_arctic_burn 4", "winter_wyvern_cold_embrace 4"), skillBuild(wyvern));
        assertEquals(0, wyvern.pendingSpentPoints);
    }

    @Test
    void withoutAnOrderOrWithoutPointTrackingOnlyAPointMakesASkillUp() {
        ReplayAnalyzer a = analyzer();
        // Two levels at once are never the one free level of a train order.
        ReplayAnalyzer.PlayerState ps = a.playerStates[0];
        ps.lastAbilityPoints = 0;
        ReplayAnalyzer.AbilityTrack ability = new ReplayAnalyzer.AbilityTrack("lion_impale", 1, 0, 500);
        a.currentTick = 9000;
        a.trainOrderTicks.put(500, 9000);
        a.recordSkillUps(ps, List.of(ability), List.of(3), 300f);
        assertTrue(ps.entityAbilityUpgrades.isEmpty());

        // An old replay without m_iAbilityPoints: the hero level caps the build, an order adds nothing.
        ReplayAnalyzer.PlayerState old = a.playerStates[1];
        old.level = 2;
        old.entityUpgradeCount = 2;
        ReplayAnalyzer.AbilityTrack third = new ReplayAnalyzer.AbilityTrack("lion_voodoo", 0, 1, 501);
        a.trainOrderTicks.put(501, 9000);
        a.recordSkillUps(old, List.of(third), List.of(1), 300f);
        assertTrue(old.entityAbilityUpgrades.isEmpty());
    }

    @Test
    void combatLogTeamsBecomeCodes() {
        assertEquals("radiant", ReplayAnalyzer.unitTeamName(ReplayAnalyzer.TEAM_RADIANT));
        assertEquals("dire", ReplayAnalyzer.unitTeamName(ReplayAnalyzer.TEAM_DIRE));
        assertEquals("neutral", ReplayAnalyzer.unitTeamName(4));  // camp creeps, Roshan
        assertNull(ReplayAnalyzer.unitTeamName(0));               // the row names no team
    }

    @Test
    void matchIdTypeGetsTheSixtyFourBitDecoder() {
        // Clarity's default for the unknown type "MatchID_t" is a 32-bit varint (id modulo 2^32).
        assertTrue(ReplayAnalyzer.registerMatchIdDecoder());
        Object decoder = skadistats.clarity.io.s2.S2DecoderFactory.createDecoder("MatchID_t").getDecoder();
        assertTrue(decoder instanceof skadistats.clarity.io.decoder.LongVarUnsignedDecoder, String.valueOf(decoder));
    }
}
