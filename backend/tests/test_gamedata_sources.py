"""Tests for where the game data comes from and which game version it is for.

``data_manager`` downloads the dotaconstants files from the first source
that delivers a complete set, reads the full version (with letter patches)
from Valve's patch feed and exposes facts the AI layer must not hard-code
(hero vision, ward / smoke values). No network: ``_http_get_json`` — the
module's only network call — is replaced by a small fake server.
"""
import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import json
import re
import time
from types import SimpleNamespace

import pytest

from backend.app.ai import data_manager, game_data

GITHUB, JSDELIVR, API = (source for source, _template in data_manager.CONSTANTS_SOURCES)

_HEROES = {
    "2": {"id": 2, "name": "npc_dota_hero_axe", "localized_name": "Axe", "primary_attr": "str",
          "attack_type": "Melee", "roles": ["Initiator"], "day_vision": 1800, "night_vision": 800},
    "60": {"id": 60, "name": "npc_dota_hero_night_stalker", "localized_name": "Night Stalker",
           "primary_attr": "str", "attack_type": "Melee", "roles": ["Carry"],
           "day_vision": 800, "night_vision": 1800},
    "35": {"id": 35, "name": "npc_dota_hero_sniper", "localized_name": "Sniper", "primary_attr": "agi",
           "attack_type": "Ranged", "roles": ["Carry"]},        # states no vision
}
_CONSTANTS = {
    "heroes": _HEROES,
    "hero_abilities": {"npc_dota_hero_axe": {"abilities": ["axe_berserkers_call", "axe_culling_blade"]}},
    "abilities": {
        "axe_berserkers_call": {"dname": "Berserker's Call", "cd": ["18", "16", "14", "12"]},
        "axe_culling_blade": {"dname": "Culling Blade", "cd": ["75", "65", "55"]},
    },
    "items": {
        "heart": {"dname": "Heart of Tarrasque", "cost": 5300, "id": 114},
        "ward_sentry": {"dname": "Sentry Ward", "cost": 50, "id": 43, "attrib": [
            {"key": "lifetime", "value": "420"}, {"key": "true_sight_range", "value": "1050"}]},
        "smoke_of_deceit": {"dname": "Smoke of Deceit", "cost": 50, "id": 188, "attrib": [
            {"key": "application_radius", "value": "1200"}, {"key": "duration", "value": "45.0"}]},
    },
    "patch": [{"name": "7.40", "date": "2025-12-16T00:50:40.281Z", "id": 59},
              {"name": "7.41", "date": "2026-03-24T00:50:59.580Z", "id": 60}],
}
# Valve's feed for 7.40c .. 7.41f (patch_timestamp is the release time, UTC)
_RELEASES = [("7.40c", "2026-01-21T08:00:00Z"), ("7.41", "2026-03-24T07:00:00Z"),
             ("7.41a", "2026-03-27T07:00:00Z"), ("7.41b", "2026-04-07T07:00:00Z"),
             ("7.41c", "2026-05-06T07:00:00Z"), ("7.41d", "2026-06-04T07:00:00Z"),
             ("7.41e", "2026-07-30T07:00:00Z"), ("7.41f", "2026-09-15T07:00:00Z")]
_REVISION = {"sha": "51138e3d950f6b20ece7b624797eed8cf8855611", "date": "2026-10-10T04:31:58Z"}
# the mirror asked for one commit instead of the branch
_JSDELIVR_COMMIT = re.compile(
    r"https://cdn\.jsdelivr\.net/gh/odota/dotaconstants@(?P<commit>[0-9a-f]{12,40})/build/(?P<name>\w+)\.json")


def _valve(releases=_RELEASES):
    return {"success": True, "patches": [
        {"patch_number": label, "patch_name": label, "patch_timestamp": int(data_manager._parse_utc(date))}
        for label, date in releases]}


def _commits(revision=_REVISION):
    return [{"sha": revision["sha"], "commit": {"committer": {"date": revision["date"]}}}]


@pytest.fixture
def net(tmp_path, monkeypatch):
    """Empty game-data dir and a fake network.

    ``net.sources`` maps a source id to the files it serves (a missing id or
    file is unreachable, an Exception value is raised); ``net.valve`` /
    ``net.revision`` are the answers of Valve's feed and of the revision
    check (None: unreachable). jsDelivr serves ``net.sources[JSDELIVR]``
    for the branch and for a commit alike, unless ``net.at_commit`` has
    files for that commit. ``net.calls`` lists the requested URLs.
    """
    gamedata = tmp_path / "gamedata"
    gamedata.mkdir()
    monkeypatch.setattr(data_manager, "GAMEDATA_DIR", gamedata)
    monkeypatch.setattr(data_manager, "MIN_TABLE_SIZES", {})      # the data set here has three heroes
    state = SimpleNamespace(gamedata=gamedata, sources={}, valve=None, revision=None, at_commit={}, calls=[])
    prefixes = {source: template.split("{name}")[0] for source, template in data_manager.CONSTANTS_SOURCES}

    def answer(payload, url):
        if payload is None:
            raise OSError(f"unreachable: {url}")
        if isinstance(payload, Exception):
            raise payload
        return json.loads(json.dumps(payload))

    def get(url, timeout=30):
        state.calls.append(url)
        if url == data_manager.VALVE_PATCH_FEED:
            return answer(state.valve, url)
        if url == data_manager.SOURCE_REVISION_URL:
            return answer(state.revision, url)
        pinned = _JSDELIVR_COMMIT.fullmatch(url)
        if pinned:
            files = state.at_commit.get(pinned["commit"], state.sources.get(JSDELIVR))
            return answer((files or {}).get(pinned["name"]), url)
        for source, prefix in prefixes.items():
            if url.startswith(prefix):
                name = url[len(prefix):].removesuffix(".json")
                return answer((state.sources.get(source) or {}).get(name), url)
        raise AssertionError(f"unexpected request: {url}")

    def fetched(source):
        """Names of the constants files requested from *source*."""
        names = [url[len(prefixes[source]):].removesuffix(".json")
                 for url in state.calls if url.startswith(prefixes[source])]
        if source == JSDELIVR:
            names += [m["name"] for m in map(_JSDELIVR_COMMIT.fullmatch, state.calls) if m]
        return names

    state.fetched, state.get = fetched, get
    monkeypatch.setattr(data_manager, "_http_get_json", get)
    data_manager._source_down_until.clear()                       # what earlier tests found unreachable
    data_manager.invalidate_all_caches()
    yield state
    data_manager._source_down_until.clear()
    data_manager.invalidate_all_caches()


def _meta(net):
    return json.loads((net.gamedata / "meta.json").read_text(encoding="utf-8"))


def _online(net, **overrides):
    """Everything reachable: master on GitHub, Valve's feed, the revision check."""
    net.sources = {GITHUB: dict(_CONSTANTS), JSDELIVR: dict(_CONSTANTS), API: dict(_CONSTANTS)}
    net.valve, net.revision = _valve(), _commits()
    for key, value in overrides.items():
        setattr(net, key, value)


# ---------------------------------------------------------------------------
# PA03: fetch source
# ---------------------------------------------------------------------------

def test_fetch_prefers_dotaconstants_master_and_records_where_the_data_came_from(net):
    _online(net)
    result = data_manager.fetch_and_save_gamedata()
    assert (result["status"], result["reason"], result["source"]) == ("updated", "no_data", GITHUB)
    assert net.fetched(GITHUB) == ["patch", "heroes", "hero_abilities", "abilities", "items"]
    assert net.fetched(JSDELIVR) == [] and net.fetched(API) == []   # the frozen npm release is not asked

    meta = _meta(net)
    assert meta["schema_version"] == data_manager.GAMEDATA_SCHEMA_VERSION
    assert meta["source"] == GITHUB
    assert meta["source_url"] == "https://raw.githubusercontent.com/odota/dotaconstants/master/build"
    assert (meta["source_revision"], meta["source_date"]) == ("51138e3d950f", "2026-10-10T04:31:58Z")
    # major version for comparisons, full label of what the data can reflect
    assert (meta["patch_version"], meta["data_label"]) == ("7.41", "7.41f")
    assert data_manager.get_meta()["source"] == GITHUB
    assert game_data.get_item_data("Heart of Tarrasque")["cost"] == 5300


def test_a_source_that_fails_for_one_file_is_given_up_as_a_whole(net):
    _online(net)
    net.sources[GITHUB]["abilities"] = OSError("connection reset")
    result = data_manager.fetch_and_save_gamedata(force=True)
    # heroes / hero_abilities of GitHub are thrown away: one fetch, one source
    assert result["source"] == JSDELIVR
    assert net.fetched(JSDELIVR) == ["heroes", "hero_abilities", "abilities", "items"]
    assert net.fetched(GITHUB)[-3:] == ["heroes", "hero_abilities", "abilities"]
    meta = _meta(net)
    assert (meta["source"], meta["source_revision"], meta["data_label"]) == (JSDELIVR, "51138e3d950f", "7.41f")


def test_the_opendota_api_is_the_last_resort_and_its_age_is_not_guessed(net):
    _online(net)
    net.sources[GITHUB] = None
    net.sources[JSDELIVR] = {"patch": _CONSTANTS["patch"], "heroes": {}}   # an empty payload is not data
    result = data_manager.fetch_and_save_gamedata(force=True)
    assert result["source"] == API
    meta = _meta(net)
    # the npm release behind the API has no known date: no revision, no letter
    assert (meta["source"], meta["source_revision"], meta["source_date"], meta["data_label"]) == (API, "", "", "")
    assert meta["patch_version"] == "7.41"
    info = data_manager.get_patch_info()
    assert (info["label"], info["data_label"], info["data_current"]) == ("7.41f", "", None)


def test_nothing_is_written_when_no_source_delivers_a_valid_set(net):
    _online(net)
    for source in (GITHUB, JSDELIVR, API):
        net.sources[source]["items"] = {"heart": "not an item"}     # normalises to nothing
    with pytest.raises(RuntimeError):
        data_manager.fetch_and_save_gamedata(force=True)
    assert [p.name for p in net.gamedata.iterdir()] == []
    # ... and an earlier, complete data set stays as it was
    net.sources[GITHUB]["items"] = _CONSTANTS["items"]
    data_manager.fetch_and_save_gamedata(force=True)
    before = {p.name: p.read_bytes() for p in net.gamedata.iterdir()}
    net.sources[GITHUB]["items"] = OSError("down")
    net.sources[JSDELIVR]["items"] = OSError("down")
    with pytest.raises(RuntimeError):
        data_manager.fetch_and_save_gamedata(force=True)
    assert {p.name: p.read_bytes() for p in net.gamedata.iterdir()} == before


def test_download_constants_set_takes_every_file_from_one_source(net):
    net.sources = {GITHUB: {"item_ids": {"1": "blink"}},
                   JSDELIVR: {"item_ids": {"1": "blink"}, "ability_ids": {"5003": "antimage_mana_break"}}}
    payloads, source = data_manager.download_constants_set(("item_ids", "ability_ids"))
    assert source == JSDELIVR and payloads["ability_ids"] == {"5003": "antimage_mana_break"}
    assert data_manager.download_constants("item_ids") == {"1": "blink"}      # GitHub has this one
    with pytest.raises(OSError):
        data_manager.download_constants("xp_level")


def test_an_unreachable_source_is_tried_last_by_the_next_download(net, monkeypatch):
    """A host that cannot reach GitHub or jsDelivr: only the first import waits for their timeouts.

    The OpenDota parser downloads a table when it meets an id the bundled
    catalog does not know; each download used to walk the sources in order
    (2 x 10 s of timeouts per file).
    """
    import urllib.error

    net.sources = {API: {"item_ids": {"1": "blink"}, "ability_ids": {"5003": "antimage_mana_break"}}}
    assert data_manager.download_constants("item_ids") == {"1": "blink"}
    assert [net.fetched(source) for source in (GITHUB, JSDELIVR, API)] == [["item_ids"], ["item_ids"], ["item_ids"]]
    net.calls.clear()
    payloads, source = data_manager.download_constants_set(("item_ids", "ability_ids"))
    assert source == API and net.fetched(GITHUB) == net.fetched(JSDELIVR) == []      # straight to the one that answers
    assert net.fetched(API) == ["item_ids", "ability_ids"] and payloads["item_ids"] == {"1": "blink"}

    # The unreachable ones stay the last resort ...
    net.calls.clear()
    net.sources = {JSDELIVR: {"xp_level": [0, 240]}}
    assert data_manager.download_constants("xp_level") == [0, 240]
    assert [len(net.fetched(source)) for source in (API, GITHUB, JSDELIVR)] == [1, 1, 1]
    # ... one that answers is in its place again (the API is now the one that failed) ...
    net.calls.clear()
    net.sources[GITHUB] = {"xp_level": [0, 240]}
    data_manager.download_constants("xp_level")
    assert net.fetched(JSDELIVR) == ["xp_level"] and net.fetched(API) == net.fetched(GITHUB) == []
    # ... and after the retry interval the configured order is back
    monkeypatch.setattr(data_manager, "SOURCE_RETRY_S", 0.0)
    data_manager._source_down_until.clear()
    net.sources = {API: {"item_ids": {"1": "blink"}}}
    data_manager.download_constants("item_ids")                   # marks expire at once
    net.calls.clear()
    data_manager.download_constants("item_ids")
    assert [len(net.fetched(source)) for source in (GITHUB, JSDELIVR, API)] == [1, 1, 1]

    # A source that answers "no such file" (or sends something unusable) is reachable: no mark
    monkeypatch.setattr(data_manager, "SOURCE_RETRY_S", 1800.0)
    data_manager._source_down_until.clear()
    missing = urllib.error.HTTPError("https://example.invalid/x.json", 404, "Not Found", None, None)
    net.sources = {GITHUB: {"no_such_table": missing, "item_ids": {"1": "blink"}},
                   JSDELIVR: {"no_such_table": ValueError("not JSON")}, API: {"no_such_table": {}}}
    assert data_manager.download_constants("no_such_table") == {}
    assert data_manager._source_down_until == {}
    net.calls.clear()
    data_manager.download_constants("item_ids")
    assert net.fetched(GITHUB) == ["item_ids"] and net.fetched(API) == []
    busy = urllib.error.HTTPError("https://example.invalid/x.json", 503, "Unavailable", None, None)
    assert data_manager._is_unreachable(busy) and data_manager._is_unreachable(TimeoutError())
    assert data_manager._is_unreachable(urllib.error.URLError("no route")) and not data_manager._is_unreachable(missing)


def test_the_game_data_fetch_walks_the_sources_in_order_whatever_was_unreachable(net):
    """Which source may replace stored data is the fetch's own decision: no reordering there."""
    net.sources = {API: {"item_ids": {"1": "blink"}}}
    data_manager.download_constants("item_ids")                   # GitHub and jsDelivr: marked unreachable
    assert set(data_manager._source_down_until) == {GITHUB, JSDELIVR}
    _online(net)
    net.calls.clear()
    result = data_manager.fetch_and_save_gamedata(force=True)
    assert result["source"] == GITHUB and net.fetched(API) == []
    assert GITHUB not in data_manager._source_down_until           # it delivered: reachable again


def test_the_mirror_is_asked_for_the_commit_that_gets_recorded(net):
    """jsDelivr caches a branch for hours. Asked for "master" it may answer
    with an older build, which would then be stored under the newest revision
    and never be fetched again."""
    _online(net)
    net.sources[GITHUB] = None
    stale = json.loads(json.dumps(_CONSTANTS))
    stale["items"]["heart"]["cost"] = 5200                          # the branch as the CDN still has it
    net.sources[JSDELIVR] = stale
    net.at_commit[_REVISION["sha"]] = dict(_CONSTANTS)              # that exact commit
    result = data_manager.fetch_and_save_gamedata()
    assert (result["source"], result["source_revision"]) == (JSDELIVR, "51138e3d950f")
    assert game_data.get_item_data("Heart of Tarrasque")["cost"] == 5300
    commit_url = f"https://cdn.jsdelivr.net/gh/odota/dotaconstants@{_REVISION['sha']}/build"
    assert _meta(net)["source_url"] == commit_url
    assert [url for url in net.calls if "jsdelivr" in url] == [
        f"{commit_url}/{name}.json" for name in ("patch", "heroes", "hero_abilities", "abilities", "items")]
    assert data_manager.check_patch_version() == ("7.41f", False)


def test_a_mirror_copy_of_unknown_age_is_stored_without_a_revision(net):
    """Without the revision the branch is all that can be asked for; what it
    serves is then recorded as of unknown age, and fetched again as soon as
    the revision can be read."""
    _online(net, revision=None)
    net.sources[GITHUB] = None
    net.sources[JSDELIVR] = json.loads(json.dumps(_CONSTANTS))
    net.sources[JSDELIVR]["items"]["heart"]["cost"] = 5200
    result = data_manager.fetch_and_save_gamedata()
    meta = _meta(net)
    assert (result["source"], meta["source_revision"], meta["data_label"]) == (JSDELIVR, "", "")
    assert meta["source_url"] == "https://cdn.jsdelivr.net/gh/odota/dotaconstants@master/build"
    net.revision = _commits()
    net.at_commit[_REVISION["sha"]] = dict(_CONSTANTS)
    assert data_manager.check_patch_version() == ("7.41f", True)
    result = data_manager.refresh_gamedata_if_outdated()
    assert (result["reason"], result["source_revision"]) == ("new_source_revision", "51138e3d950f")
    assert game_data.get_item_data("Heart of Tarrasque")["cost"] == 5300
    # an answer that is no commit id never reaches a URL
    net.revision = [{"sha": "../../evil", "commit": {"committer": {"date": "2026-10-12T00:00:00Z"}}}]
    net.calls.clear()
    data_manager.check_patch_version()
    assert not [url for url in net.calls if "evil" in url]
    assert data_manager._remote_state()["revision"] is None        # ... nor gets recorded as a revision


def _master_data_then_only_the_api(net):
    """Data fetched from master; then GitHub raw and jsDelivr stop answering
    while the OpenDota API still serves its months-old release (Heart 5100)."""
    _online(net)
    data_manager.fetch_and_save_gamedata()
    assert game_data.get_item_data("Heart of Tarrasque")["cost"] == 5300
    release = json.loads(json.dumps(_CONSTANTS))
    release["items"]["heart"]["cost"] = 5100
    net.sources = {API: release}
    return {path.name: path.read_bytes() for path in net.gamedata.iterdir() if path.name != "patches.json"}


def test_the_older_api_release_never_replaces_data_from_master(net):
    before = _master_data_then_only_the_api(net)
    # trigger 1: the source published a new build, but only the API can be reached
    net.revision = _commits({"sha": "f" * 40, "date": "2026-10-12T00:00:00Z"})
    generation = data_manager.data_generation()
    result = data_manager.refresh_gamedata_if_outdated()
    assert result["status"] == "failed" and result["patch_version"] == "7.41"
    assert result["error"].startswith(
        f"RuntimeError: Stored game data kept ({GITHUB} 51138e3d950f): the OpenDota API has no newer major "
        "version, and master failed with OSError: unreachable")
    assert "heroes" not in net.fetched(API)                         # the 3 MB were not even downloaded
    with pytest.raises(RuntimeError, match="Stored game data kept"):
        data_manager.fetch_and_save_gamedata(force=True)            # the CLI's --force is no way around it
    assert {path.name: path.read_bytes() for path in net.gamedata.iterdir() if path.name != "patches.json"} == before
    assert game_data.get_item_data("Heart of Tarrasque")["cost"] == 5300
    assert data_manager.data_generation() == generation             # no cache, no RAG store was dropped
    info = data_manager.get_patch_info()
    assert (info["data_label"], info["data_current"]) == ("7.41f", True)

    # master is back: the pending revision is fetched
    net.sources[GITHUB] = dict(_CONSTANTS)
    result = data_manager.refresh_gamedata_if_outdated()
    assert (result["status"], result["source"], result["source_revision"]) == ("updated", GITHUB, "f" * 12)


def test_the_weekly_refresh_does_not_downgrade_either(net):
    before = _master_data_then_only_the_api(net)
    meta = _meta(net)
    meta["last_fetched"] = "2026-09-01T00:00:00Z"                  # stale: the refresh is forced
    (net.gamedata / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    data_manager.invalidate_all_caches()
    assert data_manager.get_meta()["is_stale"] is True
    result = data_manager.refresh_gamedata_if_outdated()
    assert result["status"] == "failed" and "Stored game data kept" in result["error"]
    assert game_data.get_item_data("Heart of Tarrasque")["cost"] == 5300
    assert _meta(net)["source"] == GITHUB and before["items.json"] == (net.gamedata / "items.json").read_bytes()


def test_the_api_may_replace_master_data_once_it_is_a_major_version_ahead(net):
    """The npm release is cut on major patches: when it names a newer major
    version than the stored data, it is the newer of the two."""
    _master_data_then_only_the_api(net)
    net.sources[API]["patch"] = _CONSTANTS["patch"] + [{"name": "7.42", "date": "2026-12-01T00:00:00Z"}]
    net.valve = _valve(_RELEASES + [("7.42", "2026-12-01T07:00:00Z")])
    result = data_manager.fetch_and_save_gamedata(force=True)
    assert (result["status"], result["source"], result["patch_version"]) == ("updated", API, "7.42")
    assert game_data.get_item_data("Heart of Tarrasque")["cost"] == 5100
    # and data that came from the API to begin with is refreshed from it as before
    assert data_manager.fetch_and_save_gamedata(force=True)["source"] == API


def _full_constants():
    """A data set of plausible size: 120 heroes with an ability each, 360 items."""
    heroes = {str(n): {"id": n, "name": f"npc_dota_hero_h{n}", "localized_name": f"Hero {n}",
                       "primary_attr": "str", "attack_type": "Melee", "roles": []} for n in range(1, 121)}
    return {
        "heroes": heroes,
        "hero_abilities": {hero["name"]: {"abilities": [f"h{n}_strike"]} for n, hero in heroes.items()},
        "abilities": {**{f"h{n}_strike": {"dname": f"Strike {n}", "cd": "10"} for n in heroes},
                      **{f"special_bonus_unique_h{n}": {"dname": f"+{n} Strike Damage"} for n in heroes}},
        "items": {f"item_{n}": {"dname": f"Item {n}", "cost": n, "id": n} for n in range(1, 361)},
        "patch": _CONSTANTS["patch"],
    }


def test_a_truncated_upstream_build_does_not_replace_a_complete_data_set(net, monkeypatch):
    """Master is a moving branch: a build that lost most hero ability lists
    or items is not data, even though none of its tables is empty."""
    monkeypatch.undo()                                            # the real minimum sizes ...
    monkeypatch.setattr(data_manager, "GAMEDATA_DIR", net.gamedata)   # ... with the fake network kept
    monkeypatch.setattr(data_manager, "_http_get_json", lambda url, timeout=30: net.get(url, timeout))
    data_manager.invalidate_all_caches()
    assert data_manager.MIN_TABLE_SIZES == {"heroes": 100, "abilities": 100, "items": 300}
    full = _full_constants()
    # 0. nothing stored yet: a build with five hero ability lists is refused for what it is
    _online(net, sources={GITHUB: dict(full, hero_abilities=dict(list(full["hero_abilities"].items())[:5])),
                          JSDELIVR: dict(full), API: dict(full)})
    result = data_manager.fetch_and_save_gamedata()
    assert result["source"] == JSDELIVR
    assert (result["heroes"], result["abilities"], result["items"], result["talents"]) == (120, 120, 360, 120)
    before = {path.name: path.read_bytes() for path in net.gamedata.iterdir() if path.name != "patches.json"}

    # 1. below the absolute minimum: five heroes still have abilities, forty items are left
    broken = dict(full, hero_abilities=dict(list(full["hero_abilities"].items())[:5]),
                  items=dict(list(full["items"].items())[:40]))
    net.sources = {GITHUB: broken, JSDELIVR: broken, API: broken}
    net.revision = _commits({"sha": "b" * 40, "date": "2026-10-12T00:00:00Z"})
    result = data_manager.refresh_gamedata_if_outdated()
    assert result["status"] == "failed" and "Implausible constants: 5 abilities" in result["error"]
    assert {path.name: path.read_bytes() for path in net.gamedata.iterdir() if path.name != "patches.json"} == before

    # 2. above the minimum, but a fifth of the stored talents / a seventh of the items is gone
    for table, kept, message in (("abilities", {k: v for k, v in full["abilities"].items()
                                                if not k.startswith("special_bonus_") or int(k.rsplit("h", 1)[1]) > 24},
                                  "Implausible constants: 96 talents, the stored data has 120"),
                                 ("items", dict(list(full["items"].items())[:310]),
                                  "Implausible constants: 310 items, the stored data has 360")):
        shrunk = dict(full, **{table: kept})
        net.sources = {GITHUB: shrunk, JSDELIVR: dict(full), API: dict(full)}
        result = data_manager.fetch_and_save_gamedata(force=True)
        assert result["source"] == JSDELIVR, message                # the broken source is given up, the next one used
        net.sources[JSDELIVR] = shrunk
        with pytest.raises(RuntimeError, match="Stored game data kept") as raised:
            data_manager.fetch_and_save_gamedata(force=True)
        assert message in str(raised.value)
        # a look at the cause may say the loss is real: the operator can accept it
        if table == "items":
            result = data_manager.fetch_and_save_gamedata(force=True, allow_shrink=True)
            assert (result["status"], result["items"]) == ("updated", 310)

    # 3. files of an older format say nothing about what is plausible (they counted recipes as items)
    meta = _meta(net)
    meta.update(schema_version=3, item_count=484)
    (net.gamedata / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    data_manager.invalidate_all_caches()
    net.sources = {GITHUB: dict(full)}
    assert data_manager.fetch_and_save_gamedata()["items"] == 360


def test_item_names_are_one_line(net):
    """dotaconstants hands Valve's two-line name on as "Black Grimoire\\n(Warlock)"."""
    assert data_manager.clean_item_name("Black Grimoire\\n(Warlock)") == "Black Grimoire"
    assert data_manager.clean_item_name("Black Grimoire\n(Warlock)") == "Black Grimoire"
    assert data_manager.clean_item_name("  Tango  (Shared) ") == "Tango (Shared)"   # a qualifier on the same line stays
    assert data_manager.clean_item_name("\\n") == "" and data_manager.clean_item_name(None) == ""
    _online(net)
    net.sources[GITHUB]["items"] = {**_CONSTANTS["items"], "black_grimoire": {
        "dname": "Black Grimoire\\n(Warlock)", "cost": 0, "id": 1440}}
    data_manager.fetch_and_save_gamedata()
    stored = json.loads((net.gamedata / "items.json").read_text(encoding="utf-8"))
    assert stored["Black Grimoire"]["name"] == "Black Grimoire" and not [name for name in stored if "\\" in name]
    assert data_manager.item_display_name("black_grimoire") == "Black Grimoire"
    assert data_manager.resolve_item_key("Black Grimoire") == "black_grimoire"
    # files fetched before the names were cleaned are read the same way
    raw_name = "Black Grimoire\\n(Warlock)"
    stored[raw_name] = {**stored.pop("Black Grimoire"), "name": raw_name}
    (net.gamedata / "items.json").write_text(json.dumps(stored), encoding="utf-8")
    data_manager.invalidate_all_caches()
    assert data_manager.item_display_name("black_grimoire") == "Black Grimoire"
    assert data_manager.load_items()["Black Grimoire"]["name"] == "Black Grimoire"


# ---------------------------------------------------------------------------
# PA07: letter patches and the re-check
# ---------------------------------------------------------------------------

def test_default_version_without_any_download(net):
    assert data_manager.DEFAULT_PATCH == data_manager.patch_major(data_manager.DEFAULT_PATCH_LABEL)
    assert data_manager.get_current_patch() == data_manager.DEFAULT_PATCH
    info = data_manager.get_patch_info()
    assert info == {"patch": data_manager.DEFAULT_PATCH, "label": data_manager.DEFAULT_PATCH_LABEL,
                    "released": data_manager.DEFAULT_PATCH_RELEASED, "checked_at": "",
                    "data_label": "", "data_date": "", "data_current": None}
    assert data_manager.get_patch_label() == data_manager.DEFAULT_PATCH_LABEL
    assert data_manager.patch_label_for_time(1775643322) == ""
    meta = data_manager.get_meta()
    assert (meta["status"], meta["source"], meta["patch_label"]) == ("no_data", "fallback", info["label"])
    assert meta["checked_at"] == ""                               # the version was never looked up
    for label, major in (("7.41f", "7.41"), ("7.41", "7.41"), ("7.40c", "7.40"), ("", ""), (None, ""),
                         ("unknown", "unknown")):
        assert data_manager.patch_major(label) == major


def test_full_version_comes_from_valves_feed_and_majors_stay_comparable(net):
    _online(net)
    data_manager.fetch_and_save_gamedata()
    assert data_manager.get_current_patch() == "7.41"            # never with a letter
    assert game_data.PATCH_VERSION == "7.41"
    info = data_manager.get_patch_info()
    assert (info["patch"], info["label"], info["released"]) == ("7.41", "7.41f", "2026-09-15T07:00:00Z")
    assert (info["data_label"], info["data_date"], info["data_current"]) == ("7.41f", "2026-10-10T04:31:58Z", True)
    assert data_manager._parse_utc(info["checked_at"]) is not None
    meta = data_manager.get_meta()
    assert (meta["patch_version"], meta["patch_label"], meta["patch_released"]) == (
        "7.41", "7.41f", "2026-09-15T07:00:00Z")
    assert meta["checked_at"] == info["checked_at"]               # one call tells the admin page everything
    # which letter patch a match was played on (real matches 8762775364 and 8713267082)
    assert data_manager.patch_label_for_time(1775643322) == "7.41b"
    assert data_manager.patch_label_for_time(1772470634) == "7.40c"
    assert data_manager.patch_for_time(1775643322) == "7.41"     # the major table is unchanged
    for junk in (0, None, "soon", float("nan"), True):
        assert data_manager.patch_label_for_time(junk) == ""


def test_a_letter_patch_the_source_has_not_published_yet_is_reported_as_such(net):
    """Valve releases 7.41g while dotaconstants master is unchanged: the label
    moves on, the game data is flagged as older, and nothing is downloaded."""
    _online(net)
    data_manager.fetch_and_save_gamedata()
    net.valve = _valve(_RELEASES + [("7.41g", "2026-10-20T07:00:00Z")])
    net.calls.clear()
    assert data_manager.check_patch_version() == ("7.41g", False)
    assert net.fetched(GITHUB) == ["patch"]
    info = data_manager.get_patch_info()
    assert (info["patch"], info["label"], info["data_label"], info["data_current"]) == (
        "7.41", "7.41g", "7.41f", False)

    # the source rebuilds: one re-check picks it up, and the data label follows
    net.revision = _commits({"sha": "aaaabbbbccccdddd", "date": "2026-10-21T10:00:00Z"})
    assert data_manager.check_patch_version() == ("7.41g", True)
    result = data_manager.refresh_gamedata_if_outdated()
    assert (result["status"], result["reason"], result["data_label"]) == ("updated", "new_source_revision", "7.41g")
    assert _meta(net)["source_revision"] == "aaaabbbbcccc"
    assert data_manager.get_patch_info()["data_current"] is True
    assert data_manager.check_patch_version() == ("7.41g", False)


def test_without_a_source_revision_a_new_release_triggers_one_refetch(net):
    _online(net, revision=None)                                   # GitHub's API is unreachable
    data_manager.fetch_and_save_gamedata()
    meta = _meta(net)
    assert (meta["source"], meta["source_revision"], meta["data_label"]) == (GITHUB, "", "")
    assert data_manager.check_patch_version() == ("7.41f", False)
    # a release after the last fetch: the only hint that the source may have new data
    after = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 3600))
    net.valve = _valve(_RELEASES + [("7.41g", after)])
    assert data_manager.check_patch_version() == ("7.41g", True)
    # a new major version in the source's own patch list counts as well
    net.valve = _valve()
    net.sources[GITHUB]["patch"] = _CONSTANTS["patch"] + [{"name": "7.42", "date": "2026-11-01T00:00:00Z"}]
    assert data_manager.check_patch_version() == ("7.41f", True)


def test_data_from_the_fallback_source_is_fetched_again_once_master_is_reachable(net):
    _online(net)
    net.sources[GITHUB] = net.sources[JSDELIVR] = None
    assert data_manager.fetch_and_save_gamedata()["source"] == API
    assert data_manager.check_patch_version() == ("7.41f", True)             # fallback_source
    net.sources[GITHUB] = dict(_CONSTANTS)
    result = data_manager.refresh_gamedata_if_outdated()
    assert (result["status"], result["reason"], result["source"]) == ("updated", "fallback_source", GITHUB)
    assert data_manager.check_patch_version() == ("7.41f", False)


def test_refresh_is_cheap_when_current_and_never_raises(net):
    _online(net)
    assert data_manager.refresh_gamedata_if_outdated()["status"] == "updated"      # no data yet
    net.calls.clear()
    result = data_manager.refresh_gamedata_if_outdated()
    assert result == {"status": "up_to_date", "patch_version": "7.41", "patch_label": "7.41f"}
    # three small requests, none of the data files
    assert net.calls == [data_manager.VALVE_PATCH_FEED,
                         data_manager.SOURCE_REVISION_URL,
                         data_manager.CONSTANTS_SOURCES[0][1].format(name="patch")]

    # everything down: a status instead of an exception, the local data untouched
    before = {p.name: p.read_bytes() for p in net.gamedata.iterdir()}
    net.sources, net.valve, net.revision = {}, None, None
    result = data_manager.refresh_gamedata_if_outdated()
    assert result["status"] == "failed" and result["patch_version"] == "7.41" and result["error"]
    assert {p.name: p.read_bytes() for p in net.gamedata.iterdir()} == before
    with pytest.raises(RuntimeError):
        data_manager.check_patch_version()                        # the plain check still reports it
    assert data_manager.get_patch_label() == "7.41f"              # last known version, offline


def test_stale_data_is_fetched_again_but_equal_data_keeps_the_caches(net):
    _online(net)
    data_manager.fetch_and_save_gamedata()
    meta = _meta(net)
    meta["last_fetched"] = "2026-01-01T00:00:00Z"                 # older than GAMEDATA_STALE_DAYS
    (net.gamedata / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    data_manager.invalidate_all_caches()
    assert data_manager.get_meta()["is_stale"] is True
    items = game_data.get_item_table()
    generation = data_manager.data_generation()

    net.calls.clear()
    result = data_manager.refresh_gamedata_if_outdated()
    assert (result["status"], result["changed"], result["reason"]) == ("updated", False, "forced")
    assert "items" in net.fetched(GITHUB)                         # it did download
    assert data_manager.get_meta()["is_stale"] is False           # meta.json is renewed
    # same data: no cache was dropped, so nothing downstream (RAG stores) rebuilds
    assert data_manager.data_generation() == generation
    assert game_data.get_item_table() is items

    # changed data does drop them
    net.sources[GITHUB]["items"] = {**_CONSTANTS["items"],
                                    "heart": {"dname": "Heart of Tarrasque", "cost": 5400, "id": 114}}
    result = data_manager.fetch_and_save_gamedata(force=True)
    assert result["changed"] is True and data_manager.data_generation() > generation
    assert game_data.get_item_data("Heart of Tarrasque")["cost"] == 5400


# ---------------------------------------------------------------------------
# PA08 / PA15: facts the AI layer reads instead of hard-coding
# ---------------------------------------------------------------------------

def test_hero_vision_from_fetched_data_catalog_and_default(net):
    vision = data_manager.hero_vision
    # nothing fetched: the committed catalog
    assert vision("Night Stalker") == (800, 1800)
    assert vision("npc_dota_hero_slark") == (1800, 1800)
    assert vision("sniper") == (1800, 1400)
    assert vision("Axe") == data_manager.DEFAULT_HERO_VISION == (1800, 800)
    for unknown in ("Hero Nobody Knows", "", None, 42):
        assert vision(unknown) == (1800, 800)

    # fetched data wins over the catalog; a value it does not state keeps the catalog's
    _online(net)
    net.sources[GITHUB]["heroes"] = json.loads(json.dumps(_HEROES))
    net.sources[GITHUB]["heroes"]["60"].update(day_vision=900, night_vision=1900)
    net.sources[GITHUB]["heroes"]["2"]["night_vision"] = "broken"
    data_manager.fetch_and_save_gamedata()
    assert vision("Night Stalker") == (900, 1900)
    assert vision("Axe") == (1800, 800) and vision("Sniper") == (1800, 1400)
    stored = json.loads((net.gamedata / "heroes.json").read_text(encoding="utf-8"))
    assert stored["Night Stalker"]["night_vision"] == 1900 and "night_vision" not in stored["Axe"]


def test_game_rules_default_to_the_current_patch_and_follow_fetched_items(net):
    rules = data_manager.game_rules()
    assert rules == {
        "observer_ward_duration": 360, "observer_ward_vision": 1600,
        "sentry_ward_duration": 420, "sentry_ward_true_sight": 1050,
        "smoke_duration": 45, "smoke_application_radius": 1200, "smoke_break_radius": 1025,
        "gem_true_sight": 900, "dust_radius": 1050, "dust_duration": 12,
    }
    # a patch changes the sentry: the fetched attribute is used, the rest keeps its value
    _online(net)
    items = json.loads(json.dumps(_CONSTANTS["items"]))
    items["ward_sentry"]["attrib"] = [{"key": "lifetime", "value": "480"},
                                      {"key": "true_sight_range", "value": "1100"}]
    items["smoke_of_deceit"]["attrib"] = [{"key": "duration", "value": "40.5"},
                                          {"key": "application_radius", "value": ["1", "2"]}]   # not one number
    net.sources[GITHUB]["items"] = items
    data_manager.fetch_and_save_gamedata()
    rules = data_manager.game_rules()
    assert (rules["sentry_ward_duration"], rules["sentry_ward_true_sight"]) == (480, 1100)
    assert (rules["smoke_duration"], rules["smoke_application_radius"]) == (40.5, 1200)
    assert rules["observer_ward_duration"] == 360 and len(rules) == 10


def test_fetched_facts_replace_the_catalogs_for_heroes_and_items(net, tmp_path, monkeypatch):
    """PA14: a committed catalog that lags the fetch must not override it."""
    static = tmp_path / "static"
    static.mkdir()
    (static / "heroes.json").write_text(json.dumps({"by_id": {
        "2": {"id": 2, "name": "Axe", "slug": "axe", "attr": "agi", "attack_type": "Ranged", "roles": ["Support"]}}}), encoding="utf-8")
    (static / "items.json").write_text(json.dumps({
        "by_key": {"heart": {"id": 114, "key": "heart", "name": "Heart of Tarasque (old spelling)", "cost": 5000},
                   "recipe_heart": {"id": 113, "key": "recipe_heart", "name": "Heart of Tarrasque Recipe", "cost": 1300}},
        "name_to_key": {"Heart of Tarasque (old spelling)": "heart"}}), encoding="utf-8")
    monkeypatch.setattr(data_manager, "STATIC_DATA_DIR", static)
    data_manager.invalidate_all_caches()
    assert data_manager.load_heroes()["Axe"]["attribute"] == "agi"          # nothing fetched: the catalog
    assert data_manager.resolve_item_name("heart") == "Heart of Tarasque (old spelling)"

    _online(net)
    data_manager.fetch_and_save_gamedata()
    axe = data_manager.load_heroes()["Axe"]
    assert (axe["attribute"], axe["attack_type"], axe["roles"]) == ("str", "melee", ["Initiator"])
    assert data_manager.resolve_item_name("heart") == "Heart of Tarrasque"
    assert data_manager.resolve_item_key("Heart of Tarasque (old spelling)") == "heart"   # the old name still resolves
    assert data_manager.resolve_item_key("Heart of Tarrasque Recipe") == "recipe_heart"   # only the catalog has it

    from backend.app.ai.learning import base
    meta = base.load_item_meta()
    assert (meta["heart"]["name"], meta["heart"]["cost"]) == ("Heart of Tarrasque", 5300)   # not the catalog's 5000
    assert meta["recipe_heart"]["cost"] == 1300


def test_map_actions_and_recipes_are_named_in_both_languages(net):
    name = data_manager.ability_display_name
    assert (name("ability_capture", "en"), name("ability_capture", "zh")) == ("Capture Outpost", "占领前哨")
    assert (name("ability_lamp_use", "en"), name("ability_lamp_use", "zh")) == ("Activate Watcher", "启动观察者")
    assert (name("twin_gate_portal_warp", "en"), name("twin_gate_portal_warp", "zh")) == ("Twin Gate", "双生之门")
    assert name("special_bonus_attributes", "zh") == "Attribute Bonus"      # a name, shown untranslated
    # zh is the game's own Chinese name where the catalog has one, and its wording
    # ("图纸（…）") for a recipe the catalog does not name
    item = data_manager.item_display_name
    assert data_manager.is_known_item("recipe_abyssal_blade")
    assert (item("recipe_abyssal_blade", "en"), item("item_recipe_abyssal_blade", "zh")) == (
        "Abyssal Blade Recipe", "图纸（深渊之刃）")
    assert item("Abyssal Blade Recipe", "zh") == "图纸（深渊之刃）"
    assert (item("recipe_thing_nobody_knows", "en"), item("recipe_thing_nobody_knows", "zh")) == (
        "Thing Nobody Knows Recipe", "图纸（Thing Nobody Knows）")
