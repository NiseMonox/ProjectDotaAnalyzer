"""Level 4 learner: coaching knowledge from replay-analysis video transcripts.

Discovers Dota 2 coaching / replay-analysis videos (Bilibili via its public
search API, YouTube via ``yt-dlp ytsearch``), downloads their subtitles with
the ``yt-dlp`` CLI, and distills each transcript into a handful of concrete,
sourced coaching tips through the shared DeepSeek client.

Attribution & scope note: video transcripts are processed ONLY to derive
aggregate coaching knowledge — no transcript text is stored verbatim beyond
the distilled tips — and every emitted chunk carries source attribution
(``provenance = {"url", "title", "uploader"}``).

Entry point::

    learn_from_videos(langs=("zh", "en"), max_videos=3)
        -> {"chunks_written": int, "videos_processed": int, "videos_skipped": int}

Self-iteration: a registry at ``LEARNED_DIR / "_videos_seen.json"`` records
every processed video, so repeated runs skip known videos and learn from new
ones. Sources are user-editable in ``VIDEO_SOURCES_FILE``.

Degrades gracefully: missing yt-dlp, missing API key, network failures, or
videos without subtitles all log a warning / count as skipped — the entry
point never raises.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path

from .base import (
    LEARNED_DIR,
    VIDEO_SOURCES_FILE,
    LearnedChunk,
    current_patch_version,
    save_learned_file,
)

logger = logging.getLogger(__name__)

VIDEOS_SEEN_FILE = LEARNED_DIR / "_videos_seen.json"
LEARNED_FILE_NAME = "video_tips"

_BILIBILI_SEARCH_URL = "https://api.bilibili.com/x/web-interface/search/type"
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Only videos in this duration window are considered (skip shorts and 2h VODs)
_MIN_DURATION_S = 8 * 60
_MAX_DURATION_S = 60 * 60

_MAX_TRANSCRIPT_CHARS = 12000
_MIN_TRANSCRIPT_CHARS = 200
_CANDIDATES_PER_QUERY = 3
_SUBPROCESS_TIMEOUT_S = 180

_DEFAULT_SOURCES = {
    "bilibili": {
        "searches": ["dota2 复盘 教学", "dota2 上分 思路"],
        "videos": [],
    },
    "youtube": {
        "searches": ["dota 2 replay analysis coaching"],
        "videos": [],
    },
}

_SYSTEM_PROMPT = (
    "You are a 7000-MMR Dota 2 coach. You are given the subtitle transcript "
    "of a Dota 2 coaching / replay-analysis video. Extract the CONCRETE "
    "coaching tips the video actually teaches: hero-specific, position-"
    "specific, or phase-specific whenever possible (item timings, skill "
    "usage, map movement, warding spots, decision rules). REJECT generic "
    "fluff like \"practice more and you will improve\" or \"keep a positive "
    "mindset\" — every tip must be something a coach could point at during a "
    "replay review. Each tip must be bilingual (Chinese and English)."
)


def _build_user_prompt(title: str, uploader: str, platform: str, transcript: str) -> str:
    return (
        f"Video: {title!r} by {uploader!r} on {platform}.\n"
        f"Transcript (subtitles, may be noisy auto-captions):\n\n{transcript}\n\n"
        "Extract 3 to 8 concrete coaching tips. Respond with STRICT JSON only, "
        "an object of the form:\n"
        "{\n"
        '  "tips": [\n'
        "    {\n"
        '      "topic": "short_snake_case_slug",\n'
        '      "zh": "中文教练建议（具体、可执行）",\n'
        '      "en": "English coaching tip (concrete and actionable)",\n'
        '      "tags": ["hero:pudge", "phase:laning", "role:support_4"],\n'
        '      "priority": 2\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- tags use hero:<slug>, role:<carry|mid|offlane|support_4|support_5>, "
        "phase:<laning|mid_game|late_game>, mechanic:<name>, item:<slug>; 2-4 tags per tip.\n"
        "- priority: 2 = important, 3 = supplementary.\n"
        "- zh and en must express the SAME tip, natively phrased in each language.\n"
        "- If the transcript teaches nothing concrete, return {\"tips\": []}.\n"
        "- Output nothing but the JSON object."
    )


# ---------------------------------------------------------------------------
# Sources config & seen-registry
# ---------------------------------------------------------------------------

def _load_sources() -> dict:
    """Load the video source config, creating a default template if missing."""
    if not VIDEO_SOURCES_FILE.exists():
        try:
            VIDEO_SOURCES_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(VIDEO_SOURCES_FILE, "w", encoding="utf-8") as f:
                json.dump(_DEFAULT_SOURCES, f, ensure_ascii=False, indent=1)
            logger.info("Created default video sources config: %s", VIDEO_SOURCES_FILE)
        except OSError as exc:
            logger.warning("Cannot create video sources config: %s", exc)
            return dict(_DEFAULT_SOURCES)
    try:
        with open(VIDEO_SOURCES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Malformed video sources config, using defaults: %s", exc)
        return dict(_DEFAULT_SOURCES)


def _load_seen() -> dict:
    if not VIDEOS_SEEN_FILE.exists():
        return {}
    try:
        with open(VIDEOS_SEEN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Malformed videos-seen registry, resetting: %s", exc)
        return {}


def _save_seen(seen: dict) -> None:
    try:
        with open(VIDEOS_SEEN_FILE, "w", encoding="utf-8") as f:
            json.dump(seen, f, ensure_ascii=False, indent=1)
    except OSError as exc:
        logger.warning("Cannot write videos-seen registry: %s", exc)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _parse_duration(value) -> int:
    """Parse a duration to seconds. Accepts int/float or "MM:SS"/"H:MM:SS"."""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value:
        try:
            parts = [int(p) for p in value.split(":")]
        except ValueError:
            return 0
        seconds = 0
        for p in parts:
            seconds = seconds * 60 + p
        return seconds
    return 0


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


# Bilibili's search API returns HTTP 412 without session cookies; a single
# warmup request to the homepage yields the required buvid3 cookie. The same
# cookies must be handed to the yt-dlp subprocess (via --cookies), or its
# subtitle downloads hit the same 412 anti-bot wall.
_bili_opener: urllib.request.OpenerDirector | None = None
_bili_jar = None


def _get_bilibili_opener() -> urllib.request.OpenerDirector:
    global _bili_opener, _bili_jar
    if _bili_opener is None:
        import http.cookiejar

        jar = http.cookiejar.CookieJar()
        # User-provided browser cookies (see _user_bilibili_cookies) make the
        # search API reliable too — load them into the session when present.
        user_cookies = _user_bilibili_cookies()
        if user_cookies is not None:
            try:
                mozilla_jar = http.cookiejar.MozillaCookieJar(str(user_cookies))
                mozilla_jar.load(ignore_discard=True, ignore_expires=True)
                for cookie in mozilla_jar:
                    jar.set_cookie(cookie)
            except Exception as exc:
                logger.warning("Could not load user Bilibili cookies: %s", exc)
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        opener.addheaders = [("User-Agent", _BROWSER_UA)]
        opener.open("https://www.bilibili.com", timeout=20).read(64)
        _bili_opener = opener
        _bili_jar = jar
    return _bili_opener


def _user_bilibili_cookies() -> Path | None:
    """Browser-exported Bilibili cookies provided by the user, if any.

    Bilibili's subtitle endpoints often 412 for anonymous sessions; a
    browser-exported Netscape cookies.txt (the standard yt-dlp workflow)
    makes access reliable. Checked locations: $BILIBILI_COOKIES_FILE, then
    backend/data/bilibili_cookies.txt.
    """
    candidate = os.environ.get("BILIBILI_COOKIES_FILE", "")
    if candidate and Path(candidate).is_file():
        return Path(candidate)
    default = LEARNED_DIR.parent / "bilibili_cookies.txt"
    return default if default.is_file() else None


def _bili_cookies_file(tmpdir: str) -> Path | None:
    """Cookies file to pass to yt-dlp for Bilibili URLs.

    Prefers user-provided browser cookies (reliable); falls back to
    exporting the warmed anonymous session cookies (buvid3), which get
    past some but not all of Bilibili's anti-bot checks. Returns None
    (and only warns) on failure — the caller then runs without cookies.
    """
    user_cookies = _user_bilibili_cookies()
    if user_cookies is not None:
        return user_cookies
    try:
        import http.cookiejar

        _get_bilibili_opener()  # ensure the warmup happened
        mozilla_jar = http.cookiejar.MozillaCookieJar(
            str(Path(tmpdir) / "bili_cookies.txt")
        )
        for cookie in _bili_jar or []:
            mozilla_jar.set_cookie(cookie)
        mozilla_jar.save(ignore_discard=True, ignore_expires=True)
        return Path(mozilla_jar.filename)
    except Exception as exc:
        logger.warning("Could not prepare Bilibili cookies for yt-dlp: %s", exc)
        return None


def _search_bilibili(query: str) -> list[dict]:
    """Search Bilibili videos. Returns candidate dicts, best (most played) first."""
    params = urllib.parse.urlencode({"search_type": "video", "keyword": query})
    url = f"{_BILIBILI_SEARCH_URL}?{params}"
    try:
        with _get_bilibili_opener().open(url, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("Bilibili search failed for %r: %s", query, exc)
        return []
    if data.get("code") != 0:
        logger.warning("Bilibili search returned code=%s for %r", data.get("code"), query)
        return []

    results = (data.get("data") or {}).get("result") or []
    candidates = []
    for item in results:
        if item.get("type") != "video" or not item.get("bvid"):
            continue
        duration = _parse_duration(item.get("duration"))
        if not (_MIN_DURATION_S <= duration <= _MAX_DURATION_S):
            continue
        candidates.append(
            {
                "id": item["bvid"],
                "url": f"https://www.bilibili.com/video/{item['bvid']}",
                "title": _strip_html(item.get("title", "")),
                "uploader": item.get("author", ""),
                "platform": "bilibili",
                "play": int(item.get("play") or 0),
            }
        )
    candidates.sort(key=lambda c: c["play"], reverse=True)
    return candidates[:_CANDIDATES_PER_QUERY]


def _search_youtube(query: str) -> list[dict]:
    """Search YouTube via yt-dlp flat-playlist. Returns candidate dicts."""
    cmd = [
        "yt-dlp",
        f"ytsearch5:{query}",
        "--flat-playlist",
        "--dump-json",
        "--no-warnings",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT_S
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("YouTube search failed for %r: %s", query, exc)
        return []
    if proc.returncode != 0:
        logger.warning(
            "YouTube search failed for %r: %s", query, (proc.stderr or "")[:300]
        )
        return []

    candidates = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        vid = item.get("id")
        if not vid:
            continue
        duration = _parse_duration(item.get("duration") or 0)
        # duration may be missing in flat-playlist mode; only filter when known
        if duration and not (_MIN_DURATION_S <= duration <= _MAX_DURATION_S):
            continue
        candidates.append(
            {
                "id": vid,
                "url": item.get("url") or f"https://www.youtube.com/watch?v={vid}",
                "title": item.get("title", ""),
                "uploader": item.get("uploader") or item.get("channel") or "",
                "platform": "youtube",
                "play": int(item.get("view_count") or 0),
            }
        )
    candidates.sort(key=lambda c: c["play"], reverse=True)
    return candidates[:_CANDIDATES_PER_QUERY]


def _video_id_from_url(url: str, platform: str) -> str:
    if platform == "bilibili":
        m = re.search(r"(BV[0-9A-Za-z]{10})", url)
        if m:
            return m.group(1)
    else:
        m = re.search(r"(?:v=|youtu\.be/|/shorts/)([0-9A-Za-z_-]{11})", url)
        if m:
            return m.group(1)
    return url


def _discover(sources: dict) -> list[dict]:
    """Build the ordered candidate list: user-pinned URLs first, then searches."""
    candidates: list[dict] = []
    for platform in ("bilibili", "youtube"):
        cfg = sources.get(platform) or {}
        for url in cfg.get("videos") or []:
            if isinstance(url, str) and url.strip():
                url = url.strip()
                candidates.append(
                    {
                        "id": _video_id_from_url(url, platform),
                        "url": url,
                        "title": "",
                        "uploader": "",
                        "platform": platform,
                        "play": 0,
                    }
                )
    for query in (sources.get("bilibili") or {}).get("searches") or []:
        candidates.extend(_search_bilibili(query))
        time.sleep(1.0)  # be polite to the search endpoint
    for query in (sources.get("youtube") or {}).get("searches") or []:
        candidates.extend(_search_youtube(query))

    # De-duplicate while preserving order (pinned first)
    unique: list[dict] = []
    seen_ids: set[str] = set()
    for c in candidates:
        if c["id"] not in seen_ids:
            seen_ids.add(c["id"])
            unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# Transcript download & parsing
# ---------------------------------------------------------------------------

def _download_subtitles(url: str, tmpdir: str) -> Path | None:
    """Download subtitles for *url* into *tmpdir* and return the best file."""
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        # Specific tracks, not broad wildcards: requesting many auto-caption
        # variants aggravates YouTube 429 rate limits, and one failed track
        # makes yt-dlp exit non-zero. "ai-zh" is Bilibili's AI-subtitle id.
        "--sub-langs", "zh-Hans,zh-CN,zh,ai-zh,en",
        "--sub-format", "vtt/srt/best",
        "--ignore-errors",
        "--retries", "2",
        "--sleep-requests", "1",
        "--no-warnings",
        "-P", tmpdir,
    ]
    if "bilibili.com" in url:
        cookies = _bili_cookies_file(tmpdir)
        if cookies is not None:
            cmd += ["--cookies", str(cookies)]
    cmd.append(url)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT_S
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Subtitle download failed for %s: %s", url, exc)
        return None
    if proc.returncode != 0:
        logger.warning(
            "yt-dlp exited %d for %s: %s", proc.returncode, url, (proc.stderr or "")[:300]
        )
        # fall through — partial subtitle files may still exist

    sub_files = sorted(
        list(Path(tmpdir).glob("*.vtt")) + list(Path(tmpdir).glob("*.srt")),
        key=lambda p: (0 if ".zh" in p.name.lower() else 1, -p.stat().st_size),
    )
    return sub_files[0] if sub_files else None


_TIMESTAMP_RE = re.compile(r"\d{1,2}:\d{2}(:\d{2})?[.,]\d{3}\s*-->")


def _subtitle_to_text(path: Path) -> str:
    """Convert a VTT/SRT file to plain text (no timestamps, deduped lines)."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Cannot read subtitle file %s: %s", path, exc)
        return ""

    lines: list[str] = []
    prev = ""
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("WEBVTT", "NOTE", "Kind:", "Language:", "STYLE", "REGION")):
            continue
        if _TIMESTAMP_RE.search(line):
            continue
        if re.fullmatch(r"\d+", line):  # SRT cue numbers
            continue
        line = re.sub(r"<[^>]+>", "", line).strip()  # inline styling / karaoke tags
        if not line or line == prev:  # auto-captions repeat lines heavily
            continue
        lines.append(line)
        prev = line
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM distillation
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"^[a-z0-9_]+:[a-z0-9_.'\-]+(:[a-z0-9_.'\-]+)*$")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return slug[:48] or "tip"


def _validate_tips(items) -> list[dict]:
    valid: list[dict] = []
    if not isinstance(items, list):
        return valid
    for item in items:
        if not isinstance(item, dict):
            continue
        zh = str(item.get("zh") or "").strip()
        en = str(item.get("en") or "").strip()
        if not zh or not en:
            continue
        tags: list[str] = []
        for t in item.get("tags") or []:
            if isinstance(t, str):
                t = t.strip().lower().replace(" ", "_")
                if _TAG_RE.match(t) and t not in tags:
                    tags.append(t)
        try:
            priority = int(item.get("priority", 3))
        except (TypeError, ValueError):
            priority = 3
        priority = min(3, max(2, priority))
        valid.append(
            {
                "topic": _slugify(item.get("topic") or zh[:20]),
                "zh": zh,
                "en": en,
                "tags": tags,
                "priority": priority,
            }
        )
        if len(valid) >= 8:
            break
    return valid


async def _distill_transcript(video: dict, transcript: str) -> list[dict]:
    from ..evaluator import MODEL, get_shared_client

    client = get_shared_client()
    response = await client.chat.completions.create(
        model=MODEL,
        max_tokens=6000,
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_prompt(
                    video.get("title", ""),
                    video.get("uploader", ""),
                    video["platform"],
                    transcript[:_MAX_TRANSCRIPT_CHARS],
                ),
            },
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        return _validate_tips(parsed.get("tips") or [])
    return _validate_tips(parsed)


def _tips_to_chunks(video: dict, tips: list[dict], langs: tuple[str, ...], patch: str) -> list[LearnedChunk]:
    provenance = {
        "url": video["url"],
        "title": video.get("title", ""),
        "uploader": video.get("uploader", ""),
    }
    chunks: list[LearnedChunk] = []
    for idx, tip in enumerate(tips):
        for lang in langs:
            content = tip.get(lang)
            if not content:
                continue
            chunks.append(
                LearnedChunk(
                    chunk_id=f"learned:video:{video['platform']}:{video['id']}:{idx:02d}",
                    content=content,
                    lang=lang,
                    category="video_tip",
                    tags=list(tip["tags"]),
                    priority=tip["priority"],
                    source=f"video:{video['platform']}",
                    patch_version=patch,
                    provenance=provenance,
                )
            )
    return chunks


def _load_prior_chunks() -> list[LearnedChunk]:
    """Read existing video_tips.json so re-saving doesn't drop prior videos."""
    path = LEARNED_DIR / f"{LEARNED_FILE_NAME}.json"
    if not path.exists():
        return []
    field_names = {f.name for f in dataclass_fields(LearnedChunk)}
    prior: list[LearnedChunk] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for c in payload.get("chunks", []):
            if isinstance(c, dict) and c.get("chunk_id") and c.get("content"):
                prior.append(LearnedChunk(**{k: v for k, v in c.items() if k in field_names}))
    except Exception as exc:
        logger.warning("Cannot carry forward prior video tips: %s", exc)
    return prior


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def learn_from_videos(langs: tuple[str, ...] = ("zh", "en"), max_videos: int = 3) -> dict:
    """Learn coaching tips from up to *max_videos* new videos.

    Returns ``{"chunks_written", "videos_processed", "videos_skipped"}``.
    Never raises.
    """
    summary = {"chunks_written": 0, "videos_processed": 0, "videos_skipped": 0}

    if shutil.which("yt-dlp") is None:
        logger.warning("video learner: yt-dlp CLI not found, skipping")
        return summary
    try:
        from ..evaluator import is_api_available
        if not is_api_available():
            logger.warning("video learner: DEEPSEEK_API_KEY not set, skipping")
            return summary
    except Exception as exc:
        logger.warning("video learner: LLM client unavailable: %s", exc)
        return summary

    sources = _load_sources()
    seen = _load_seen()

    try:
        candidates = _discover(sources)
    except Exception as exc:
        logger.warning("video learner: discovery failed: %s", exc)
        return summary
    logger.info("video learner: %d candidate videos discovered", len(candidates))

    patch = current_patch_version()
    new_chunks: list[LearnedChunk] = []

    for video in candidates:
        if summary["videos_processed"] >= max_videos:
            break
        if video["id"] in seen:
            continue

        with tempfile.TemporaryDirectory(prefix="dota_video_subs_") as tmpdir:
            sub_file = _download_subtitles(video["url"], tmpdir)
            if sub_file is None:
                logger.warning("video learner: no subtitles for %s, skipping", video["url"])
                summary["videos_skipped"] += 1
                continue
            transcript = _subtitle_to_text(sub_file)

        if len(transcript) < _MIN_TRANSCRIPT_CHARS:
            logger.warning(
                "video learner: transcript too short (%d chars) for %s, skipping",
                len(transcript), video["url"],
            )
            summary["videos_skipped"] += 1
            continue

        try:
            tips = asyncio.run(_distill_transcript(video, transcript))
        except Exception as exc:
            logger.warning("video learner: LLM distillation failed for %s: %s", video["url"], exc)
            summary["videos_skipped"] += 1
            continue
        if not tips:
            logger.info("video learner: no concrete tips extracted from %s", video["url"])
            summary["videos_skipped"] += 1
            continue

        chunks = _tips_to_chunks(video, tips, langs, patch)
        new_chunks.extend(chunks)
        summary["videos_processed"] += 1
        seen[video["id"]] = {
            "url": video["url"],
            "title": video.get("title", ""),
            "processed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "chunk_count": len(chunks),
        }
        _save_seen(seen)
        logger.info(
            "video learner: %s -> %d tips (%d chunks)",
            video["url"], len(tips), len(chunks),
        )

    if new_chunks:
        # save_learned_file overwrites, so carry prior chunks forward
        prior = _load_prior_chunks()
        new_ids = {c.chunk_id for c in new_chunks}
        merged = [c for c in prior if c.chunk_id not in new_ids] + new_chunks
        save_learned_file(
            LEARNED_FILE_NAME,
            merged,
            meta={"videos_total": len(seen), "videos_this_run": summary["videos_processed"]},
        )
        summary["chunks_written"] = len(new_chunks)

    return summary
