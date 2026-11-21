"""
Metadata and SEO generator for n8n automation and social media posting (TikTok, Shorts, Reels).
Optimized for CIS audience (Russian language captions, titles, hashtags, and CTA).
Uses Qwen hook titles, detected CS2 entities, and viral retention triggers.
"""

import json
import os
import re
from typing import Dict, Any, Optional

from pipeline.analyzer_qwen import MAP_KEYWORDS_RU, NADE_KEYWORDS_RU, CALLOUT_KEYWORDS_RU


MAP_NAMES_RU = {
    "mirage": "Mirage",
    "inferno": "Inferno",
    "anubis": "Anubis",
    "ancient": "Ancient",
    "nuke": "Nuke",
    "dust2": "Dust 2",
    "vertigo": "Vertigo",
    "overpass": "Overpass",
    "train": "Train",
}

NADE_TITLE_RU = {
    "smoke": "Имбовый смок",
    "flash": "Слепящая флешка",
    "molly": "Огненный молотов",
    "he": "Осколочная граната",
    "incendiary": "Зажигательная граната",
    "decoy": "Фейк-граната",
}

NADE_HASHTAGS_RU = {
    "smoke": ["#смокикс2", "#cs2smokes"],
    "flash": ["#флешкикс2", "#cs2flash"],
    "molly": ["#молотовыкс2", "#cs2molly"],
    "he": ["#гранатыкс2"],
    "incendiary": ["#зажигалкакс2"],
    "decoy": ["#фейкикс2"],
}


def _detect_from_text_ru(text: str) -> Dict[str, str]:
    """Fallback entity detection in Russian text."""
    t = text.lower()
    ent = {"map": "", "nade": "", "callout": ""}
    for k, v in MAP_KEYWORDS_RU.items():
        if re.search(r"\b" + re.escape(k) + r"\b", t):
            ent["map"] = v
            break
    for k, v in NADE_KEYWORDS_RU.items():
        if re.search(r"\b" + re.escape(k) + r"\b", t):
            ent["nade"] = v
            break
    for k, v in CALLOUT_KEYWORDS_RU.items():
        if re.search(r"\b" + re.escape(k) + r"\b", t):
            ent["callout"] = v
            break
    return ent


def generate_clip_metadata(
    clip: Dict[str, Any],
    video_stem: str,
    output_json_path: str,
    config: dict,
    analyzer: Optional[Any] = None
) -> str:
    """
    Generates rich Russian JSON metadata for n8n / TikTok uploaders.
    Prioritizes Qwen hook title > entity template > Russian viral CTA.
    """
    clip_text = clip.get("text", "")
    entities = clip.get("entities") or _detect_from_text_ru(clip_text)

    detected_map = MAP_NAMES_RU.get(str(entities.get("map", "")).lower(), entities.get("map", "") or "CS2")
    detected_nade = NADE_TITLE_RU.get(str(entities.get("nade", "")).lower(), entities.get("nade", "") or "Фишка и раскид")

    # 1. Post Title: Prefer Qwen viral hook title
    hook_title = clip.get("hook_title", "")
    if hook_title:
        post_title = hook_title
    elif clip.get("title_ru"):
        post_title = clip["title_ru"]
    elif entities.get("map") or entities.get("nade"):
        callout = entities.get("callout", "")
        where = f" на {callout}" if callout else ""
        post_title = f"{detected_nade}{where} — {detected_map} CS2 🎯"
    else:
        post_title = f"{detected_nade} на {detected_map} в CS2 🎯"

    # 2. Viral CTA for CIS audience (байт на комменти про ранг, спори)
    promo_enabled = config.get("promo_midroll", {}).get("enabled", False) or config.get("banner", {}).get("enabled", False)
    if promo_enabled:
        cta_text = "🎁 Забирай бонус в шапке профиля! А в комментах пиши свой ранг в CS2 👇"
    else:
        cta_text = "🔥 Знал эту фишку или тиммейты руинят так катку?\n👇 Пиши в комменты свой ранг/эло в CS2, проверим сколько тут 10 LVL!"

    # 3. Target Hashtags for CIS TikTok / Shorts algorithms
    hashtags = [
        "#кс2", "#cs2", "#ксго", "#csgo",
        "#фишкикс2", "#раскидкикс2", "#тиктокснг",
        "#counterstrike2", "#cs2clips", "#cs2guide"
    ]
    nade_tag = NADE_HASHTAGS_RU.get(str(entities.get("nade", "")).lower(), [])
    if nade_tag:
        hashtags.insert(4, nade_tag[0])
    if detected_map and detected_map != "CS2":
        map_clean = detected_map.lower().replace(" ", "")
        hashtags.append(f"#кс2{map_clean}")
        hashtags.append(f"#{map_clean}")

    # 4. Description / Caption formatting in Russian
    caption_parts = [post_title]

    # Quote punchy line from speech if present
    hook_line = clip.get("hook_line", "")
    if hook_line and hook_line not in post_title and len(hook_line) > 10:
        caption_parts.append(f"💬 «{hook_line}»")

    caption_parts.append(cta_text)
    caption = "\n\n".join(caption_parts) + "\n\n" + " ".join(hashtags[:8])

    meta = {
        "clip_index": clip.get("index", 1),
        "source_video": video_stem,
        "title": post_title,
        "hook_title": hook_title,
        "viral_reason": clip.get("viral_reason", "CS2 вирусный момент"),
        "caption": caption,
        "hashtags": hashtags,
        "detected_map": detected_map,
        "detected_type": detected_nade,
        "start_time": round(float(clip.get("start", 0.0)), 3),
        "end_time": round(float(clip.get("end", 0.0)), 3),
        "duration_seconds": round(float(clip.get("duration", 0.0)), 2),
        "target_audience": "CIS (RU)",
    }

    if clip.get("score") is not None:
        meta["retention_score"] = clip["score"]
    if clip.get("_speed"):
        meta["render_speed"] = clip["_speed"]
    if clip.get("_uniq"):
        meta["uniquization"] = clip["_uniq"]

    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return output_json_path
