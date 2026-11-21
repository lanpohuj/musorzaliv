"""
Subtitle generation module (ASS format) for TikTok/Shorts vertical video (9:16).
Features:
- Hook Title Overlay (0.0s - 3.5s) in top-middle third: bold uppercase, bright yellow/white, thick black stroke.
- Dynamic Subtitles (2-3 words per line) centered below crosshair/hook with active-word highlight animation.
- Handles timing scaling with render speed and internal pause trimming.
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple


def format_ass_time(seconds: float) -> str:
    """Format seconds into ASS timestamp H:MM:SS.cs"""
    seconds = max(0.0, seconds)
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        cs = 99
    return f"{hrs}:{mins:02d}:{secs:02d}.{cs:02d}"


def wrap_hook_title(title: str, max_words_per_line: int = 3) -> str:
    """Wraps hook title into 1-2 punchy lines with ASS line break \\N."""
    if not title:
        return ""
    words = title.strip().split()
    if len(words) <= max_words_per_line:
        return " ".join(words)
    
    mid = (len(words) + 1) // 2
    line1 = " ".join(words[:mid])
    line2 = " ".join(words[mid:])
    return f"{line1}\\N{line2}"


class SubtitleGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.cfg = config.get("subtitles", {})
        self.enabled = bool(self.cfg.get("enabled", True))
        self.font_name = self.cfg.get("font_name", "DejaVu Sans")
        self.font_size = int(self.cfg.get("font_size", 54))
        self.words_per_group = int(self.cfg.get("words_per_group", 2))  # 2-3 words max
        self.bottom_margin = int(self.cfg.get("bottom_margin", 680))    # Centered below crosshair
        self.default_speed = float(config.get("video", {}).get("speed", 1.04))

        # Hook Title Overlay configuration (0.0s - 3.5s in top-middle third)
        hook_cfg = self.cfg.get("hook_overlay", {})
        self.hook_enabled = bool(hook_cfg.get("enabled", True))
        self.hook_duration = float(hook_cfg.get("duration", 3.5))
        self.hook_font_size = int(hook_cfg.get("font_size", 66))
        self.hook_color = hook_cfg.get("font_color", "&H0000FFFF")      # Yellow (&H0000FFFF)
        self.hook_outline_color = hook_cfg.get("outline_color", "&H00000000") # Black
        self.hook_outline_width = int(hook_cfg.get("outline_width", 8))
        self.hook_top_margin = int(hook_cfg.get("top_margin", 360))     # Top-middle third

        # Dynamic subtitles styling
        self.sub_color = self.cfg.get("font_color", "&H00FFFFFF")       # White
        self.highlight_color = self.cfg.get("active_word_color", "&H0000FFFF") # Yellow highlight
        self.highlight_enabled = bool(self.cfg.get("active_word_highlight", True))

    def _get_words_for_clip(self, clip: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all words belonging to this clip in relative time scale."""
        clip_start = clip["start"]
        clip_end = clip["end"]
        clip_words = []

        # Check if clip has pre-extracted words
        if clip.get("words"):
            for w in clip["words"]:
                if w["start"] >= (clip_start - 0.2) and w["end"] <= (clip_end + 0.3):
                    clip_words.append(w)
        else:
            for seg in clip.get("segments", []):
                if seg.get("words"):
                    for w in seg["words"]:
                        if w["start"] >= (clip_start - 0.2) and w["end"] <= (clip_end + 0.3):
                            clip_words.append(w)
                elif seg.get("text"):
                    words = seg["text"].split()
                    if words:
                        dur = (seg["end"] - seg["start"]) / len(words)
                        for idx, w in enumerate(words):
                            ws = seg["start"] + idx * dur
                            we = ws + dur
                            if ws >= (clip_start - 0.2) and we <= (clip_end + 0.3):
                                clip_words.append({"word": w, "start": ws, "end": we})

        # Convert to relative timestamps from clip_start
        rel_words = []
        for w in clip_words:
            rw_start = max(0.0, w["start"] - clip_start)
            rw_end = max(rw_start + 0.1, w["end"] - clip_start)
            clean_word = re.sub(r"[^\w\s\?!,\.-]", "", w["word"]).strip().upper()
            if clean_word:
                rel_words.append({
                    "word": clean_word,
                    "start": rw_start,
                    "end": rw_end
                })

        return rel_words

    def _build_highlight_dialogue_events(
        self,
        words: List[Dict[str, Any]],
        speed: float,
        time_offset: float = 0.0
    ) -> List[str]:
        """
        Builds ASS dialogue events with active-word highlight animation.
        Groups words into 2-3 words per phrase.
        """
        events = []
        if not words:
            return events

        group_size = max(1, min(3, self.words_per_group))

        for i in range(0, len(words), group_size):
            group = words[i:i + group_size]
            g_start = group[0]["start"]
            g_end = max(g_start + 0.35, group[-1]["end"])

            # If active-word highlight is enabled, generate sub-frames for each active word
            if self.highlight_enabled and len(group) > 1:
                for k, active_w in enumerate(group):
                    w_s = active_w["start"]
                    w_e = active_w["end"]
                    # Ensure non-zero duration
                    if w_e <= w_s:
                        w_e = w_s + 0.2

                    # Frame line with active word highlighted
                    line_parts = []
                    for j, other_w in enumerate(group):
                        w_text = other_w["word"].replace("\\", "")
                        if j == k:
                            line_parts.append(f"{{\\c{self.highlight_color}}}{w_text}{{\\c{self.sub_color}}}")
                        else:
                            line_parts.append(w_text)

                    ass_text = " ".join(line_parts)
                    s_str = format_ass_time(max(0.0, (w_s - time_offset) / speed))
                    e_str = format_ass_time(max(0.2 / speed, (w_e - time_offset) / speed))
                    events.append(f"Dialogue: 0,{s_str},{e_str},Subtitle,,0,0,0,,{ass_text}")
            else:
                # Static group line
                full_text = " ".join(w["word"].replace("\\", "") for w in group)
                s_str = format_ass_time(max(0.0, (g_start - time_offset) / speed))
                e_str = format_ass_time(max(0.3 / speed, (g_end - time_offset) / speed))
                events.append(f"Dialogue: 0,{s_str},{e_str},Subtitle,,0,0,0,,{full_text}")

        return events

    def generate_ass(
        self,
        clip: Dict[str, Any],
        output_path: str,
        speed: Optional[float] = None
    ) -> str:
        """
        Generates ASS subtitle file with:
        1. Top-middle third Hook Title Overlay (0.0s - 3.5s)
        2. Dynamic animated subtitles centered below crosshair
        """
        if not self.enabled:
            return ""

        sp = float(speed) if speed else self.default_speed
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: HookTitle,{self.font_name},{self.hook_font_size},{self.hook_color},&H000000FF,{self.hook_outline_color},&H80000000,-1,0,0,0,100,100,1,0,1,{self.hook_outline_width},3,8,40,40,{self.hook_top_margin},1
Style: Subtitle,{self.font_name},{self.font_size},{self.sub_color},&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,6,2,2,40,40,{self.bottom_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        events = []

        # 1. Render Hook Title Overlay (0.0s to 3.5s)
        hook_title = clip.get("hook_title", "")
        if self.hook_enabled and hook_title:
            hook_wrapped = wrap_hook_title(hook_title)
            hook_duration = min(self.hook_duration, float(clip.get("duration", 20.0)))
            s_str = format_ass_time(0.0)
            e_str = format_ass_time(hook_duration / sp)
            events.append(f"Dialogue: 1,{s_str},{e_str},HookTitle,,0,0,0,,{hook_wrapped}")

        # 2. Render Dynamic Subtitles
        words = self._get_words_for_clip(clip)
        sub_events = self._build_highlight_dialogue_events(words, sp)
        events.extend(sub_events)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header + "\n".join(events) + "\n")

        return output_path

    def generate_ass_split(
        self,
        clip: Dict[str, Any],
        mid_time: float,
        out_part1: str,
        out_part2: str,
        speed: Optional[float] = None
    ) -> Tuple[str, str]:
        """Split subtitles into Part 1 and Part 2 around mid-roll promo."""
        if not self.enabled:
            return "", ""

        sp = float(speed) if speed else self.default_speed
        words = self._get_words_for_clip(clip)
        words_p1 = [w for w in words if w["end"] <= (mid_time + 0.1)]
        words_p2 = [w for w in words if w["start"] >= (mid_time - 0.1)]

        # Header template
        def make_header() -> str:
            return f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: HookTitle,{self.font_name},{self.hook_font_size},{self.hook_color},&H000000FF,{self.hook_outline_color},&H80000000,-1,0,0,0,100,100,1,0,1,{self.hook_outline_width},3,8,40,40,{self.hook_top_margin},1
Style: Subtitle,{self.font_name},{self.font_size},{self.sub_color},&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,6,2,2,40,40,{self.bottom_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        # Part 1: includes hook title + words before mid-roll
        p1_events = []
        hook_title = clip.get("hook_title", "")
        if self.hook_enabled and hook_title:
            hook_wrapped = wrap_hook_title(hook_title)
            hook_duration = min(self.hook_duration, mid_time)
            s_str = format_ass_time(0.0)
            e_str = format_ass_time(hook_duration / sp)
            p1_events.append(f"Dialogue: 1,{s_str},{e_str},HookTitle,,0,0,0,,{hook_wrapped}")
        p1_events.extend(self._build_highlight_dialogue_events(words_p1, sp))

        with open(out_part1, "w", encoding="utf-8") as f:
            f.write(make_header() + "\n".join(p1_events) + "\n")

        # Part 2: words after mid-roll with offset
        p2_events = self._build_highlight_dialogue_events(words_p2, sp, time_offset=mid_time)
        with open(out_part2, "w", encoding="utf-8") as f:
            f.write(make_header() + "\n".join(p2_events) + "\n")

        return out_part1, out_part2
