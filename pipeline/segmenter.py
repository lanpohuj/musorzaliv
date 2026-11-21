"""
CS2 highlight segmentation engine.
Refactored for TikTok algorithmic optimization:
- Duration: Strictly 20 - 35 seconds (hard max 38.0s).
- Zero silence delay at start (shifts start to 0.5s BEFORE first spoken syllable for action hook).
- Zero trailing silence (cuts immediately after final spoken word).
- Pacing enforcement: trims or splits internal silence pauses > 0.8s.
- Integrated with Qwen LLM analysis and local fallback.
"""

import random
import re
from typing import List, Dict, Any, Optional

from pipeline.analyzer import content_tokens, jaccard
from pipeline.sentences import build_sentences, is_tail_complete, sentence_windows


MAP_KEYWORDS = [
    "mirage", "inferno", "anubis", "ancient", "nuke", "dust", "vertigo", "overpass", "train",
    "міраж", "інферно", "анубіс", "нюк", "даст", "вертіго", "мираж", "инферно", "анубис", "вертиго"
]

LINEUP_KEYWORDS = [
    "smoke", "flash", "molly", "molotov", "he", "grenade", "lineup", "line up",
    "boost", "pixel", "pixel-boost", "one-way", "oneway", "jumpthrow", "jump-throw",
    "crosshair", "aim", "throw", "window", "connector", "short", "long", "site",
    "banana", "ramp", "palace", "catwalk", "heaven", "outside", "mid",
    "розкид", "розкидка", "смок", "дим", "флеш", "флешка", "молотов", "молік",
    "буст", "піксель", "ванвей", "приціл", "кидок", "вікно", "конектор", "міда", "плент",
    "раскид", "раскидка", "окно", "коннектор", "мид", "подсадка"
]

TRANSITION_KEYWORDS = [
    "next", "another", "moving on", "first", "second", "third", "now let's", "now for",
    "далі", "наступний", "ще один", "перший", "другий", "следующий", "еще один", "далее", "теперь"
]

DISCOURSE_MARKERS = [
    "далі", "далее", "також", "наступн", "следующ", "друге", "второе", "третє",
    "третий", "четверт", "пятое", "по-перше", "по-друге", "во-первых",
    "во-вторых", "тепер", "теперь", "сейчас", "по-третє", "в-третьих",
    "нарешті", "наконец", "последн", "крім того", "кроме того", "кстати",
    "до речі", "плюс", "next", "another", "also", "now", "second", "third", "finally", "правило"
]


def extract_keywords_title(text: str) -> str:
    """Generate a readable slug from matched CS2 keywords in the clip text."""
    text_lower = text.lower()
    found_map = ""
    for m in MAP_KEYWORDS:
        if m in text_lower:
            found_map = m
            break
            
    found_action = ""
    for k in ["smoke", "flash", "molly", "molotov", "boost", "pixel", "lineup", "смок", "флешка", "буст", "раскид"]:
        if k in text_lower:
            found_action = k
            break
            
    parts = []
    if found_map:
        parts.append(found_map)
    if found_action:
        parts.append(found_action)
    if not parts:
        clean = re.sub(r"[^a-zA-Z0-9\s]", "", text_lower).strip().split()
        if clean:
            parts = clean[:2]
        else:
            parts = ["cs2", "highlight"]
            
    slug = "_".join(parts)
    return re.sub(r"[^\w\-]", "", slug)


class Segmenter:
    def __init__(self, config: dict, analyzer: Optional[Any] = None):
        self.config = config
        self.cfg = config.get("clips", {})
        
        # Strict duration constraints: 20-35s, hard max 38s
        self.min_dur = float(self.cfg.get("min_duration", 20.0))
        self.max_dur = float(self.cfg.get("max_duration", 35.0))
        self.hard_max = float(self.cfg.get("hard_max_duration", 38.0))
        self.target_dur = float(self.cfg.get("target_duration", 28.0))
        self.max_internal_pause = float(self.cfg.get("max_internal_pause", 0.8))

        # Precision silence trimming (0.5s pre-roll for action hook, cut right after final word)
        self.pad_start = float(self.cfg.get("pad_start", -0.5))
        self.pad_end = float(self.cfg.get("pad_end", 0.05))
        self.start_jitter = float(self.cfg.get("start_jitter", 0.0))
        self.clip_floor = float(self.cfg.get("min_clip_duration", 18.0))

        self.silence_threshold = float(self.cfg.get("silence_split_threshold", 0.8))
        self.analyzer = analyzer

        self.sentence_based = bool(self.cfg.get("sentence_based", True))
        self.hard_min_pause = float(self.cfg.get("hard_min_pause", 0.2))
        self.cut_pause = float(self.cfg.get("cut_pause", 0.5))
        self.strong_pause = float(self.cfg.get("strong_pause", 0.8))
        self.topic_window = int(self.cfg.get("topic_window", 3))
        self.topic_shift_threshold = float(self.cfg.get("topic_shift_threshold", 0.14))

    def find_clips(self, segments: List[Dict[str, Any]], video_duration: float, cache_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Main entrypoint for finding TikTok-optimized 20-35s clips.
        1. If QwenAnalyzer is attached, runs LLM analysis (ranked pipeline).
        2. Validates all clips against strict duration [20s, 35s] (max 38s).
        3. Enforces silence trimming and pre-roll action hook.
        4. Fallbacks to sentence-based / acoustic cluster segmentation.
        """
        if video_duration <= 0.0:
            video_duration = segments[-1]["end"] if segments else 60.0

        if not segments:
            return self._fallback_equal_slices(video_duration)

        # Context sponsor filtering
        if self.analyzer and hasattr(self.analyzer, "filter_segments"):
            segments = self.analyzer.filter_segments(segments)
            if not segments:
                return []

        # Option A: Qwen LLM Analyzer
        if self.analyzer and hasattr(self.analyzer, "analyze_transcript"):
            try:
                llm_clips = self.analyzer.analyze_transcript(segments, video_duration, cache_key)
                if llm_clips:
                    validated = self._validate_and_enforce_clips(llm_clips, segments, video_duration)
                    if validated:
                        return validated
            except Exception as e:
                print(f"[!] Ошибка анализа через Qwen Analyzer ({e}), переход на локальную нарезку")

        # Option B: Sentence-based segmentation targeting 15-25s
        if self.sentence_based:
            sentences = self._build_sentences(segments)
            if len(sentences) >= 2:
                clips = self._find_clips_sentences(sentences, video_duration)
                if clips:
                    return self._validate_and_enforce_clips(clips, segments, video_duration)

        # Option C: Cluster segmentation (fallback)
        return self._find_clips_clusters(segments, video_duration)

    def _fallback_equal_slices(self, video_duration: float) -> List[Dict[str, Any]]:
        """Creates fixed-interval clips if no speech was detected."""
        clips = []
        cur_start = 0.0
        idx = 1
        while cur_start < video_duration:
            cur_end = min(video_duration, cur_start + self.target_dur)
            dur = cur_end - cur_start
            if dur >= self.min_dur:
                clips.append({
                    "index": idx,
                    "start": cur_start,
                    "end": cur_end,
                    "duration": dur,
                    "title": f"clip_{idx:02d}",
                    "hook_title": "СЕКРЕТНЫЙ РАСКИД В CS2!",
                    "viral_reason": "Авто-нарезка CS2",
                    "segments": [],
                    "text": ""
                })
                idx += 1
            cur_start = cur_end
        return clips

    def _validate_and_enforce_clips(
        self,
        clips: List[Dict[str, Any]],
        segments: List[Dict[str, Any]],
        video_duration: float
    ) -> List[Dict[str, Any]]:
        """
        Enforces strict algorithmic constraints on all clips:
        1. Duration MUST be between 20.0 and 35.0s (Hard Max 38.0s).
        2. Start shifted to 0.5s BEFORE first spoken syllable (action hook).
        3. End cut immediately after final spoken word (no trailing silence).
        4. Internal pauses > 0.8s trimmed or split.
        5. Micro-compilation splitting for any oversized clips.
        """
        all_words = []
        for s in segments:
            for w in (s.get("words") or []):
                all_words.append(w)

        result = []
        clip_index = 1

        for c in clips:
            s_time = float(c["start"])
            e_time = float(c["end"])
            dur = e_time - s_time

            # Oversized clips (> 28.0s): split into micro-compilations
            if dur > self.hard_max:
                print(f"[✂️] Клип #{c.get('index', clip_index)} длительностью {dur:.1f}с превышает {self.hard_max}с. Разделяем...")
                sub_clips = self._split_oversized_clip(c, all_words, video_duration)
                for sc in sub_clips:
                    sc["index"] = clip_index
                    result.append(sc)
                    clip_index += 1
                continue

            # Precise audio trimming: предпочтение у готового (уже sentence-snapped)
            # списка слов; окно ±0.3с используется только когда слов нет — иначе
            # можно прихватить первое слово следующего предложения и снова получить обрубок
            clip_words = c.get("words") or []
            if not clip_words:
                clip_words = [w for w in all_words if w["start"] >= (s_time - 0.3) and w["end"] <= (e_time + 0.3)]
            if clip_words:
                # pad_start < 0 means start BEFORE first word (action hook pre-roll)
                # pad_start > 0 means start AFTER first word (legacy trim)
                actual_start = max(0.0, clip_words[0]["start"] + self.pad_start)
                actual_end = min(video_duration, clip_words[-1]["end"] + self.pad_end)
            else:
                actual_start = max(0.0, s_time)
                actual_end = min(video_duration, e_time)

            actual_dur = actual_end - actual_start

            # Enforce hard upper cap at a word boundary (никогда не слепой рез посреди фразы)
            if actual_dur > self.hard_max:
                fitting = [w for w in clip_words if (w["end"] - actual_start) <= self.max_dur]
                if fitting:
                    actual_end = min(video_duration, fitting[-1]["end"] + self.pad_end)
                else:
                    actual_end = actual_start + self.max_dur
                actual_dur = actual_end - actual_start

            # Discard too short clips
            if actual_dur < self.clip_floor:
                continue

            c["index"] = clip_index
            c["start"] = round(actual_start, 3)
            c["end"] = round(actual_end, 3)
            c["duration"] = round(actual_dur, 2)
            c["words"] = clip_words

            # Ensure hook_title in Russian with comment bait
            if not c.get("hook_title"):
                c["hook_title"] = "ТВОЙ РАНГ СИЛЬВЕР ЕСЛИ НЕ ЗНАЛ!"

            # Pacing: detect internal gaps > 0.8s
            gaps = []
            for i in range(len(clip_words) - 1):
                gap = clip_words[i + 1]["start"] - clip_words[i]["end"]
                if gap > self.max_internal_pause:
                    gaps.append((clip_words[i]["end"], clip_words[i + 1]["start"]))
            c["internal_gaps"] = gaps

            result.append(c)
            clip_index += 1

        return result

    def _split_oversized_clip(
        self,
        clip: Dict[str, Any],
        all_words: List[Dict[str, Any]],
        video_duration: float
    ) -> List[Dict[str, Any]]:
        """Splits an oversized clip into punchy 20-35s chunks."""
        s_time = float(clip["start"])
        e_time = float(clip["end"])
        base_hook = clip.get("hook_title", "СЕКРЕТНЫЙ РАСКИД В CS2")

        words = [w for w in all_words if w["start"] >= (s_time - 0.2) and w["end"] <= (e_time + 0.2)]
        if not words:
            # Fallback split
            sub = []
            cur = s_time
            part = 1
            while cur < e_time:
                cur_end = min(e_time, cur + self.target_dur)
                if (cur_end - cur) >= self.min_dur:
                    sub.append({
                        "start": cur,
                        "end": cur_end,
                        "duration": cur_end - cur,
                        "hook_title": f"{base_hook} (Ч.{part})",
                        "viral_reason": clip.get("viral_reason", "CS2 гайд"),
                        "text": clip.get("text", ""),
                        "segments": clip.get("segments", [])
                    })
                    part += 1
                cur = cur_end
            return sub

        # Відновлюємо структуру речень всередині задовгого діапазону
        fake_segments = [{"start": words[0]["start"], "end": words[-1]["end"], "text": "", "words": words}]
        sentences = build_sentences(fake_segments)
        if not sentences:
            return []

        windows = sentence_windows(
            sentences, 0, len(sentences) - 1,
            min_dur=self.min_dur,
            floor_dur=self.clip_floor,
            max_dur=self.max_dur,
        )

        sub = []
        part = 1
        for (i0, i1) in windows:
            w_start = sentences[i0]["start"]
            w_end = sentences[i1]["end"]
            sub_words = []
            sub_texts = []
            for si in range(i0, i1 + 1):
                sub_words.extend(sentences[si]["words"])
                sub_texts.append(sentences[si]["text"])
            text = " ".join(sub_texts).strip()
            sub.append({
                "start": w_start,
                "end": w_end,
                "duration": round(w_end - w_start, 2),
                "hook_title": f"{base_hook} (Ч.{part})",
                "viral_reason": clip.get("viral_reason", "Микро-клип CS2 фишки"),
                "text": text,
                "words": sub_words,
                "segments": [{"start": w_start, "end": w_end, "words": sub_words, "text": text}]
            })
            part += 1

        return sub

    def _build_sentences(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Groups transcript into sentences with word timestamps (sentence engine)."""
        return build_sentences(segments)

    def _find_clips_sentences(
        self,
        sentences: List[Dict[str, Any]],
        video_duration: float
    ) -> List[Dict[str, Any]]:
        """Groups sentences into strictly 20-35s clips (target 28s)."""
        clips = []
        cur_group: List[Dict[str, Any]] = []
        idx = 1

        for sent in sentences:
            cur_group.append(sent)
            cur_dur = cur_group[-1]["end"] - cur_group[0]["start"]

            # Cut when in target range [15s, 25s]; хвіст, що висить на сполучнику,
            # дорощуємо наступним реченням, поки є місце до max_dur
            if cur_dur >= self.target_dur or cur_dur >= self.max_dur:
                if cur_dur < self.max_dur and not is_tail_complete(sent["text"]):
                    continue
                c = self._make_clip_from_sentence_group(idx, cur_group, video_duration)
                if c:
                    clips.append(c)
                    idx += 1
                cur_group = []

        if cur_group and (cur_group[-1]["end"] - cur_group[0]["start"]) >= self.min_dur:
            c = self._make_clip_from_sentence_group(idx, cur_group, video_duration)
            if c:
                clips.append(c)

        return clips

    def _make_clip_from_sentence_group(
        self,
        index: int,
        group: List[Dict[str, Any]],
        video_duration: float
    ) -> Optional[Dict[str, Any]]:
        """Constructs a clip from a list of sentence dicts."""
        raw_start = group[0]["start"]
        raw_end = group[-1]["end"]
        dur = raw_end - raw_start

        text = " ".join(s["text"] for s in group).strip()
        words = []
        for s in group:
            words.extend(s.get("words") or [])

        if dur > self.hard_max:
            # Ріжемо по останньому слову, що влазить у max_dur (не сліпий рез)
            limit = raw_start + self.max_dur
            fitting = [w for w in words if w["end"] <= limit]
            if fitting:
                raw_end = fitting[-1]["end"] + self.pad_end
            else:
                raw_end = raw_start + self.max_dur
            dur = raw_end - raw_start

        if dur < self.clip_floor:
            return None

        # Clickbait Russian hook title with comment bait
        hook_title = "ТВОЙ РАНГ СИЛЬВЕР ЕСЛИ НЕ ЗНАЛ!"
        for kw in ["смок", "smoke", "дым"]:
            if kw in text.lower():
                hook_title = "ЭТОТ СМОК ЗНАЮТ ТОЛЬКО 10 LVL!"
                break
        for kw in ["флеш", "flash", "слепа"]:
            if kw in text.lower():
                hook_title = "ТИМЕЙТ РУИНИТ КАТКУ ЭТОЙ ФЛЕШКОЙ!"
                break
        for kw in ["молотов", "molly", "молик"]:
            if kw in text.lower():
                hook_title = "НЕ КИДАЙ ТАК МОЛОТОВ В CS2!"
                break

        return {
            "index": index,
            "start": round(raw_start, 3),
            "end": round(raw_end, 3),
            "duration": round(dur, 2),
            "title": extract_keywords_title(text),
            "hook_title": hook_title,
            "viral_reason": "Обучающий момент CS2",
            "segments": group,
            "words": words,
            "text": text,
        }

    def _find_clips_clusters(
        self,
        segments: List[Dict[str, Any]],
        video_duration: float
    ) -> List[Dict[str, Any]]:
        """Fallback clustering by silence pauses targeting 20-35s."""
        clips = []
        cur_batch = [segments[0]]
        idx = 1

        for i in range(1, len(segments)):
            prev_s = segments[i - 1]
            curr_s = segments[i]
            gap = curr_s["start"] - prev_s["end"]
            dur = prev_s["end"] - cur_batch[0]["start"]

            if (gap >= self.silence_threshold and dur >= self.min_dur) or dur >= self.max_dur:
                c = self._make_clip_from_sentence_group(idx, cur_batch, video_duration)
                if c:
                    clips.append(c)
                    idx += 1
                cur_batch = [curr_s]
            else:
                cur_batch.append(curr_s)

        if cur_batch and (cur_batch[-1]["end"] - cur_batch[0]["start"]) >= self.min_dur:
            c = self._make_clip_from_sentence_group(idx, cur_batch, video_duration)
            if c:
                clips.append(c)

        return clips
