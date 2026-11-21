"""
Qwen LLM Analyzer Module for TikTok CS2 Clipper Bot.
Integrates local Ollama Qwen model for viral moment detection,
micro-compilation splitting, and Russian clickbait hook generation.
"""

import json
import re
import urllib.request
from typing import List, Dict, Any, Optional, Tuple

from pipeline.sentences import (
    build_sentences,
    sentence_windows,
    slice_words_and_text,
    snap_clip_window,
)
from pipeline.units import (
    complete_myth_window,
    looks_like_myth_intro,
    looks_like_verdict,
    merge_windows_with_units,
    resolve_sent_window,
)
from pipeline import labeler as interest_labeler
from pipeline import ranker as clip_ranker


# Russian CS2 entity dictionaries for fallback and enrichment
MAP_KEYWORDS_RU = {
    "mirage": "Mirage", "мираж": "Mirage", "мираже": "Mirage", "миража": "Mirage",
    "inferno": "Inferno", "инферно": "Inferno",
    "anubis": "Anubis", "анубис": "Anubis", "анубисе": "Anubis",
    "ancient": "Ancient", "эншент": "Ancient", "эйншент": "Ancient",
    "nuke": "Nuke", "нюк": "Nuke", "нюке": "Nuke",
    "dust": "Dust 2", "даст": "Dust 2", "дасте": "Dust 2", "dust2": "Dust 2",
    "vertigo": "Vertigo", "вертиго": "Vertigo",
    "overpass": "Overpass", "оверпасс": "Overpass", "овер": "Overpass",
    "train": "Train", "трейн": "Train", "трейне": "Train",
}

NADE_KEYWORDS_RU = {
    "smoke": "Смок", "смок": "Смок", "смока": "Смок", "смоки": "Смоки", "дым": "Смок",
    "flash": "Флешка", "флеш": "Флешка", "флешка": "Флешка", "флешку": "Флешка", "слепа": "Флешка",
    "molly": "Молотов", "molotov": "Молотов", "молотов": "Молотов", "молик": "Молотов", "огонь": "Молотов",
    "he": "Хаешка", "граната": "Граната", "хае": "Хаешка", "грена": "Граната",
    "incendiary": "Зажигательная", "зажигалка": "Зажигательная",
    "decoy": "Фейк", "ложная": "Фейк",
}

CALLOUT_KEYWORDS_RU = {
    "mid": "мид", "мида": "мид", "миде": "мид",
    "window": "окно", "окна": "окно", "окне": "окно",
    "short": "шорт", "шорта": "шорт", "шорте": "шорт",
    "long": "лонг", "лонга": "лонг", "лонге": "лонг",
    "connector": "коннектор", "коннектора": "коннектор", "коннекторе": "коннектор",
    "banana": "банан", "банана": "банан", "банане": "банан",
    "palace": "паласы", "паласах": "паласы", "палас": "паласы",
    "ramp": "рампа", "рампы": "рампа", "рампе": "рампа",
    "catwalk": "кэт", "кэтволк": "кэт",
    "heaven": "хэвен", "хэвене": "хэвен",
    "hell": "хелл", "андер": "андер", "подземка": "андер",
    "site": "плент", "плента": "плент", "пленте": "плент", "плант": "плент",
    "a site": "А плент", "b site": "Б плент", "а плент": "А плент", "б плент": "Б плент",
    "outside": "улица", "улице": "улица",
    "boost": "буст", "подсадка": "буст", "подсадке": "буст",
    "pixel": "пиксель", "пиксель-буст": "пиксель-буст",
    "van": "машина", "фургон": "машина",
    "jumpthrow": "джамптроу", "джамп": "джамптроу",
}

HOOK_TRIGGERS_RU = [
    r"\bимб", r"\bсекрет", r"\bтоп[- ]?\d", r"\bлют", r"\bжестк", r"\bлучш",
    r"\bникто не\b", r"\bмало кто\b", r"\bпрост", r"\bлегк", r"\bпро\b", r"\bпро-",
    r"\bчитер", r"\bфишк", r"\bфич", r"\bлайфхак", r"\bбаг", r"\bраскид",
    r"\bзапомни", r"\bникогда", r"\bне делай", r"\bошибк",
]


def parse_timestamp_seconds(val: Any) -> float:
    """Converts '00:01:23.500', '01:23.500', or numeric float/int to seconds."""
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str):
        return 0.0
    val = val.strip().replace(",", ".")
    if ":" in val:
        parts = val.split(":")
        try:
            if len(parts) == 3:
                h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
                return h * 3600.0 + m * 60.0 + s
            elif len(parts) == 2:
                m, s = float(parts[0]), float(parts[1])
                return m * 60.0 + s
        except ValueError:
            return 0.0
    try:
        return float(val)
    except ValueError:
        return 0.0


def format_timestamp(seconds: float) -> str:
    """Formats float seconds into 'HH:MM:SS.mmm'."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


class QwenAnalyzer:
    """
    Qwen LLM Analyzer for TikTok CS2 Clips.
    Connects to local Ollama (e.g. qwen2.5:3b) to identify viral 20-35s moments.
    """

    def __init__(self, config: dict):
        self.config = config
        cfg = config.get("analysis", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.use_llm = bool(cfg.get("use_llm", True))
        self.llm_url = cfg.get("llm_url", "http://localhost:11434")
        self.llm_model = cfg.get("llm_model", "qwen2.5:3b")
        self.llm_timeout = float(cfg.get("llm_timeout", 180.0))
        self.language = cfg.get("language", "ru")

        # Duration constraints
        clips_cfg = config.get("clips", {})
        self.min_duration = float(clips_cfg.get("min_duration", 20.0))
        self.max_duration = float(clips_cfg.get("max_duration", 35.0))
        self.hard_max = float(clips_cfg.get("hard_max_duration", 38.0))
        self.max_internal_pause = float(clips_cfg.get("max_internal_pause", 0.8))
        self.pad_start = float(clips_cfg.get("pad_start", -0.5))
        self.pad_end = float(clips_cfg.get("pad_end", 0.05))

        # Sentence snapping: кліпи ріжуться за межами речень Whisper, а не за сирими таймкодами LLM
        self.sentence_snap = bool(clips_cfg.get("sentence_snap", True))
        self.snap_max_extend = float(clips_cfg.get("snap_max_extend", 8.0))
        self.sentence_soft_pause = float(clips_cfg.get("sentence_soft_pause", 0.6))
        self.clip_floor = float(clips_cfg.get("min_clip_duration", 18.0))

        # WOW-CUTTER: конвеєр «розмітка → кандидати → скоринг → реранк»
        rank_cfg = config.get("ranking", {})
        self.ranking_enabled = bool(rank_cfg.get("enabled", True))
        self.ranking_top_n = int(rank_cfg.get("top_n", 8))
        self.ranking_rerank_top = int(rank_cfg.get("rerank_top", 20))
        self.ranking_min_score = float(rank_cfg.get("min_total_score", 35.0))
        self.ranking_weights = rank_cfg.get("weights") or {}
        self.preferred_model = str(rank_cfg.get("llm_model", "qwen2.5:7b"))
        self.auto_pull_model = bool(rank_cfg.get("auto_pull", True))
        self._pull_tried = False

        self._llm_checked = False
        self._llm_ok = False

    def is_available(self) -> bool:
        """
        Перевіряє Ollama і вибирає модель: спершу ranking.llm_model (qwen2.5:7b),
        за відсутності — один раз пробує її стягнути (ollama pull), далі відкат
        на analysis.llm_model (3b), потім на будь-яку наявну.
        """
        if not self.enabled or not self.use_llm:
            return False
        if self._llm_checked:
            return self._llm_ok
        self._llm_checked = True

        import shutil as _sh
        import subprocess as _sp

        def _models():
            try:
                req = urllib.request.Request(
                    f"{self.llm_url}/api/tags",
                    method="GET",
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return [m.get("name", "") for m in data.get("models", [])]
            except Exception:
                return None

        models = _models()
        if models is None:
            self._llm_ok = False
            return False

        preferred_base = self.preferred_model.split(":")[0]
        fallback_base = self.llm_model.split(":")[0]

        # 1) Преферована модель (7b)
        if any(m == self.preferred_model or m.startswith(preferred_base) for m in models):
            self.llm_model = next((m for m in models if m == self.preferred_model or m.startswith(preferred_base)), self.preferred_model)
            self._llm_ok = True
            print(f"[🧠] Qwen-ранкер: використовую {self.llm_model}")
            return True

        # 2) Авто-пул преференої — У ФОНІ: пайплайн не чекає 4.7ГБ завантаження,
        #    цей запуск іде на fallback-моделі, наступні — вже на 7b
        if self.auto_pull_model and not self._pull_tried and _sh.which("ollama"):
            self._pull_tried = True

            def _pull_background():
                try:
                    print(f"[🧠] Стягую {self.preferred_model} у фоні — цей запуск на {fallback_base}, наступні на 7b...")
                    proc = _sp.Popen(
                        ["ollama", "pull", self.preferred_model],
                        stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True,
                    )
                    last_logged = 0
                    for line in iter(proc.stdout.readline, ""):
                        line = line.strip()
                        # ollama пише прогрес типу " pulling 47%|█████ | 2.2/4.7 GB"
                        m = re.search(r"(\d+)%", line)
                        if m and int(m.group(1)) >= last_logged + 25:
                            last_logged = int(m.group(1))
                            print(f"[🧠] ollama pull {self.preferred_model}: {last_logged}%")
                    proc.wait(timeout=1800)
                    models = _models() or []
                    if proc.returncode == 0 and any(m == self.preferred_model or m.startswith(preferred_base) for m in models):
                        print(f"[✓] Модель {self.preferred_model} завантажена — наступні запуски ранжуватимуть на ній")
                    else:
                        print(f"[!] ollama pull не вдався (код {proc.returncode}) — лишаємось на {fallback_base}")
                except Exception as e:
                    print(f"[!] ollama pull не вдався ({e}) — лишаємось на {fallback_base}")

            import threading as _threading
            _threading.Thread(target=_pull_background, daemon=True).start()

        # 3) Відкат: analysis.llm_model (3b) або будь-яка наявна
        if any(m.startswith(fallback_base) for m in models):
            self._llm_ok = True
            print(f"[🧠] Qwen-ранкер: {self.preferred_model} недоступна, використовую {self.llm_model}")
            return True
        if models:
            self.llm_model = models[0]
            self._llm_ok = True
            return True

        self._llm_ok = False
        return False

    def _call_qwen(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Make a JSON-formatted request to local Ollama."""
        try:
            payload = json.dumps({
                "model": self.llm_model,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.5,
                    "num_predict": 2048,
                    "top_p": 0.9,
                }
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self.llm_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=self.llm_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return (data.get("response") or "").strip()
        except Exception as e:
            print(f"[!] Ошибка вызова Qwen LLM ({self.llm_model}): {e}")
            return None

    def _parse_qwen_json(self, raw_resp: str, sentences: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Robust parser for Qwen output into a list of clip dicts."""
        if not raw_resp:
            return []

        # Strip markdown fences if present
        clean = re.sub(r"^```(?:json)?\s*", "", raw_resp.strip(), flags=re.MULTILINE)
        clean = re.sub(r"\s*```$", "", clean.strip(), flags=re.MULTILINE)

        # Remove trailing commas before closing braces/brackets
        clean = re.sub(r",\s*([\]}])", r"\1", clean)

        parsed_data = None
        try:
            parsed_data = json.loads(clean)
        except json.JSONDecodeError:
            # Try to locate array bracket
            m = re.search(r"\[\s*\{.*\}\s*\]", clean, re.DOTALL)
            if m:
                try:
                    parsed_data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass

            if not parsed_data:
                # Try single dict object or wrapper
                m_obj = re.search(r"\{.*\}", clean, re.DOTALL)
                if m_obj:
                    try:
                        d = json.loads(m_obj.group(0))
                        if isinstance(d, dict):
                            for k in ["clips", "moments", "results", "items"]:
                                if k in d and isinstance(d[k], list):
                                    parsed_data = d[k]
                                    break
                            if not parsed_data:
                                parsed_data = [d]
                    except json.JSONDecodeError:
                        pass

        if not parsed_data or not isinstance(parsed_data, list):
            return []

        valid_clips = []
        for item in parsed_data:
            if not isinstance(item, dict):
                continue

            # Вікно в НОМЕРАХ речень (основний режим): точно, без галюцинації секунд
            sent_window = resolve_sent_window(sentences, item) if sentences else None
            if sent_window:
                s_sec = sentences[sent_window[0]]["start"]
                e_sec = sentences[sent_window[1]]["end"]

            if not sent_window:
                s_val = item.get("start_time") or item.get("start")
                e_val = item.get("end_time") or item.get("end")
                s_sec = parse_timestamp_seconds(s_val)
                e_sec = parse_timestamp_seconds(e_val)
            dur = float(item.get("duration_seconds", 0.0)) or (e_sec - s_sec)

            if s_sec >= e_sec or dur <= 3.0:
                continue

            hook = str(item.get("hook_title", "")).strip().upper()
            reason = str(item.get("viral_reason", "")).strip()
            unit_type = str(item.get("unit_type") or item.get("type") or "").strip().lower()
            if unit_type not in ("myth", "tip", "rule"):
                unit_type = None

            valid_clips.append({
                "clip_id": item.get("clip_id", len(valid_clips) + 1),
                "start": s_sec,
                "end": e_sec,
                "duration": dur,
                "hook_title": hook,
                "viral_reason": reason,
                "sent_window": sent_window,
                "unit_type": unit_type,
            })

        return valid_clips

    def analyze_transcript(
        self,
        segments: List[Dict[str, Any]],
        video_duration: float,
        cache_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Main analysis entrypoint (WOW-CUTTER):
        1. Sentence engine поверх Whisper word-таймстампів.
        2. Ранжований конвеєр: розмітка інтересу → кандидати → скоринг → реранк.
        3. Fallback: legacy-режим вікон Qwen / локальна нарізка.
        """
        all_words = []
        for s in segments:
            for w in (s.get("words") or []):
                all_words.append(w)

        if not segments:
            return []

        # Основний шлях: конвеєр «розмітка → кандидати → скоринг»
        if self.ranking_enabled and self.sentence_snap:
            sentences = build_sentences(segments, soft_pause=self.sentence_soft_pause)
            if sentences:
                clips = self._run_ranked_pipeline(segments, all_words, video_duration, sentences, cache_key)
                if clips:
                    return clips
                print("[🧠] Ранжований конвеєр не дав кліпів — fallback на legacy-режим")

        # Legacy-режим: вікна від Qwen (unit_type, номери речень)
        if self.is_available():
            clips = self._run_qwen_analysis(segments, all_words, video_duration)
            if clips:
                return clips
            print("[🧠] Qwen не вернул валидных клипов — переключение на локальный алгоритм")

        # Fallback to local high-precision rule-based segmentation
        return self._run_local_fallback(segments, all_words, video_duration)

    # ------------------------------------------------------------------
    # WOW-CUTTER: розмітка → кандидати → скоринг → реранк
    # ------------------------------------------------------------------

    def _run_ranked_pipeline(
        self,
        segments: List[Dict[str, Any]],
        all_words: List[Dict[str, Any]],
        video_duration: float,
        sentences: List[Dict[str, Any]],
        cache_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        llm = self._call_qwen if self.is_available() else None
        cache_path = None
        if cache_key:
            cache_path = interest_labeler.cache_file_path("temp", cache_key, sentences)

        labels = interest_labeler.label_transcript(
            sentences,
            llm_call=llm,
            cache_path=cache_path,
            progress=lambda done, total: print(f"[🏷] Розмітка інтересу: батч {done}/{total}"),
        )

        print(f"[📊] Генерація кандидатів (вікна {self.min_duration:.0f}-{self.max_duration:.0f}с)...", flush=True)
        candidates = clip_ranker.generate_candidates(
            sentences,
            min_dur=self.min_duration,
            max_dur=self.max_duration,
            hard_max=self.hard_max,
        )
        if not candidates:
            return []
        clip_ranker.score_all(candidates, sentences, labels, self.ranking_weights)
        print(f"[📊] Кандидатів: {len(candidates)}; топ-5 за скором: "
              + ", ".join(f"{c['scores']['total']:.0f}" for c in sorted(candidates, key=lambda x: x['scores']['total'], reverse=True)[:5]))

        # Шорт-лист для реранку: найкращі, вже без перетинів
        shortlist = clip_ranker.select_top(
            candidates,
            top_n=self.ranking_rerank_top,
            min_score=self.ranking_min_score * 0.6,  # реранку даємо ширший список
            max_overlap=0.4,
        )
        if not shortlist:
            return []

        # LLM-реранк: порівняти готові тексти і розставити хуки (надійно навіть для 3b)
        pick_order, hooks = self._llm_rerank(shortlist)
        if pick_order:
            by_id = {i: c for i, c in enumerate(shortlist)}
            final = [by_id[i] for i in pick_order if i in by_id]
            final += [c for c in shortlist if id(c) not in {id(x) for x in final}]
            print(f"[🧠] Qwen-реранк: переупорядкував {len(pick_order)} фіналістів")
        else:
            final = shortlist
            print("[🧠] Реранк недоступний — порядок за детермінованим скором")

        clips = []
        for cand in final[: self.ranking_top_n]:
            if cand["scores"]["total"] < self.ranking_min_score and clips:
                continue  # перший (найкращий) пропускаємо тільки якщо взагалі нічого немає
            cand = clip_ranker.apply_cold_open(
                cand, sentences, labels,
                floor_dur=self.clip_floor,
                max_shift=self.snap_max_extend * 0.5,
            )
            hook = hooks.get(id(cand))
            if cand.get("parts_total", 1) > 1:
                hook = self._unit_hook(hook, cand["text"], cand.get("part", 1), cand["parts_total"])
            clip = self._create_clip_object(
                len(clips) + 1, cand["start"], cand["end"], segments, all_words, video_duration,
                hook, f"score {cand['scores']['total']:.0f}/100 ({cand['source']})",
                clip_words=cand["words"], clip_text=cand["text"],
            )
            if clip:
                clip["score_breakdown"] = dict(cand["scores"])
                clips.append(clip)

        # Порядок за часом + захист від перетинів після cold-open зсувів
        clips.sort(key=lambda c: c["start"])
        final_clips = []
        prev_end = -1.0
        for c in clips:
            if c["start"] < prev_end - 1.0:
                continue
            prev_end = c["end"]
            final_clips.append(c)
        for i, c in enumerate(final_clips, 1):
            c["index"] = i
            c["clip_id"] = i
        print(f"[📊] Ранжований відбір: {len(final_clips)} кліпів (top_n={self.ranking_top_n})")
        return final_clips

    def _llm_rerank(self, shortlist: List[Dict[str, Any]]):
        """Qwen порівнює готові тексти кандидатів. Повертає (порядок id | None, хуки)."""
        if not self.is_available():
            return None, {}
        lines = []
        for i, c in enumerate(shortlist):
            text = c["text"][:220].replace("\n", " ")
            lines.append(
                f"[{i}] ({c['duration']:.0f}с, бал {c['scores']['total']:.0f}) {text}"
            )
        system = (
            "Ты — редактор вирусных TikTok-клипов по CS2 (СНГ аудитория). Тебе дают готовые кандидаты клипов "
            "с их текстом и оценкой. Твоя задача — отобрать клипы, которые наберут МАКСИМУМ КОММЕНТАРИЕВ и УДЕРЖАНИЯ.\n\n"
            "Приоритет: цепляющее начало (хук), динамика, законченный результат/вердикт.\n\n"
            "ГЛАВНОЕ: для каждого выбранного клипа сгенерируй ЖЕСТКИЙ БАЙТ НА КОММЕНТАРИИ в hook_title.\n"
            "hook_title должен быть 3-5 слов ALL-CAPS, провоцирующий зрителя спорить, доказывать или исправлять тебя в комментариях:\n"
            "- Провокация по скиллу/рангу: 'ТОЛЬКО 10 LVL ЭТО ЗНАЕТ', 'ТВОЙ РАНГ СИЛЬВЕР ЕСЛИ НЕ ЗНАЛ', 'ФИШКА ДЛЯ ПРО ИГРОКОВ'\n"
            "- Байт на спор/ошибку: 'ЭТОТ СМОК НИКТО НЕ ЗНАЕТ', 'ТИМЕЙТ РУИНИТ КАТКУ ЭТИМ', 'НИКОГДА НЕ ДЕЛАЙ ТАК В CS2'\n"
            "- Шок-контент и мифы: 'МИФ УНИЧТОЖЕН НАВСЕГДА!', 'ЭТО ПОЛНЫЙ ФЕЙК В CS2', 'ЧИТЕРСКИЙ ПИКСЕЛЬ-БУСТ'\n\n"
            "Return JSON ONLY: {\"picks\": [{\"id\": номер, \"hook_title\": \"...\"}]} "
            "— отсортированы от лучшего к худшему, максимум 12 штук."
        )
        user = "Кандидаты:\n\n" + "\n\n".join(lines) + "\n\nОтбери лучшие и верни JSON."
        raw = self._call_qwen(system, user)
        if not raw:
            return None, {}

        try:
            clean = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw.strip(), flags=re.MULTILINE)
            data = json.loads(clean)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                return None, {}
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None, {}

        picks = data.get("picks") if isinstance(data, dict) else data
        if not isinstance(picks, list):
            return None, {}

        order, hooks = [], {}
        for p in picks:
            if not isinstance(p, dict):
                continue
            pid = p.get("id", p.get("n"))
            # моделі часто повертають номери рядками ("3") або float
            if isinstance(pid, str) and pid.strip().isdigit():
                pid = int(pid.strip())
            elif isinstance(pid, float) and pid.is_integer():
                pid = int(pid)
            if not isinstance(pid, int) or not (0 <= pid < len(shortlist)):
                continue
            if pid in order:
                continue
            order.append(pid)
            hook = str(p.get("hook_title", "")).strip().upper()
            if self._hook_is_valid(hook):
                hooks[id(shortlist[pid])] = hook
        return (order if len(order) >= 2 else None), hooks

    @staticmethod
    def _hook_is_valid(hook: str) -> bool:
        if not hook or len(hook) > 60:
            return False
        words = hook.split()
        return 2 <= len(words) <= 7

    def regen_hook(self, clip: Dict[str, Any]) -> str:
        """Перегенерація хука для одного кліпу (TUI-огляд, клавіша R)."""
        text = (clip.get("text") or "")[:400]
        if self.is_available():
            system = (
                "Ты пишешь вирусные кликбейт-заголовки с байтом на комментарии для TikTok-клипов про CS2 (СНГ). "
                "Заголовок: 3-5 слов, ALL-CAPS, жесткая провокация, чтобы зритель побежал писать комментарий "
                "(примеры: 'ЭТОТ СМОК НИКТО НЕ ЗНАЕТ', 'ТИМЕЙТ РУИНИТ КАТКУ', 'ТОЛЬКО ДЛЯ 10 LVL', 'ТВОЙ РАНГ СИЛЬВЕР?'). "
                "Верни JSON ONLY: {\"variants\": [\"...\", \"...\", \"...\"]}"
            )
            raw = self._call_qwen(system, f"Текст клипа:\n{text}\n\nПридумай 3 варианта hook_title.")
            if raw:
                try:
                    clean = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw.strip(), flags=re.MULTILINE)
                    data = json.loads(clean)
                    for v in data.get("variants", []):
                        v = str(v).strip().upper()
                        if self._hook_is_valid(v):
                            return v
                except Exception:
                    pass
        return self._generate_russian_hook(clip.get("text", ""))

    def _run_qwen_analysis(
        self,
        segments: List[Dict[str, Any]],
        all_words: List[Dict[str, Any]],
        video_duration: float
    ) -> List[Dict[str, Any]]:
        """Sends transcript chunks to Qwen LLM and post-processes results."""
        print(f"[🧠] Запуск Qwen LLM ({self.llm_model}) для поиска вирусных моментов CS2...")

        # Межі речень із Whisper: єдине джерело істини. Будуються ДО промпту,
        # бо транскрипт нумерується саме реченнями (Qwen вибирає НОМЕРА, а не секунди)
        sentences = build_sentences(segments, soft_pause=self.sentence_soft_pause) if self.sentence_snap else []

        system_prompt = (
            "Ты — TikTok Viral Content Editor для CS2 клипов (СНГ аудитория). "
            "Твоя цель — клипы с МАКСИМАЛЬНЫМ УДЕРЖАНИЕМ (20-35 секунд) и ОГРОМНЫМ КОЛИЧЕСТВОМ КОММЕНТАРИЕВ.\n"
            "Видео состоит из СЕМАНТИЧЕСКИХ ЕДИНИЦ, каждая длиннее одного предложения:\n"
            "- ФИШКА: совет -> как сделать пошагово -> результат.\n"
            "- МИФ/ПРОВЕРКА: тезис мифа -> проверка/тест -> ВЕРДИКТ (миф оказался правдой или фейком).\n"
            "- ПРАВИЛО: правило -> объяснение -> пример из игры.\n"
            "ГЛАВНОЕ ПРАВИЛО: один клип = ОДНА полная единица, а не одно предложение. "
            "Каждый клип обязан содержать ЗАВЯЗКУ + ДЕЙСТВИЕ + РЕЗУЛЬТАТ. "
            "МИФ без ВЕРДИКТА возвращать ЗАПРЕЩЕНО: бери от тезиса мифа до вердикта целиком. "
            "Границы клипа — всегда начала/концы целых предложений, никогда не середина фразы. "
            "Если единица длиннее 38 секунд — раздели её на 2 части: Ч.1 заканчивается вопросом или "
            "самым интересным местом (клиффхэнгер), Ч.2 содержит ВЕРДИКТ. "
            "Если фишка не влезает в 25-30 секунд целиком — ищи более простую, но НИКОГДА не оставляй ответ оборванным. "
            "Один клип = одна единица. Без вступлений, приветствий и воды.\n\n"
            "hook_title: 3-5 слов ALL-CAPS. ОБЯЗАТЕЛЬНО делай БАЙТ НА КОММЕНТАРИИ и ПРОВОКАЦИЮ:\n"
            "- 'ЭТОТ СМОК НИКТО НЕ ЗНАЕТ!', 'ТИМЕЙТ РУИНИТ КАТКУ ЭТИМ!', 'ТОЛЬКО 10 LVL ЭТО ЗНАЕТ!', "
            "'ТВОЙ РАНГ СИЛЬВЕР ЕСЛИ НЕ ЗНАЛ!', 'МИФ УНИЧТОЖЕН НАВСЕГДА!', 'ЧИТЕРСКИЙ БУСТ В CS2!'.\n"
            "Зритель должен захотеть поспорить или высказаться в комментариях!\n"
            "Return JSON ONLY matching the required schema."
        )

        # Транскрипт нумерований реченнями: Qwen оперує НОМЕРАМИ, не галюцинує секунди
        lines = []
        for n, s in enumerate(sentences, 1):
            st = format_timestamp(s["start"])
            lines.append(f"[{n}] ({st}) {s['text']}")

        transcript_text = "\n".join(lines)
        if len(transcript_text) > 28000:
            transcript_text = transcript_text[:28000] + "\n...[остальное опущено]"

        user_prompt = (
            "Вот транскрипт видео по CS2. Каждая строка — предложение с номером в скобках [N]:\n\n"
            f"{transcript_text}\n\n"
            "ЗАДАЧА:\n"
            "Выдели СЕМАНТИЧЕСКИЕ ЕДИНИЦЫ (фишки, мифы с проверкой, правила) — каждую как ЗАКОНЧЕННУЮ историю.\n"
            "ПРАВИЛА:\n"
            "1. Длина КАЖДОГО клипа СТРОГО от 20 до 35 секунд (максимум 38 секунд, идеальный TikTok retention).\n"
            "2. Структура КАЖДОГО клипа: Завязка + Действие + Результат/Вердикт. Только законченные мысли!\n"
            "3. МИФ бери от тезиса до ВЕРДИКТА (миф оказался правдой/фейком). Миф без вердикта — запрещён.\n"
            "4. Миф длиннее 38с: два клипа — Ч.1 (тезис, обрыв на вопросе) и Ч.2 (проверка + ВЕРДИКТ).\n"
            "5. start_sent — НОМЕР первого предложения клипа, end_sent — номер последнего "
            "(того, где звучит результат/вердикт). В start_text/end_text скопируй первые/последние 3-4 слова этих предложений.\n"
            "6. Если совет не влезает в 25-35 секунд целиком — выбери более короткую/простую фишку. НЕ ОБРЫВАЙ на середине.\n"
            "7. hook_title: 3-5 слов ALL-CAPS, ОБЯЗАТЕЛЬНО делай БАЙТ НА КОММЕНТАРИИ ('ТОЛЬКО 10 LVL ЗНАЕТ ЭТО', 'ТИМЕЙТ РУИНИТ КАТКУ', 'ЭТОТ СМОК НИКТО НЕ ЗНАЕТ').\n"
            "8. Выведи СТРОГО JSON-массив следующего формата:\n"
            "[\n"
            "  {\n"
            '    "clip_id": 1,\n'
            '    "start_sent": 12,\n'
            '    "end_sent": 18,\n'
            '    "start_text": "смотрим следующий момент",\n'
            '    "end_text": "оценку фейк",\n'
            '    "unit_type": "myth",\n'
            '    "start_time": "00:01:23.500",\n'
            '    "end_time": "00:01:41.200",\n'
            '    "duration_seconds": 17.7,\n'
            '    "hook_title": "МИФ УНИЧТОЖЕН НАВСЕГДА!",\n'
            '    "viral_reason": "Момент оказался фейком, вердикт с доказательством"\n'
            "  }\n"
            "]\n"
            'unit_type: "myth" (миф/проверка), "tip" (фишка) или "rule" (правило). '
            "Поля start_time/end_time — запасные, указывай время по таймкодам выбранных предложений.\n"
            "Верни ТОЛЬКО валидный JSON без лишнего текста."
        )

        raw = self._call_qwen(system_prompt, user_prompt)
        raw_clips = self._parse_qwen_json(raw, sentences)
        if not raw_clips:
            return []

        # Post-process and normalize clips with word alignment and duration limits
        processed = []
        for idx, item in enumerate(raw_clips, 1):
            s_raw = item["start"]
            e_raw = item["end"]
            dur = e_raw - s_raw

            # Micro-compilation split if Qwen returned > hard_max
            if dur > self.hard_max:
                print(f"[🧠] Qwen вернул клип длительностью {dur:.1f}с (> {self.hard_max}с) — разделяем на микро-клипы 15-25с")
                subclips = self._split_long_range(
                    s_raw, e_raw, sentences, segments, all_words, video_duration,
                    item.get("hook_title"), item.get("viral_reason"),
                    unit_type=item.get("unit_type"),
                    sent_window=item.get("sent_window"),
                )
                for sc in subclips:
                    processed.append(sc)
                continue

            clip_objs = self._build_unit_clips(
                idx, s_raw, e_raw, sentences, segments, all_words, video_duration,
                item.get("hook_title"), item.get("viral_reason"),
                sent_window=item.get("sent_window"),
                unit_type=item.get("unit_type"),
            )
            for c_obj in clip_objs:
                processed.append(c_obj)

        # Enforce uniqueness and order; перекриття вікон = дублікати моменту
        processed.sort(key=lambda c: c["start"])
        final_clips = []
        prev_end = -1.0
        for c in processed:
            if c["start"] < prev_end - 1.0:
                print(f"[🧠] Клип [{c['start']:.1f}-{c['end']:.1f}с] пересекается с предыдущим — пропускаем (дубликат)")
                continue
            prev_end = c["end"]
            final_clips.append(c)

        for i, c in enumerate(final_clips, 1):
            c["index"] = i
            c["clip_id"] = i

        print(f"[🧠] Qwen успешно сформировал {len(final_clips)} вирусных 15-25с клипов с хуками")
        return final_clips

    def _build_unit_clips(
        self,
        index: int,
        s_raw: float,
        e_raw: float,
        sentences: List[Dict[str, Any]],
        segments: List[Dict[str, Any]],
        all_words: List[Dict[str, Any]],
        video_duration: float,
        hook_title: Optional[str] = None,
        viral_reason: Optional[str] = None,
        sent_window: Optional[Tuple[int, int]] = None,
        unit_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Будує кліпи з вікна LLM з урахуванням СЕМАНТИЧНИХ ОДИНИЦЬ.

        1. Вікно від Qwen приходить у НОМЕРАХ речень (безпечний режим) або
           секундами (резерв); обидва варіанти зводяться до меж речень.
        2. Міф/перевірка (unit_type від Qwen або маркер тези в тексті):
           вердикт гарантовано в кадрі — complete_myth_window дорощує вікно до
           вердикту, довгий міф ділиться на Ч.1 (кліфгенгер) + Ч.2 (вердикт).
        3. Звичайна фішка: межі снапляться до речень Whisper, хвіст мусить бути
           завершеним; завершену думку покласти у вікно неможливо — відкидаємо.
        """
        if not sentences:
            # Немає word-таймстампів — працюємо по сирому вікну (стара поведінка)
            c = self._create_clip_object(
                index, s_raw, e_raw, segments, all_words, video_duration,
                hook_title, viral_reason
            )
            return [c] if c else []

        # --- Вікно в індексах речень
        if sent_window and 0 <= sent_window[0] <= sent_window[1] < len(sentences):
            first_idx, last_idx = sent_window
        else:
            first_idx = next((i for i, s in enumerate(sentences) if s["end"] > s_raw), None)
            last_idx = next(
                (len(sentences) - 1 - i for i, s in enumerate(reversed(sentences)) if s["start"] < e_raw),
                None,
            )
            if first_idx is None or last_idx is None or first_idx > last_idx:
                return []

        # --- Міф-режим: unit_type від Qwen АБО маркер тези у вікні (±2 речення до старту)
        intro_lo = max(0, first_idx - 2)
        myth_like = (
            unit_type == "myth"
            or any(looks_like_myth_intro(sentences[m]["text"]) for m in range(intro_lo, last_idx + 1))
        )
        if myth_like:
            planned = complete_myth_window(
                sentences, first_idx, last_idx,
                min_dur=self.min_duration, floor_dur=self.clip_floor,
                max_dur=self.max_duration, hard_max=self.hard_max,
            )
            if planned:
                if len(planned) > 1:
                    span = sentences[planned[-1][1]]["end"] - sentences[planned[0][0]]["start"]
                    print(f"[🧠] Клип #{index}: миф/проверка длиной {span:.0f}с — делим на {len(planned)} частей, вердикт в последней")
                clips = []
                for p_i, (i0, i1) in enumerate(planned, 1):
                    clip_words, clip_text = slice_words_and_text(sentences, i0, i1)
                    hook = self._unit_hook(hook_title, clip_text, p_i, len(planned))
                    c = self._create_clip_object(
                        index, sentences[i0]["start"], sentences[i1]["end"],
                        segments, all_words, video_duration,
                        hook, viral_reason, clip_words=clip_words, clip_text=clip_text
                    )
                    if c:
                        clips.append(c)
                if clips:
                    return clips
                # planned не дав жодного валідного кліпу — падаємо у звичайний шлях

        # --- Звичайне вікно: снап по реченнях (для sent_window межі вже точні —
        # snap лише підтвердить їх і застосуує стелі тривалості)
        snap = snap_clip_window(
            sentences,
            sentences[first_idx]["start"],
            sentences[last_idx]["end"],
            min_dur=self.min_duration,
            floor_dur=self.clip_floor,
            max_dur=self.max_duration,
            hard_max=self.hard_max,
            max_extend=self.snap_max_extend,
        )
        if snap is None:
            print(f"[✂️] Клип #{index} [{s_raw:.1f}-{e_raw:.1f}с] не содержит законченной мысли — отброшен")
            return []

        shift = abs(snap["start"] - s_raw) + abs(snap["end"] - e_raw)
        if shift > 0.5:
            print(
                f"[✂️] Клип #{index}: границы выровнены по предложениям "
                f"({s_raw:.1f}-{e_raw:.1f}с -> {snap['start']:.1f}-{snap['end']:.1f}с)"
            )
        clip_words, clip_text = slice_words_and_text(sentences, snap["first_idx"], snap["last_idx"])
        if not clip_words:
            return []
        c = self._create_clip_object(
            index, snap["start"], snap["end"], segments, all_words, video_duration,
            hook_title, viral_reason, clip_words=clip_words, clip_text=clip_text
        )
        return [c] if c else []

    def _unit_hook(self, base_hook: Optional[str], part_text: str, part: int, total: int) -> Optional[str]:
        """
        Хук для частини юніту: усі частини, крім фінальної, брендуються базовим
        хуком Qwen ('ФЕЙК ИЛИ НЕТ? Ч.2'); фінальна — з власного тексту, бо саме
        там звучить вердикт ('МИФ УНИЧТОЖЕН В CS2! Ч.3').
        """
        if total <= 1:
            return base_hook
        if part < total and base_hook:
            return f"{base_hook} Ч.{part}"
        hook = self._generate_russian_hook(part_text, base_hook, part)
        return hook if f"Ч.{part}" in hook else f"{hook} Ч.{part}"

    def _split_long_range(
        self,
        start_time: float,
        end_time: float,
        sentences: List[Dict[str, Any]],
        segments: List[Dict[str, Any]],
        all_words: List[Dict[str, Any]],
        video_duration: float,
        base_hook: Optional[str],
        viral_reason: Optional[str] = None,
        unit_type: Optional[str] = None,
        sent_window: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        """Splits an oversized range (> 28s) into complete 15-25s sentence-aligned micro-stories."""
        # Без word-таймстампів — аварійний різ по часу (стара поведінка)
        if not sentences:
            slices = []
            cur_s = start_time
            part = 1
            while cur_s < end_time:
                cur_e = min(end_time, cur_s + self.max_duration)
                if (cur_e - cur_s) >= self.min_duration:
                    hook = f"{base_hook or 'СЕКРЕТНЫЙ РАСКИД В CS2'} (ЧАСТЬ {part})"
                    slices.append({
                        "start": cur_s,
                        "end": cur_e,
                        "duration": cur_e - cur_s,
                        "hook_title": hook,
                        "viral_reason": viral_reason or f"Часть {part} CS2 гайда",
                        "text": "",
                        "segments": []
                    })
                    part += 1
                cur_s = cur_e
            return slices

        idxs = [i for i, s in enumerate(sentences) if s["end"] > start_time and s["start"] < end_time]
        if not idxs:
            return []

        # Міф від Qwen: вердикт гарантовано в останній частині
        if unit_type == "myth" or (sent_window and any(
            looks_like_myth_intro(sentences[m]["text"])
            for m in range(max(0, sent_window[0] - 2), min(sent_window[1] + 1, len(sentences)))
        )):
            verdict_end = sent_window[1] if sent_window else idxs[-1]
            planned = complete_myth_window(
                sentences, sent_window[0] if sent_window else idxs[0], verdict_end,
                min_dur=self.min_duration, floor_dur=self.clip_floor,
                max_dur=self.max_duration, hard_max=self.hard_max,
            )
            if planned:
                windows = planned
            else:
                windows = merge_windows_with_units(
                    sentences, idxs[0], idxs[-1],
                    min_dur=self.min_duration,
                    floor_dur=self.clip_floor,
                    max_dur=self.max_duration,
                    hard_max=self.hard_max,
                )
        else:
            # Міф-юніти всередині діапазону плануються з гарантією вердикту,
            # решта контенту — звичайними вікнами завершених думок
            windows = merge_windows_with_units(
                sentences, idxs[0], idxs[-1],
                min_dur=self.min_duration,
                floor_dur=self.clip_floor,
                max_dur=self.max_duration,
                hard_max=self.hard_max,
            )

        clips = []
        total = len(windows)
        for w_i, (i0, i1) in enumerate(windows, 1):
            c_start = sentences[i0]["start"]
            c_end = sentences[i1]["end"]
            clip_words, clip_text = slice_words_and_text(sentences, i0, i1)
            if total > 1:
                hook = self._unit_hook(base_hook, clip_text, w_i, total)
                if not hook:
                    hook = self._generate_russian_hook(clip_text, base_hook, w_i)
            else:
                hook = self._sanitize_russian_hook(base_hook, clip_text)
            c = self._create_clip_object(
                w_i, c_start, c_end, segments, all_words, video_duration,
                hook, viral_reason, clip_words=clip_words, clip_text=clip_text
            )
            if c:
                clips.append(c)

        return clips

    def _create_clip_object(
        self,
        index: int,
        start_time: float,
        end_time: float,
        segments: List[Dict[str, Any]],
        all_words: List[Dict[str, Any]],
        video_duration: float,
        hook_title: Optional[str] = None,
        viral_reason: Optional[str] = None,
        clip_words: Optional[List[Dict[str, Any]]] = None,
        clip_text: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Builds a verified clip object applying:
        - Precise audio trimming (starts on exact first word millisecond, ends on last word).
        - Hard duration bounds [floor, 28.0s].
        - Pacing verification.
        - Russian clickbait hook title in ALL-CAPS.

        clip_words/clip_text: готовий sentence-snapped список слів та текст —
        використовуються як є, щоб повторна фільтрація по вікну не зламала межі.
        """
        # Слова кліпу: готовий список АБО вікно навколо сирого таймкоду LLM
        if clip_words is None:
            clip_words = [
                w for w in all_words
                if w["start"] >= (start_time - 0.4) and w["end"] <= (end_time + 0.4)
            ]

        if clip_words:
            # Shift start to catch action before the first spoken syllable (action hook)
            actual_start = max(0.0, clip_words[0]["start"] + self.pad_start)
            # Cut end immediately after the final word is spoken (zero trailing silence)
            actual_end = min(video_duration, clip_words[-1]["end"] + self.pad_end)
        else:
            actual_start = max(0.0, start_time + self.pad_start)
            actual_end = min(video_duration, end_time + self.pad_end)

        dur = actual_end - actual_start

        # Enforce strict 20-35s constraint (max 38.0s)
        if dur > self.hard_max:
            # Trim from end to max_duration at word boundary
            target_end = actual_start + self.max_duration
            closest_words = [w for w in clip_words if w["end"] <= target_end]
            if closest_words:
                actual_end = closest_words[-1]["end"] + self.pad_end
                dur = actual_end - actual_start
            else:
                actual_end = actual_start + self.max_duration
                dur = self.max_duration

        if dur < self.clip_floor:
            return None

        # Collect segment data for subtitles
        clip_segments = []
        for s in segments:
            if s["end"] >= actual_start and s["start"] <= actual_end:
                seg_words = [w for w in (s.get("words") or []) if w["start"] >= actual_start and w["end"] <= actual_end]
                clip_segments.append({
                    "start": max(actual_start, s["start"]),
                    "end": min(actual_end, s["end"]),
                    "text": s.get("text", ""),
                    "words": seg_words
                })

        if not clip_text:
            clip_text = " ".join(s.get("text", "") for s in clip_segments).strip()

        # Generate or sanitize Russian hook title
        final_hook = self._sanitize_russian_hook(hook_title, clip_text)

        # Detect CS2 entities (map, nade, callout) in Russian
        entities = self.detect_entities_ru(clip_text)

        # Pacing analysis: identify internal gaps > 0.8s
        internal_gaps = []
        for i in range(len(clip_words) - 1):
            gap = clip_words[i + 1]["start"] - clip_words[i]["end"]
            if gap > self.max_internal_pause:
                internal_gaps.append({
                    "gap_start": clip_words[i]["end"],
                    "gap_end": clip_words[i + 1]["start"],
                    "duration": gap
                })

        return {
            "index": index,
            "start": round(actual_start, 3),
            "end": round(actual_end, 3),
            "duration": round(dur, 2),
            "title": self._generate_slug(entities, final_hook),
            "hook_title": final_hook,
            "viral_reason": viral_reason or "CS2 вирусный момент",
            "text": clip_text,
            "segments": clip_segments,
            "words": clip_words,
            "entities": entities,
            "internal_gaps": internal_gaps,
        }

    def _sanitize_russian_hook(self, hook: Optional[str], text: str) -> str:
        """Ensures hook_title is a punchy 3-5 word Russian ALL-CAPS string."""
        if hook:
            clean = hook.strip().upper()
            # Remove punctuation except exclamation marks and part suffix dot (Ч.1)
            clean = re.sub(r"[^\w\s!\?.-]", "", clean)
            words = clean.split()
            if 2 <= len(words) <= 7:
                return clean

        return self._generate_russian_hook(text)

    def _generate_russian_hook(self, text: str, base_hook: Optional[str] = None, part: int = 1) -> str:
        """
        Generates a provocative Russian CS2 hook title with aggressive engagement bait
        (байт на комментарии: споры о ранге, 10 lvl, тиммейты, мифы).
        """
        ent = self.detect_entities_ru(text)
        nade = (ent.get("nade") or "").upper()
        map_name = (ent.get("map") or "").upper()
        callout = (ent.get("callout") or "").upper()
        t = text.lower()

        is_boost = bool(re.search(r"\bбуст|\bподсадк|\bзалез|\bзапрыгн|\bboost", t))
        is_pixel = bool(re.search(r"пиксель|піксель|pixel", t))
        is_myth_intro = looks_like_myth_intro(t)
        is_myth_verdict = looks_like_verdict(t)

        part_suffix = f" Ч.{part}" if part > 1 else ""
        variants: List[str] = []

        # Міфи: вердикт або провокація
        if is_myth_verdict:
            variants.append(f"МИФ УНИЧТОЖЕН НАВСЕГДА!{part_suffix}")
            variants.append(f"ЭТО ПОЛНЫЙ ФЕЙК В CS2!{part_suffix}")
        elif is_myth_intro:
            variants.append(f"ПРОВЕРКА: ФЕЙК ИЛИ НЕТ?{part_suffix}")
            variants.append(f"99% ИГРОКОВ ВЕРЯТ В ЭТОТ МИФ!{part_suffix}")

        # Провокації з бустами / пікселями
        if is_pixel and callout:
            variants.append(f"ЗАПРЕЩЕННЫЙ ПИКСЕЛЬ НА {callout}!{part_suffix}")
        if is_boost and callout:
            variants.append(f"ЧИТЕРСКИЙ БУСТ НА {callout}!{part_suffix}")
        if is_boost and map_name:
            variants.append(f"ТОЛЬКО 10 LVL ЗНАЕТ БУСТ НА {map_name}!{part_suffix}")

        # Провокації з гранатами (байт на комменти про ранг)
        if callout and nade:
            variants.append(f"ЭТОТ {nade} НА {callout} НИКТО НЕ ЗНАЕТ!{part_suffix}")
            variants.append(f"ТИМЕЙТ РУИНИТ КАТКУ ЭТИМ {nade}!{part_suffix}")
        if nade and map_name:
            variants.append(f"ТОЛЬКО НУБЫ КИДАЮТ ТАК {nade} НА {map_name}!{part_suffix}")
            variants.append(f"ЛУЧШИЙ {nade} НА {map_name} В CS2!{part_suffix}")

        if callout:
            variants.append(f"ТИМЕЙТ СЛИВАЕТ {callout} В CS2!{part_suffix}")
            variants.append(f"ТВОЙ РАНГ СИЛЬВЕР ЕСЛИ НЕ ЗНАЛ {callout}!{part_suffix}")

        if map_name:
            variants.append(f"НЕ ДЕЛАЙ ТАК НА {map_name} В CS2!{part_suffix}")
            variants.append(f"КАК ВЫИГРАТЬ {map_name} В СОЛО!{part_suffix}")

        variants.append(f"ЭТУ ФИШКУ ЗНАЮТ ТОЛЬКО ПРО!{part_suffix}")
        variants.append(f"ТИМЕЙТ РУИНИТ КАТКУ ЭТИМ!{part_suffix}")
        return variants[0]

    def _generate_slug(self, entities: Dict[str, str], hook: str) -> str:
        """Generate safe filesystem slug."""
        m = entities.get("map", "cs2").lower()
        n = entities.get("nade", "tip").lower()
        clean_hook = re.sub(r"[^\w\s]", "", hook.lower()).strip()
        words = clean_hook.split()[:2]
        slug = f"{m}_{n}_" + "_".join(words)
        return re.sub(r"[^\w\-]", "", slug)[:32] or "cs2_clip"

    def detect_entities_ru(self, text: str) -> Dict[str, str]:
        """Extracts CS2 map, grenade, and callout in Russian."""
        t = text.lower()
        found = {"map": "", "nade": "", "callout": ""}
        for k, v in MAP_KEYWORDS_RU.items():
            if re.search(r"\b" + re.escape(k) + r"\b", t):
                found["map"] = v
                break
        for k, v in NADE_KEYWORDS_RU.items():
            if re.search(r"\b" + re.escape(k) + r"\b", t):
                found["nade"] = v
                break
        for k, v in CALLOUT_KEYWORDS_RU.items():
            if re.search(r"\b" + re.escape(k) + r"\b", t):
                found["callout"] = v
                break
        return found

    def _run_local_fallback(
        self,
        segments: List[Dict[str, Any]],
        all_words: List[Dict[str, Any]],
        video_duration: float
    ) -> List[Dict[str, Any]]:
        """
        Local fallback engine: groups the whole transcript into complete-thought
        windows of 15-25s (max 28s), aligned to Whisper sentence boundaries.
        """
        print("[🧠] Работает локальный алгоритм нарезки 15-25с клипов...")
        sentences = build_sentences(segments, soft_pause=self.sentence_soft_pause)
        if not sentences:
            return []

        # Міф-юніти (теза -> перевірка -> вердикт) плануються окремо,
        # решта — звичайні вікна завершених думок
        windows = merge_windows_with_units(
            sentences, 0, len(sentences) - 1,
            min_dur=self.min_duration,
            floor_dur=self.clip_floor,
            max_dur=self.max_duration,
            hard_max=self.hard_max,
        )

        clips = []
        for w_i, (i0, i1) in enumerate(windows, 1):
            c_start = sentences[i0]["start"]
            c_end = sentences[i1]["end"]
            clip_words, clip_text = slice_words_and_text(sentences, i0, i1)
            hook = self._generate_russian_hook(clip_text)
            c = self._create_clip_object(
                w_i, c_start, c_end, segments, all_words, video_duration,
                hook, "Обучающий момент CS2",
                clip_words=clip_words, clip_text=clip_text
            )
            if c:
                clips.append(c)

        return clips
