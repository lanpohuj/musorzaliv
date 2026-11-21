"""
Context Intelligence Layer (100% безкоштовний, локальний).

Дворівнева архітектура розуміння контексту:
  Рівень 1 (завжди працює, чистий Python, 0 залежностей):
    - детекція зміни теми через лексичну когезію (Jaccard по змістовних словах)
    - класифікація сегментів: контент / інтро / аутро / реклама-спонсор
    - скоринг кліпів на віральність (хуки, щільність мови, ключові слова, сміття)
    - витягування сутностей (карта, граната, позиція) та хук-фрази
  Рівень 2 (опційно, якщо встановлений локальний Ollama LLM):
    - генерація заголовків/описів українською
    - ранжування кліпів на віральність
    Якщо Ollama недоступний — автоматичний фолбек на Рівень 1.
"""

import json
import re
import urllib.request
from typing import List, Dict, Any, Optional, Tuple

from pipeline.analyzer_qwen import QwenAnalyzer, MAP_KEYWORDS_RU, NADE_KEYWORDS_RU


# ----------------------------------------------------------------------------
# Словники (UA / RU / EN) — розширювані через конфіг не потрібні, тут повний набір
# ----------------------------------------------------------------------------

MAP_KEYWORDS = {
    "mirage": "mirage", "міраж": "mirage", "мираж": "mirage",
    "inferno": "inferno", "інферно": "inferno", "инферно": "inferno",
    "anubis": "anubis", "анубіс": "anubis", "анубис": "anubis",
    "ancient": "ancient", "енчієнт": "ancient",
    "nuke": "nuke", "нюк": "nuke", "ньюк": "nuke",
    "dust": "dust2", "daст": "dust2", "даст": "dust2", "даст2": "dust2", "dust2": "dust2", "песок": "dust2",
    "vertigo": "vertigo", "вертіго": "vertigo", "вертиго": "vertigo",
    "overpass": "overpass", "овер": "overpass",
    "train": "train", "трейн": "train",
}

NADE_KEYWORDS = {
    "smoke": "smoke", "смок": "smoke", "смоків": "smoke", "смокчик": "smoke", "дим": "smoke", "дыма": "smoke",
    "flash": "flash", "флеш": "flash", "флешка": "flash", "флешку": "flash", "світло": "flash",
    "molotov": "molly", "molly": "molly", "молотов": "molly", "молік": "molly", "молотова": "molly", "огонь": "molly",
    "he": "he", "граната": "he", "хейку": "he", "осколкова": "he",
    "incendiary": "incendiary", "запальна": "incendiary",
    "decoy": "decoy", "приманка": "decoy",
}

CALLOUT_KEYWORDS = {
    # позиції загальні
    "mid": "mid", "міда": "mid", "мида": "mid",
    "window": "window", "вікно": "window", "окно": "window",
    "short": "short", "шорт": "short",
    "long": "long", "лонг": "long",
    "connector": "connector", "коннектор": "connector", "конектор": "connector",
    "banana": "banana", "банан": "banana",
    "palace": "palace", "палац": "palace",
    "ramp": "ramp", "рампа": "ramp",
    "catwalk": "catwalk", "кетвок": "catwalk",
    "heaven": "heaven", "хевен": "heaven",
    "hell": "hell",
    "site": "site", "плент": "site", "плант": "site", "plant": "site",
    "a site": "a_site", "b site": "b_site",
    "outside": "outside", "аутсайд": "outside",
    "boost": "boost", "буст": "boost", "підсадка": "boost", "подсадка": "boost",
    "pixel": "pixel", "піксель": "pixel", "пиксель": "pixel",
    "van": "van", "фургон": "van",
    "trip": "triple", "чотири": "quad",
    "jumpthrow": "jumpthrow", "джамп": "jumpthrow",
}

HOOK_PATTERNS = [
    r"\bімб", r"\bимб", r"\bсекрет", r"\bsecret\b", r"\bтоп[- ]?\d", r"\btop[- ]?\d",
    r"\bлют", r"\bжест", r"\bнайкращ", r"\bлучш", r"\bbest\b", r"\bmeta\b", r"\bмета\b",
    r"\bмало хто\b", r"\bніхто не\b", r"\bnobody\b", r"\bпрост", r"\beasy\b", r"\bлегк",
    r"\bpro\b", r"\bпро\b", r"\bчикат", r"\bчитер", r"\bнов", r"\bновачк",
    r"\bлайфхак", r"\bфішк", r"\bфич", r"\bтрюк", r"\btrick",
]

QUESTION_PATTERN = re.compile(r"\?|^(чому|як|що|де|коли|why|how|what|where|яким|який)\b", re.IGNORECASE)

# Сегменти, що не несуть цінності для шортсів: інтро, аутро, реклама, заклики
SPONSOR_PATTERNS = [
    r"підпису", r"подпиш", r"subscribe", r"subscrib",
    r"лайк", r"ставь лайк", r"\blike\b.*\bvideo\b", r"дізве", r"звон", r"колокольчик", r"\bbell\b",
    r"промокод", r"промо-?код", r"promo", r"промо\b",
    r"спонсор", r"sponsor", r"патрон", r"patreon", r"boosty", r"бусті",
    r"донат", r"donate", r"підтрим", r"поддерж",
    r"discord", r"дискорд", r"telegram", r"телеграм", r"тілеграм",
    r"instagram", r"інстаграм", r"инстаграм", r"tiktok", r"тікток",
    r"в описі", r"в опис", r"у опис", r"внизу відео", r"description",
    r"плейліст", r"playlist", r"попереднє відео", r"наступне відео", r"на канал",
    r"казино", r"рулетк", r"ставк", r"бонус", r"знижк", r"скидк",
    r"привіт.*друз", r"привіт.*друзі", r"вітаю", r"hello.*guys", r"hi.*guys", r"what's up",
    r"побачимос", r"до зустрічі", r"всім Peace", r"goodbye", r"пока\b",
]

FILLER_WORDS = {
    "е", "ее", "еее", "ээ", "ну", "типу", "типа", "значить", "значит",
    "какби", "какбы", "впринципі", "впринципе", "короче", "взагалі",
    "вобщем", "вобщто", "мм", "ммм",
}

STOPWORDS = {
    "і", "й", "та", "і", "the", "a", "an", "of", "to", "in", "on", "at",
    "і", "у", "в", "на", "з", "із", "до", "для", "це", "цей", "ця", "це",
    "що", "як", "так", "там", "тут", "тільки", "ще", "уже", "вже", "був",
    "is", "are", "was", "be", "this", "that", "it", "and", "or", "but",
    "он", "она", "они", "мы", "вы", "я", "ты", "не", "нет", "да", "же",
    "би", "б", "от", "по", "за", "из", "не", "он", "нам", "вас", "нас",
}

SENTENCE_END = (".", "!", "?")


# ----------------------------------------------------------------------------
# Утиліти токенізації
# ----------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Розбиває текст на нормалізовані токени (нижній регістр, без пунктуації)."""
    return re.findall(r"[a-zа-яіїєґ0-9]+", text.lower(), re.UNICODE)


def content_tokens(text: str) -> List[str]:
    """Тільки змістовні слова (без стоп-слів та філерів) для порівняння тем."""
    return [t for t in tokenize(text) if t not in STOPWORDS and t not in FILLER_WORDS and len(t) > 2]


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ----------------------------------------------------------------------------
# Головний клас
# ----------------------------------------------------------------------------

class ContextAnalyzer:
    def __init__(self, config: dict):
        cfg = config.get("analysis", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.min_score = float(cfg.get("min_score", 4.0))
        self.max_clips = int(cfg.get("max_clips", 0)) or 0  # 0 = без обмежень
        self.drop_sponsor = bool(cfg.get("drop_sponsor_segments", True))
        self.topic_shift_threshold = float(cfg.get("topic_shift_threshold", 0.12))
        self.cohesion_window = int(cfg.get("cohesion_window", 2))

        # Ollama (опційний локальний LLM — безкоштовний)
        self.use_llm = bool(cfg.get("use_llm", True))
        self.llm_url = cfg.get("llm_url", "http://localhost:11434")
        self.llm_model = cfg.get("llm_model", "qwen2.5:3b")
        self.llm_timeout = float(cfg.get("llm_timeout", 180.0))
        self._llm_checked = False
        self._llm_ok = False

        self.sponsor_re = [re.compile(p, re.IGNORECASE) for p in SPONSOR_PATTERNS]
        self.hook_re = [re.compile(p, re.IGNORECASE) for p in HOOK_PATTERNS]
        self.qwen = QwenAnalyzer(config)

    def analyze_transcript(self, segments: List[Dict[str, Any]], video_duration: float, cache_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Delegates transcript analysis to Qwen LLM module."""
        return self.qwen.analyze_transcript(segments, video_duration, cache_key)

    # ------------------------------------------------------------------
    # Ollama LLM (Рівень 2) — тільки локально, тільки безкоштовно
    # ------------------------------------------------------------------

    def llm_available(self) -> bool:
        """Перевіряє один раз, чи піднятий локальний Ollama з потрібною моделлю."""
        if not self.enabled or not self.use_llm:
            return False
        if self._llm_checked:
            return self._llm_ok
        self._llm_checked = True
        try:
            req = urllib.request.Request(
                f"{self.llm_url}/api/tags",
                method="GET",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                # Модель може бути з тегом або без — шукаємо будь-який префікс
                base = self.llm_model.split(":")[0]
                self._llm_ok = any(m == self.llm_model or m.startswith(base) for m in models)
                if not self._llm_ok and models:
                    # Модель не тягнута, але Ollama живий — використаємо першу доступну
                    self.llm_model = models[0]
                    self._llm_ok = True
        except Exception:
            self._llm_ok = False
        return self._llm_ok

    def llm_generate(self, system: str, prompt: str, max_tokens: int = 256) -> Optional[str]:
        """Генерація через /api/generate. Повертає None при будь-якій помилці."""
        if not self.llm_available():
            return None
        try:
            payload = json.dumps({
                "model": self.llm_model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.7,
                    "num_predict": max_tokens,
                },
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{self.llm_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.llm_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return (data.get("response") or "").strip()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Класифікація сегментів (Рівень 1)
    # ------------------------------------------------------------------

    def classify_segment(self, text: str) -> str:
        """Класифікує сегмент мови: content / sponsor / outro / intro."""
        if not text:
            return "content"
        hits = sum(1 for rx in self.sponsor_re if rx.search(text))
        words = tokenize(text)
        # Короткий сегмент з одним збігом — ігноруємо (наприклад, "а тепер смок...")
        density = hits / max(1, len(words) / 8.0)
        if hits >= 2 and density >= 0.4:
            return "sponsor"
        if hits >= 1 and density >= 0.8:
            return "sponsor"
        return "content"

    def filter_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Викидає рекламні/службові сегменти, зберігає порядок."""
        if not self.enabled or not self.drop_sponsor or not segments:
            return segments
        kept = []
        removed = 0
        for s in segments:
            label = self.classify_segment(s.get("text", ""))
            if label == "sponsor":
                removed += 1
            else:
                kept.append(s)
        if removed:
            print(f"[🧠] Контекст-фільтр: викинуто {removed} рекламних/службових сегментів")
        return kept if kept else segments

    # ------------------------------------------------------------------
    # Детекція зміни теми (лексична когезія)
    # ------------------------------------------------------------------

    def detect_topic_boundaries(self, segments: List[Dict[str, Any]]) -> set:
        """
        Шукає моменти, де лексика різко змінюється — це природні межі тем.
        Повертає множину індексів сегментів, з яких починається НОВА тема.
        """
        if not self.enabled or len(segments) < 4:
            return set()

        boundaries = set()
        w = max(1, self.cohesion_window)

        for i in range(w, len(segments) - w + 1):
            before = set()
            for s in segments[i - w:i]:
                before.update(content_tokens(s.get("text", "")))
            after = set()
            for s in segments[i:i + w]:
                after.update(content_tokens(s.get("text", "")))

            sim = jaccard(before, after)

            if sim < self.topic_shift_threshold and (before and after):
                boundaries.add(i)

        # Прибираємо межі, що стоять поруч (пік змінності один)
        sorted_b = sorted(boundaries)
        pruned = []
        for b in sorted_b:
            if pruned and b - pruned[-1] <= self.cohesion_window:
                continue
            pruned.append(b)
        return set(pruned)

    # ------------------------------------------------------------------
    # Сутності та хуки
    # ------------------------------------------------------------------

    def detect_entities(self, text: str) -> Dict[str, str]:
        """Витягує карту, тип гранати та позицію з тексту кліпу."""
        tokens = tokenize(text)
        found = {"map": "", "nade": "", "callout": ""}
        bigrams = [" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)]

        for t in tokens:
            if not found["map"] and t in MAP_KEYWORDS:
                found["map"] = MAP_KEYWORDS[t]
            if not found["nade"] and t in NADE_KEYWORDS:
                found["nade"] = NADE_KEYWORDS[t]
            if not found["callout"] and t in CALLOUT_KEYWORDS:
                found["callout"] = CALLOUT_KEYWORDS[t]
        for bg in bigrams:
            if bg in ("a site", "b site"):
                found["callout"] = bg.replace(" ", "_")
        return found

    def find_hook_line(self, clip: Dict[str, Any]) -> str:
        """
        Знаходить найзахопливішу фразу кліпу для заголовка/опису:
        перше речення з хук-словом, питанням або суперлативом.
        """
        text = clip.get("text", "")
        if not text:
            return ""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        best = ""
        best_score = -1
        for s in sentences:
            sc = sum(1 for rx in self.hook_re if rx.search(s))
            if QUESTION_PATTERN.search(s):
                sc += 1
            digits = len(re.findall(r"\d", s))
            sc += min(1.0, digits * 0.3)
            if sc > best_score:
                best_score = sc
                best = s
        if best_score <= 0:
            best = sentences[0] if sentences else ""
        # Обрізаємо до читабельної довжини
        if len(best) > 90:
            best = best[:87].rstrip() + "..."
        return best

    # ------------------------------------------------------------------
    # Скоринг кліпів (0..10) — «наскільки цей уривок вартий шортсу»
    # ------------------------------------------------------------------

    def score_clip(self, clip: Dict[str, Any]) -> Dict[str, Any]:
        """
        Оцінює кліп за 5 компонентами:
          speech_density  — щільність мови (слів/сек), мертве повітря карається
          keyword_density — насиченість CS2-термінами
          hook            — хук-слова/питання/цифри на початку
          cleanliness     — відсутність філерів і реклами
          completeness    — кліп починається і закінчується на межі речення
        """
        text = clip.get("text", "")
        dur = max(1.0, float(clip.get("duration", 1.0)))
        words = tokenize(text)

        # 1. Щільність мови: 2.2 сл/сек — ідеал, >3.5 — кабута, <0.8 — мовчання
        wps = len(words) / dur
        if wps >= 1.6:
            speech = 1.0
        elif wps >= 1.0:
            speech = (wps - 0.8) / 0.8
        else:
            speech = max(0.0, wps / 1.6) * 0.3

        # 2. Ключові CS2-слова: 3+ на кліп — повний бал
        kw_set = set(MAP_KEYWORDS) | set(NADE_KEYWORDS) | set(CALLOUT_KEYWORDS)
        kw_hits = sum(1 for t in words if t in kw_set)
        keyword = min(1.0, kw_hits / 3.0)

        # 3. Хук: шукаємо в перших ~8 секундах мови
        segs = clip.get("segments", [])
        clip_start = float(clip.get("start", 0.0))
        head_text = " ".join(
            s.get("text", "") for s in segs
            if (s.get("start", 0.0) - clip_start) <= 8.0
        ) or text[:200]
        hook_hits = sum(1 for rx in self.hook_re if rx.search(head_text))
        if QUESTION_PATTERN.search(head_text):
            hook_hits += 1
        hook = min(1.0, hook_hits / 2.0)

        # 4. Чистота: філери та реклама
        fillers = sum(1 for t in words if t in FILLER_WORDS)
        filler_ratio = fillers / max(1, len(words))
        sponsor_hits = sum(1 for rx in self.sponsor_re if rx.search(text))
        sponsor_ratio = min(1.0, sponsor_hits * 0.25)
        cleanliness = max(0.0, 1.0 - filler_ratio * 4.0 - sponsor_ratio)

        # 5. Завершеність: початок з великої/нового речення, кінець зі знаком
        first_text = segs[0].get("text", "") if segs else text
        last_text = segs[-1].get("text", "") if segs else text
        completeness = 0.0
        if first_text[:1].isupper() or not text:
            completeness += 0.3
        if last_text.endswith(SENTENCE_END):
            completeness += 0.7

        total = round(10.0 * (
            0.22 * speech +
            0.22 * keyword +
            0.24 * hook +
            0.14 * cleanliness +
            0.18 * completeness
        ), 1)

        return {
            "total": total,
            "components": {
                "speech": round(speech, 2),
                "keywords": round(keyword, 2),
                "hook": round(hook, 2),
                "clean": round(cleanliness, 2),
                "complete": round(completeness, 2),
            },
            "keywords_found": kw_hits,
            "words_per_sec": round(wps, 2),
        }

    # ------------------------------------------------------------------
    # Заголовки та описи (Рівень 1: шаблони + контекст)
    # ------------------------------------------------------------------

    NADE_TITLE_UA = {
        "smoke": "Імбовий смок",
        "flash": "Сліпуча флешка",
        "molly": "Вогняний молотов",
        "he": "Фрагова граната",
        "incendiary": "Запальна",
        "decoy": "Фейк-граната",
    }

    NADE_TITLE_RU = {
        "smoke": "Имбовый смок",
        "flash": "Слепящая флешка",
        "molly": "Огненный молотов",
        "he": "Осколочная граната",
        "incendiary": "Зажигательная",
        "decoy": "Фейк-граната",
    }

    def build_title_ru(self, clip: Dict[str, Any]) -> str:
        """Контекстный заголовок на русском языке для СНГ аудитории."""
        ent = self.detect_entities(clip.get("text", ""))
        parts = []
        if ent["nade"]:
            parts.append(self.NADE_TITLE_RU.get(ent["nade"], "Фишка"))
        else:
            parts.append("Секретная фишка")
        if ent["callout"]:
            callout = ent["callout"].replace("_", " ").title()
            parts.append(f"на {callout}")
        if ent["map"]:
            map_name = {"dust2": "Dust 2"}.get(ent["map"], ent["map"].title())
            parts.append(f"({map_name})")
        return " ".join(parts)

    def build_title_ua(self, clip: Dict[str, Any]) -> str:
        """Контекстний заголовок українською з сутностей та хуку."""
        ent = self.detect_entities(clip.get("text", ""))
        parts = []
        if ent["nade"]:
            parts.append(self.NADE_TITLE_UA.get(ent["nade"], "Фішка"))
        else:
            parts.append("Секретна фішка")
        if ent["callout"]:
            callout = ent["callout"].replace("_", " ").title()
            parts.append(f"на {callout}")
        if ent["map"]:
            map_ua = {"dust2": "Dust 2"}.get(ent["map"], ent["map"].title())
            parts.append(f"({map_ua})")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # LLM-верифікація меж розрізів (семантична завершеність думки)
    # ------------------------------------------------------------------

    def llm_boundary_completeness(self, sentences: List[Dict[str, Any]]) -> Dict[int, bool]:
        """
        Питає локальний LLM для кожного кандидата-межі (пауза >= 0.2с):
        чи завершена думка після цього речення. Повертає {idx: True} для
        підтверджених меж. Без LLM повертає {} — сегментер працює на акустиці.
        """
        if not (self.enabled and self.use_llm and self.llm_available()):
            return {}

        # Кандидати: тільки межі з мінімальною паузою (різати «впотік» не можна)
        cands = []
        for i in range(len(sentences) - 1):
            pause = sentences[i + 1]["start"] - sentences[i]["end"]
            if pause >= 0.2:
                cands.append(i)
        if not cands:
            return {}

        print(f"[🧠] LLM перевіряє завершеність думок на {len(cands)} кандидат-межах...")

        system = (
            "Ти — досвідчений відеоредактор, що нарізає монологи на завершені "
            "фрагменти. Відповідай ТІЛЬКИ валідним JSON без пояснень."
        )
        approved: Dict[int, bool] = {}
        CHUNK = 25
        for cs in range(0, len(cands), CHUNK):
            chunk = []
            for i in cands[cs:cs + CHUNK]:
                chunk.append({
                    "id": i,
                    "prev": sentences[i - 1]["text"][:160] if i > 0 else "",
                    "sent": sentences[i]["text"][:200],
                    "next": sentences[i + 1]["text"][:160],
                })
            prompt = (
                "Монолог про CS2 (мова: українська/російська). Для кожного кандидата "
                "перевір речення \"sent\": чи ЗАВЕРШЕНА думка після нього — правило, "
                "приклад або пояснення закінчені і \"next\" починає НОВУ думку.\n"
                "complete=0, якщо \"next\" логічно продовжує те саме (починається з "
                "Тоді/Тому/Але/І/Наприклад/По-перше, або розкриває деталь з \"sent\", "
                "або \"sent\" сам обірваний посеред пояснення).\n"
                f"{json.dumps(chunk, ensure_ascii=False)}\n\n"
                'Поверни JSON-обʼєкт {"complete": [id1, id2, ...]} — ТІЛЬКИ id, де '
                "думка завершена і різати можна без втрат смислу."
            )
            raw = self.llm_generate(system, prompt, max_tokens=6 * len(chunk) + 40)
            if not raw:
                continue
            ids = None
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    ids = parsed.get("complete") or parsed.get("ids") or parsed.get("results")
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    try:
                        ids = json.loads(m.group(0)).get("complete")
                    except json.JSONDecodeError:
                        ids = None
            if not ids:
                continue
            for x in ids:
                try:
                    approved[int(x)] = True
                except (ValueError, TypeError):
                    continue

        if approved:
            print(f"[🧠] LLM підтвердив {len(approved)}/{len(cands)} меж як завершені думки")
        else:
            print("[🧠] LLM не відповів на перевірку меж — ріжемо за акустикою")
        return approved

    # ------------------------------------------------------------------
    # Публічний API: аналіз набору кліпів
    # ------------------------------------------------------------------

    def analyze_clips(self, clips: List[Dict[str, Any]], segments: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Повний цикл аналізу:
          1. скоринг кожного кліпу
          2. відсіювання слабких (нижче min_score), якщо є з що відсіювати
          3. опційний LLM-ранкінг та LLM-заголовки
          4. сортування за скором (порядок таймкодів зберігається окремо)
        """
        if not self.enabled or not clips:
            return clips

        for c in clips:
            s = self.score_clip(c)
            c["score"] = s["total"]
            c["score_detail"] = s["components"]
            c["hook_line"] = self.find_hook_line(c)
            c["entities"] = self.detect_entities(c.get("text", ""))
            c["title_ua"] = self.build_title_ua(c)
            c["title_ru"] = self.build_title_ru(c)

        # Відсіюємо слабкі, але лише якщо після фільтра хоч щось лишається
        strong = [c for c in clips if c["score"] >= self.min_score]
        dropped = len(clips) - len(strong)
        if dropped and strong:
            print(f"[🧠] Контекст-скоринг: відсіяно {dropped} слабких кліпів (score < {self.min_score})")
            clips = strong
        elif dropped:
            print(f"[🧠] Всі кліпи нижче порігу {self.min_score} — залишаю найкращий")
            clips = [max(clips, key=lambda c: c["score"])]

        # Обмеження кількості (найкращі за скором)
        if self.max_clips and len(clips) > self.max_clips:
            clips.sort(key=lambda c: (-c["score"], c["start"]))
            clips = clips[:self.max_clips]
            clips.sort(key=lambda c: c["start"])
            print(f"[🧠] Залишено топ-{self.max_clips} кліпів за скорингом")

        # Ollama: LLM-заголовки/описи (не блокуємо пайплайн при фейлі)
        if self.llm_available():
            try:
                self._llm_enhance_clips(clips)
            except Exception as e:
                print(f"[🧠] LLM-покращення пропущено: {e}")

        # Фінальний порядок — за таймкодом
        clips.sort(key=lambda c: c["start"])
        for i, c in enumerate(clips, 1):
            c["index"] = i
        return clips

    def _llm_enhance_clips(self, clips: List[Dict[str, Any]]) -> None:
        """Просить локальний LLM оцінити віральність та написати заголовки (чанками)."""
        print(f"[🧠] Підключено локальний LLM ({self.llm_model}) для контексту")
        system = (
            "Ти — експерт з вірусних коротких відео про CS2 (TikTok/Shorts, українською). "
            "Відповідай ТІЛЬКИ валідним JSON без markdown-огорож."
        )

        CHUNK = 8
        applied_total = 0
        failed_chunks = 0

        for chunk_start in range(0, len(clips), CHUNK):
            chunk = clips[chunk_start:chunk_start + CHUNK]
            payload = [
                {"id": chunk_start + i, "text": c.get("text", "")[:400]}
                for i, c in enumerate(chunk)
            ]
            prompt = (
                "Ось транскрипти кліпів CS2 (мова: українська/російська/англійська):\n"
                f"{json.dumps(payload, ensure_ascii=False)}\n\n"
                "Для КОЖНОГО кліпу поверни JSON-ОБʼЄКТ:\n"
                '{"clips": [{"id": 0, "viral_score": 1-10, "title": "заголовок до 60 символів, цепляющий, українською"}]}\n'
                'Поле "id" обовʼязкове в кожному обʼєкті та дорівнює id з входу. '
                "Лише JSON, без пояснень."
            )
            # Просимо лише заголовок+скор: описи генеруємо локально з хук-цитати (швидше і точніше)
            raw = self.llm_generate(system, prompt, max_tokens=45 * len(chunk) + 80)
            if not raw:
                failed_chunks += 1
                if failed_chunks >= 2:
                    print("[🧠] LLM двічі не відповів — подальші чанки пропускаю")
                    break
                continue

            # Парсинг: об'єкт {"clips":[...]} або чистий масив
            items = None
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    items = parsed.get("clips") or parsed.get("items") or parsed.get("results")
                elif isinstance(parsed, list):
                    items = parsed
            except json.JSONDecodeError:
                m = re.search(r"\[.*\]", raw, re.DOTALL)
                if m:
                    try:
                        items = json.loads(m.group(0))
                    except json.JSONDecodeError:
                        items = None
            if not items:
                failed_chunks += 1
                continue

            for item_index, item in enumerate(items):
                try:
                    # Модель іноді губить "id" — тоді використовуємо порядок у масиві
                    idx = int(item.get("id", chunk_start + item_index))
                    if 0 <= idx < len(clips):
                        title = str(item.get("title", "")).strip()
                        caption = str(item.get("caption", "")).strip()
                        viral = float(item.get("viral_score", 0))
                        if title and 8 <= len(title) <= 80:
                            clips[idx]["title_llm"] = title
                        if caption and len(caption) <= 400:
                            clips[idx]["caption_ua"] = caption
                        if 0 < viral <= 10:
                            clips[idx]["llm_viral_score"] = round(viral, 1)
                        applied_total += 1
                except (ValueError, TypeError):
                    continue

        # Розумне злиття заголовків: контекстні (карта/граната/позиція) — точні,
        # LLM — креативні. Коли є фактичні сутності, точність важливіша.
        for c in clips:
            llm_t = c.get("title_llm")
            ctx_t = c.get("title_ua", "")
            ent = c.get("entities", {})
            has_facts = bool(ent.get("map") or ent.get("nade"))
            if llm_t and not has_facts:
                c["title_ua"] = llm_t
            # інакше лишаємо контекстний заголовок

        if applied_total:
            print(f"[🧠] LLM покращив {applied_total}/{len(clips)} кліпів (заголовки + скоринг віральності)")
        else:
            print("[🧠] LLM не відповів — використовуються локальні заголовки")
