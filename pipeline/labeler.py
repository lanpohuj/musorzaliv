"""
Interest labeling of the transcript via Qwen micro-batches (WOW-CUTTER Step 1).

Замість одного мега-промпту «знайди вірусні моменти» (яке 3B-модель тягне
погано) транскрипт ріжеться на батчі по ~20 речень, і модель виконує просту
рокля: для кожного речення назвати роль (hook/setup/action/result/verdict/
filler/offtopic) та оцінку цікавості 0-10.

Результат — «карта цікавості» відео, яку детермінований скоринг
(pipeline/ranker.py) перетворює на відбір кліпів. Карта кешується в temp/,
тому повторні запуски того самого відео не ганяють LLM знову.

Без LLM працює евристична розмітка (регулярки за жанром CS2) — того самого
формату, тож рангер не розрізняє джерело.
"""

import hashlib
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional

from pipeline.units import looks_like_myth_intro, looks_like_verdict

ROLES = ("hook", "setup", "action", "result", "verdict", "filler", "offtopic")

SYSTEM_PROMPT = (
    "Ты размечаешь транскрипт CS2-видео для TikTok-нарезки. Для КАЖДОГО пронумерованного "
    "предложения верни JSON-массив объектов вида {\"n\": номер, \"role\": роль, \"interest\": 0-10}. "
    "Роли: hook (захватывающее начало: вопрос, интрига, провокация), setup (контекст/подводка), "
    "action (конкретное действие: как делать, что нажимать), result (результат/совет), "
    "verdict (вердикт мифа/проверки), filler (вода, пустые фразы), offtopic (не про игру/реклама). "
    "interest — насколько предложение звучит интересно зрителю (0-вода, 10-цепляет мгновенно). "
    "Верни ТОЛЬКО валидный JSON-массив, по одному объекту на каждое предложение."
)

# Філери, що вбивають темп (RU + UA)
FILLER_RE = re.compile(
    r"\b(ну|короче|типу|типа|значит|значить|как бы|як би|вот|в принципі|в принципе|"
    r"короче кажучи|якось|вроде|мм+|е+м+|нуу+|так скажем|так сказати|якби)\b",
    re.IGNORECASE,
)

# Провокації та хук-маркери
HOOK_RE = re.compile(
    r"(смотри|смотрите|дивись|дивіться|никогда|ніколи|никто|ніхто|секрет|тайн|прихован|"
    r"імба|imba|лайфхак|ошибк|помилк|фейк|міф|миф|провер|перевір|топ[- ]?\d|правило|"
    r"все делают|всі роблять|вы не знали|ви не знали|шок|жестк|жорстк|убийствен|смертельн)",
    re.IGNORECASE,
)

# Дієслова дії (action)
ACTION_RE = re.compile(
    r"(зроби|роби|кида|кидай|став|стави|стриба|залаз|заход|натисн|тисн|клікн|приціл|"
    r"прицел|зажми|затисн|откр|відкр|поверн|обвед|запусти|перевір|провер)",
    re.IGNORECASE,
)

RESULT_RE = re.compile(
    r"(в итоге|итог|результат|висновок|підсумок|ось так|вот так|готово|виходить|выходит|"
    r"таким образом|чудово|отлично|идеально|ідеально|працює|работает|спрацював)",
    re.IGNORECASE,
)

CS2_TERMS_RE = re.compile(
    r"(\bсмок\w*|\bsmoke|\bфлеш\w*|\bflash|\bмолотов\w*|\bmolly|\bгранат\w*|\bhe\b|"
    r"розкид\w*|раскид\w*|\bбуст\w*|\bboost|\bджамптроу|jumpthrow|\bплент\w*|\bsite|"
    r"\bмид\b|\bmid\b|\bлонг\b|\blong\b|\bшорт\b|\bshort\b|\bокно\b|\bwindow\b|"
    r"банан\w*|banana|рамп\w*|ramp|конектор\w*|connector|Mirage|Inferno|Anubis|Ancient|"
    r"Nuke|Dust|Vertigo|Overpass|Train|\braунд\w*|\bround|\bеко\b|\beco\b|\bфорс\b|\bforce|"
    r"прицел|приціл|\bфлик\w*|\bflick|спрей|spray|\bхедшот\w*|headshot|\bпиксель|pixel)",
    re.IGNORECASE,
)

DIGIT_RE = re.compile(r"\d")


def _looks_hook(text: str) -> bool:
    t = text.strip()
    if t.endswith("?"):
        return True
    if looks_like_myth_intro(t):
        return True
    if HOOK_RE.search(t):
        return True
    if re.match(r"^\s*(топ|top)[- ]?\d", t, re.IGNORECASE):
        return True
    return False


def _words(text: str) -> List[str]:
    return [w for w in re.split(r"\s+", text.strip()) if w]


def heuristic_labels(sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Розмітка без LLM: регулярки за жанром CS2. Той самий формат, що й LLM-розмітка."""
    labels: List[Dict[str, Any]] = []
    for s in sentences:
        t = s["text"]
        words = _words(t)
        n_words = max(1, len(words))
        filler_hits = len(FILLER_RE.findall(t))

        if looks_like_verdict(t):
            role = "verdict"
        elif _looks_hook(t):
            role = "hook"
        elif filler_hits >= 2 or n_words <= 3:
            role = "filler"
        elif RESULT_RE.search(t):
            role = "result"
        elif ACTION_RE.search(t):
            role = "action"
        else:
            role = "setup"

        interest = 5.0
        if role == "hook":
            interest += 3.0
        if role in ("verdict", "result"):
            interest += 2.5
        if role == "filler":
            interest -= 3.0
        if DIGIT_RE.search(t):
            interest += 1.0
        if CS2_TERMS_RE.search(t):
            interest += 1.0
        interest = max(0.0, min(10.0, interest))

        labels.append({"role": role, "interest": round(interest, 1)})
    return labels


def _parse_label_batch(raw: str, expected_n: int) -> Optional[Dict[int, Dict[str, Any]]]:
    """Розбирає відповідь Qwen на батч: [{'n': 1, 'role': ..., 'interest': ...}, ...]."""
    if not raw:
        return None
    clean = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw.strip(), flags=re.MULTILINE)
    clean = re.sub(r",\s*([\]}])", r"\1", clean)
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        m = re.search(r"\[\s*\{.*\}\s*\]", clean, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, list):
        return None

    out: Dict[int, Dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        n = item.get("n", item.get("id", item.get("index")))
        if not isinstance(n, (int, float)):
            continue
        n = int(n)
        if not (1 <= n <= expected_n):
            continue
        role = str(item.get("role", "setup")).strip().lower()
        if role not in ROLES:
            role = "setup"
        try:
            interest = float(item.get("interest", 5.0))
        except (TypeError, ValueError):
            interest = 5.0
        interest = max(0.0, min(10.0, interest))
        out[n] = {"role": role, "interest": interest}
    # Батч вважається валідним, якщо розмічено хоча б 70% речень
    if len(out) < expected_n * 0.7:
        return None
    return out


def cache_file_path(temp_dir: str, video_stem: str, sentences: List[Dict[str, Any]]) -> Optional[str]:
    """Шлях кешу розмітки; None, якщо ключ будувати не з чого."""
    texts = "\n".join(s["text"] for s in sentences)
    if not texts.strip():
        return None
    key = hashlib.md5(texts.encode("utf-8")).hexdigest()[:10]
    safe_stem = re.sub(r"[^\w\-]+", "_", video_stem)[:40]
    return os.path.join(temp_dir, f"labels_{safe_stem}_{key}.json")


def load_labels_cache(path: Optional[str], expected_len: int) -> Optional[List[Dict[str, Any]]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        labels = data.get("labels")
        if isinstance(labels, list) and len(labels) == expected_len:
            return labels
    except Exception:
        pass
    return None


def save_labels_cache(path: Optional[str], labels: List[Dict[str, Any]]) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"labels": labels}, f, ensure_ascii=False)
    except Exception:
        pass


def label_transcript(
    sentences: List[Dict[str, Any]],
    llm_call: Optional[Callable[[str, str], Optional[str]]] = None,
    *,
    batch_size: int = 20,
    cache_path: Optional[str] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Розмічує речення: {"role": str, "interest": float} на кожне.

    llm_call(system, user) -> str — інжектований виклик Qwen (None = евристика).
    progress(done_batches, total_batches) — для прогресу в TUI.
    """
    cached = load_labels_cache(cache_path, len(sentences))
    if cached:
        print(f"[🏷] Розмітка інтересу взята з кешу ({len(cached)} речень)")
        return cached

    if not sentences:
        return []

    if llm_call is None:
        labels = heuristic_labels(sentences)
        save_labels_cache(cache_path, labels)
        return labels

    labels: List[Optional[Dict[str, Any]]] = [None] * len(sentences)
    batches = [range(i, min(i + batch_size, len(sentences))) for i in range(0, len(sentences), batch_size)]
    total = len(batches)
    print(f"[🏷] Розмітка інтересу транскрипту: {len(sentences)} речень, {total} батчів...")

    for b_i, rng in enumerate(batches, 1):
        idxs = list(rng)
        chunk = [f"[{k + 1}] {sentences[i]['text']}" for k, i in enumerate(idxs)]
        user = (
            "Розмить кожне речення (роль + interest 0-10):\n\n"
            + "\n".join(chunk)
            + f"\n\nВерни JSON-масив з {len(idxs)} об'єктами: n, role, interest."
        )

        parsed = None
        for attempt in (1, 2):
            raw = llm_call(SYSTEM_PROMPT, user)
            parsed = _parse_label_batch(raw or "", len(idxs))
            if parsed is not None:
                break
            if attempt == 1:
                user = user + "\n\nУВАГА: попередня відповідь була невалідною. Лише JSON-масив, кожен елемент: {\"n\": номер, \"role\": ..., \"interest\": число}."

        fallback = heuristic_labels([sentences[i] for i in idxs])
        for k, i in enumerate(idxs):
            if parsed and (k + 1) in parsed:
                labels[i] = parsed[k + 1]
            else:
                labels[i] = fallback[k]

        if progress:
            progress(b_i, total)

    result = [l or {"role": "setup", "interest": 5.0} for l in labels]
    save_labels_cache(cache_path, result)
    print(f"[✓] Розмітка завершена: {len(result)} речень")
    return result
