"""
Semantic unit engine for the MUSORZALIV clipper.

Контент у CS2-відео живе не реченнями, а одиницями, кожна з яких довша за
одне речення:
    - ФІШКА:   порада -> як зробити -> результат;
    - МІФ:     теза міфа -> перевірка/тест -> ВЕРДИКТ (правда/фейк);
    - ПРАВИЛО: правило -> пояснення -> приклад.

Цей модуль шукає в транскрипті міф-юніти (за маркерами тези та вердикту) і
планує кліпи так, щоб вердикт гарантовано потрапив у кадр: юніт вміщується
в 28с — беремо ціликом; ні — ділимо на частини з кліфгенгером (Ч.1 закінчується
на питанні/найцікавішому місці, Ч.2 містить вердикт).
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from pipeline.sentences import sentence_windows

# ---------------------------------------------------------------- маркери

# Речення, яке ЗАПУСКАЄ міф/перевірку (висока точність, щоб не чіпати звичайні фішки)
MYTH_INTRO_PATTERNS = [
    r"(?:перев|пров)[еіи]р(?:им|ім|яем|яєм|имо|ите|ка)\b",   # перевірим/проверим/проверяем/проверка
    r"\bтест(?:уєм|уємо|уем|уемо|ируем|ує|уем)\b",
    r"(?:міф|миф)\w*\s*(?:№|номер)?\s*\d",                   # міф №1, миф 5, миф номер 3
    r"правда\s+(?:чи|или|або)\s",                            # правда чи нет
    r"\bчи\s+(?:можна|працює|реально|есть|існує|буває)",
    r"(?:работает|працює)\s+ли\b",
    r"(?:говор(?:ят|ять|ит|ить)|каж(?:ут|уть|е)|дума(?:ют|ють)|вважають)[^.!?]{0,40}\b(?:что|що)\b",
    r"(?:разоблачаем|разоблачаю|разоблачение|викрива(?:єм|ю)|розвінчу(?:єм|вали|ю))",
    r"фейк[- ]момент",                                        # жанр "фейк-моменты" (складне слово, не голе "фейк")
    r"(?:сейчас|зараз)\s+(?:провер|перевір|узна|дізна)",
    r"(?:узна(?:ем|єм)|дізнаємось)[^.!?]{0,30}(?:правду|истину)",
    r"в комментариях (?:спорят|пишут|говорят)",
]

# Речення-ВЕРДИКТ (результат перевірки). Питання вердиктом не рахується.
VERDICT_PATTERNS = [
    r"\bвердикт",
    r"(?:міф|миф)\w*[^.!?]{0,40}\b(?:спростован|розвінчан|знищ|підтвердж|разоблачен|разоблачён|опроверг|подтверд|неправда|неправдив|врань[её]|фейк|бред|чушь)",
    r"(?:оказал[ося]+|виявилось|виявилося|виявився|з'ясувалось|зясувалось|опинилось|опинився)",
    r"\b(?:итог|итоге|підсумок|висновок|вывод)\b",
    r"\bспойлер\b",
    r"(?:спростован|розвінчан|разоблачил|знищили)",
    r"(?:міф|миф)\s*(?:—|-|–)?\s*(?:правда|неправда)\s*[.!?]?\s*$",
    # жанр "проверка моментов/фейк-моментов": вердикт підбивається оцінкою
    r"получает\s+оценку",
    r"оценк[ауие]\s*[:\-—]?\s*(?:фейк|реальн|правд)",
    r"(?:в сухом остатке|сухой остаток)",
]

# Перехід до тесту: "смотрим что будет", "перевіримо на практиці"
TEST_TRANSITION_RE = re.compile(
    r"(?:смотрим|дивимось|що\s+буде|что\s+будет|спробуєм|попробуем|результат|перевір(?:им|имо)|провер(?:им|яем))",
    re.IGNORECASE,
)

# Перехід до нової новини/теми (новинні відео): межі новинних юнітів
NEWS_TRANSITION_PATTERNS = [
    r"(?:наступн\w+|следующ\w+|нов\w+)\s+новин\w*",           # наступна новина / новая новость
    r"(?:перв\w+|перш\w+|друг\w+|друг\w+|трет\w+|четверт\w+|п'ят\w+)\s+(?:новин|новост)\w*",
    r"новин\w*\s*(?:№|номер)?\s*\d",                            # новина №2
    r"новост\w*\s*(?:№|номер)?\s*\d",
    r"(?:і еще|и еще|і ще|ще)\s+(?:одн\w+|дв\w+)\s+(?:новин|новост)\w*",
    r"переходим(?:о)?\s+(?:до|к)\s+(?:новин|новост|наступн\w+|следующ\w+)",
    r"(?:у інших|в других|во всех|у всіх)\s+(?:новинах|новостях|новин)",
    r"(?:також повідомля|также сообща|повідомляють|сообщается)",
    r"(?:по новостях|по новинам|новинний дайджест|новостной дайджест)",
    r"коротко\s+про\s+(?:інше|другое|решту)",
    r"продовжуючи\s+(?:новинну|новостную)\s+тему",
]

NEWS_TRANSITION_RE = [re.compile(p, re.IGNORECASE) for p in NEWS_TRANSITION_PATTERNS]


def looks_like_news_transition(text: str) -> bool:
    """True, якщо речення — перехід до нової новини/теми (межа юніта)."""
    if not text:
        return False
    return any(rx.search(text) for rx in NEWS_TRANSITION_RE)


def detect_news_units(sentences: List[Dict[str, Any]], lo: int, hi: int) -> List[Dict[str, int]]:
    """
    Новинні юніти: кліп = одна новина від заголовка до кінця.

    Межі — переходи («наступна новина», «переходимо до новин»...). Голий перехід
    з менше ніж 6 слів ("Наступна новина.") пропускається — кліп стартує з
    заголовка новини. Повертає [{"first_idx", "last_idx"}].
    """
    units: List[Dict[str, int]] = []
    i = lo
    while i <= hi:
        if looks_like_news_transition(sentences[i]["text"]):
            start = i
            if len(sentences[i]["text"].split()) < 6 and i + 1 <= hi:
                start = i + 1  # голий перехід не тягнемо в кліп — стартуємо з заголовка
            j = i + 1
            while j <= hi and not looks_like_news_transition(sentences[j]["text"]):
                j += 1
            last = j - 1
            if last >= start:
                units.append({"first_idx": start, "last_idx": last})
            i = j
            continue
        i += 1
    return units

MYTH_INTRO_RE = [re.compile(p, re.IGNORECASE) for p in MYTH_INTRO_PATTERNS]
VERDICT_RE = [re.compile(p, re.IGNORECASE) for p in VERDICT_PATTERNS]


def looks_like_myth_intro(text: str) -> bool:
    """True, якщо речення запускає міф/перевірку."""
    if not text:
        return False
    return any(rx.search(text) for rx in MYTH_INTRO_RE)


def looks_like_verdict(text: str) -> bool:
    """True, якщо речення — вердикт. Питання ('...чи правда?') вердиктом не рахується."""
    if not text or text.rstrip().endswith("?"):
        return False
    return any(rx.search(text) for rx in VERDICT_RE)


def is_cliffhanger(text: str) -> bool:
    """Речення, на якому можна оборвати частину 1: питання або перехід до тесту."""
    if text.rstrip().endswith("?"):
        return True
    return bool(TEST_TRANSITION_RE.search(text))


# ---------------------------------------------------------------- детекція юнітів

def detect_units(
    sentences: List[Dict[str, Any]],
    lo: int,
    hi: int,
    max_unit_span: float = 90.0,
) -> List[Dict[str, int]]:
    """
    Шукає міф-юніти в діапазоні речень [lo..hi].

    Юніт = {first_idx: теза міфа, last_idx: останнє речення вердикту}.
    Вердикт шукається в межах max_unit_span секунд від тези (міф довший за
    ~1.5 хвилини все одно не опублікуєш зв'язно). Юніт без вердикту — не юніт.
    """
    def span(a: int, b: int) -> float:
        return sentences[b]["end"] - sentences[a]["start"]

    units: List[Dict[str, int]] = []
    i = lo
    while i <= hi:
        if looks_like_myth_intro(sentences[i]["text"]):
            verdict_idx: Optional[int] = None
            j = i + 1
            while j <= hi and span(i, j) <= max_unit_span:
                if looks_like_myth_intro(sentences[j]["text"]) and verdict_idx is not None:
                    break  # вердикт уже був — далі починається наступний міф
                # інтро-речення ДО вердикту — частина тези поточного міфа ("Говорят, что...")
                if looks_like_verdict(sentences[j]["text"]):
                    verdict_idx = j  # тримаємо ОСТАННІЙ вердикт у межах бюджету
                j += 1
            if verdict_idx is not None:
                units.append({"first_idx": i, "last_idx": verdict_idx})
                i = verdict_idx + 1
                continue
        i += 1
    return units


# ---------------------------------------------------------------- планування частин

def plan_unit_parts(
    sentences: List[Dict[str, Any]],
    u0: int,
    v: int,
    *,
    min_dur: float,
    floor_dur: float,
    max_dur: float,
    hard_max: float,
) -> List[Tuple[int, int]]:
    """
    План кліпів (пари індексів речень) для одного міф-юніту [u0..v].

    - Вміщується в hard_max -> один кліп (теза + перевірка + вердикт ціликом).
    - Ні -> хвостова частина з вердиктом максимально довга, голова ріжеться
      звичайними вікнами; якщо голова коротша за floor — шукаємо кліфгенгер-межу.
    """
    def span(a: int, b: int) -> float:
        return sentences[b]["end"] - sentences[a]["start"]

    if span(u0, v) <= hard_max:
        return [(u0, v)]

    # Хвіст (містить вердикт) беремо максимально довгим, але в межах max_dur
    k = v
    while k - 1 > u0 and span(k - 1, v) <= max_dur:
        k -= 1
    head_hi = k - 1
    if head_hi < u0:
        return [(u0, v)]

    if span(u0, head_hi) >= floor_dur:
        # Голова ріжеться звичайними вікнами завершених думок + хвіст з вердиктом
        parts = sentence_windows(
            sentences, u0, head_hi,
            min_dur=min_dur, floor_dur=floor_dur, max_dur=max_dur,
        )
        parts.append((k, v))
        return parts

    # Голова коротша за floor: шукаємо одну кліфгенгер-межу на 2 частини
    b = pick_cliffhanger(sentences, u0, v, floor_dur=floor_dur, max_dur=max_dur)
    if b is not None:
        return [(u0, b), (b + 1, v)]
    return [(k, v)]


def pick_cliffhanger(
    sentences: List[Dict[str, Any]],
    u0: int,
    v: int,
    *,
    floor_dur: float,
    max_dur: float,
) -> Optional[int]:
    """
    Межа b у [u0..v-1], щоб [u0..b] і [b+1..v] обидві влазили в max_dur і були
    >= floor_dur. Пріоритет: питання (+3), перехід до тесту (+2), сильна межа (+1).
    """
    def span(a: int, b: int) -> float:
        return sentences[b]["end"] - sentences[a]["start"]

    best: Optional[int] = None
    best_score = -1
    for b in range(u0, v):
        d1 = span(u0, b)
        d2 = span(b + 1, v)
        if d1 < floor_dur or d2 < floor_dur or d1 > max_dur or d2 > max_dur:
            continue
        score = 0
        if sentences[b]["text"].rstrip().endswith("?"):
            score += 3
        if TEST_TRANSITION_RE.search(sentences[b]["text"]):
            score += 2
        if sentences[b].get("strong"):
            score += 1
        if score > best_score:
            best_score = score
            best = b
    return best


def complete_myth_window(
    sentences: List[Dict[str, Any]],
    first_idx: int,
    last_idx: int,
    *,
    min_dur: float,
    floor_dur: float,
    max_dur: float,
    hard_max: float,
    max_unit_span: float = 90.0,
) -> Optional[List[Tuple[int, int]]]:
    """
    Гарантує ВЕРДИКТ у міф-вікні [first_idx..last_idx].

    Шукає останній вердикт від початку вікна (вперед до max_unit_span):
    - вердикт пізніше кінця вікна -> вікно дорощується до нього;
    - у вікні після вердикту є зайві речення (початок наступного моменту)
      -> хвіст обрізається по вердикту;
    - теза (intro) на 1-2 речення раніше вікна -> старт підтягується до неї.
    Повертає план частин (plan_unit_parts) або None, якщо вердикт не знайдено.
    """
    def span(a: int, b: int) -> float:
        return sentences[b]["end"] - sentences[a]["start"]

    if not (0 <= first_idx <= last_idx < len(sentences)):
        return None

    # Останній вердикт у межах [first_idx .. first_idx + max_unit_span]
    verdict_idx: Optional[int] = None
    for j in range(first_idx, len(sentences)):
        if span(first_idx, j) > max_unit_span:
            break
        if looks_like_verdict(sentences[j]["text"]):
            verdict_idx = j
    if verdict_idx is None:
        return None

    # Теза міфа: усередині вікна або трохи раніше (Qwen міг стартувати з тесту)
    intro_lo = max(0, first_idx - 12)
    intro_idxs = [m for m in range(intro_lo, verdict_idx) if looks_like_myth_intro(sentences[m]["text"])]
    u0 = min(intro_idxs) if intro_idxs else first_idx
    if u0 > verdict_idx:
        return None

    return plan_unit_parts(
        sentences, u0, verdict_idx,
        min_dur=min_dur, floor_dur=floor_dur, max_dur=max_dur, hard_max=hard_max,
    )


# ---------------------------------------------------------------- резолв номерів речень

def norm_text(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def match_sentence_idx(
    sentences: List[Dict[str, Any]],
    idx_hint: int,
    echo_text: str,
    is_end: bool,
) -> Optional[int]:
    """
    Уточнює номер речення за ехо-фразою від LLM (перші/останні слова речення).
    Шукає в діапазоні ±3 від підказки; так виправляється збій нумерації
    (0-based проти 1-based) та пропущені речення.
    """
    e = norm_text(echo_text)
    if not e:
        return None
    for d in (0, -1, 1, -2, 2, -3, 3):
        i = idx_hint + d
        if 0 <= i < len(sentences):
            t = norm_text(sentences[i]["text"])
            if not t:
                continue
            if e in t or t in e or (not is_end and t.startswith(e)) or (is_end and t.endswith(e)):
                return i
    return None


def resolve_sent_window(
    sentences: List[Dict[str, Any]],
    item: Dict[str, Any],
) -> Optional[Tuple[int, int]]:
    """
    Витягує з відповіді LLM вікно в номерах речень: start_sent/end_sent
    (1-based) з ехо-фразами start_text/end_text для самокорекції.
    Повертає (first_idx, last_idx) або None.
    """
    idx_keys_start = ("start_sent", "sent_start", "start_sentence", "first_sentence", "sentence_start")
    idx_keys_end = ("end_sent", "sent_end", "end_sentence", "last_sentence", "sentence_end")

    s_val = next((int(item[k]) for k in idx_keys_start if k in item and isinstance(item[k], (int, float)) and not isinstance(item[k], bool)), None)
    e_val = next((int(item[k]) for k in idx_keys_end if k in item and isinstance(item[k], (int, float)) and not isinstance(item[k], bool)), None)
    if s_val is None or e_val is None:
        return None

    first_hint = max(0, s_val - 1)  # 1-based -> 0-based
    last_hint = max(0, e_val - 1)

    echo_start = str(item.get("start_text") or item.get("first_text") or "").strip()
    echo_end = str(item.get("end_text") or item.get("last_text") or "").strip()

    first_idx = match_sentence_idx(sentences, first_hint, echo_start, is_end=False) if echo_start else None
    if first_idx is None:
        first_idx = min(first_hint, len(sentences) - 1)
    last_idx = match_sentence_idx(sentences, last_hint, echo_end, is_end=True) if echo_end else None
    if last_idx is None:
        last_idx = min(max(last_hint, first_idx), len(sentences) - 1)

    if first_idx > last_idx:
        first_idx, last_idx = last_idx, first_idx
    return (first_idx, last_idx)


def merge_windows_with_units(
    sentences: List[Dict[str, Any]],
    lo: int,
    hi: int,
    *,
    min_dur: float,
    floor_dur: float,
    max_dur: float,
    hard_max: float,
) -> List[Tuple[int, int]]:
    """
    Повне розбиття діапазону [lo..hi]: міф-юніти плануються окремо (з гарантією
    вердикту), решта — звичайними вікнами завершених думок. Вікна йдуть по
    порядку без перетинів.
    """
    def span(a: int, b: int) -> float:
        return sentences[b]["end"] - sentences[a]["start"]

    units = detect_units(sentences, lo, hi)
    windows: List[Tuple[int, int]] = []
    cur = lo
    for u in units:
        if u["first_idx"] < cur:
            continue  # страховка від перетину
        if u["first_idx"] > cur:
            windows.extend(sentence_windows(
                sentences, cur, u["first_idx"] - 1,
                min_dur=min_dur, floor_dur=floor_dur, max_dur=max_dur,
            ))
        windows.extend(plan_unit_parts(
            sentences, u["first_idx"], u["last_idx"],
            min_dur=min_dur, floor_dur=floor_dur, max_dur=max_dur, hard_max=hard_max,
        ))
        cur = u["last_idx"] + 1
    if cur <= hi:
        windows.extend(sentence_windows(
            sentences, cur, hi,
            min_dur=min_dur, floor_dur=floor_dur, max_dur=max_dur,
        ))
    return windows
