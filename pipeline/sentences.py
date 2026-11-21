"""
Sentence-boundary engine for the MUSORZALIV clipper.

Whisper (word_timestamps=True) дає таймстампи на кожне слово. Цей модуль
збирає потік слів у речення та прив'язує вікно кліпу від LLM до реальних
меж речень, щоб кліп ніколи не починався і не обривався на півслові
("обрубок").

Правила:
- Межа речення = слово з крапкою/«!»/«?» (сильна) або пауза довша за
  soft_pause секунд (м'яка — страховка, коли Whisper губить пунктуацію).
- Кінець кліпу снапиться до НАЙБЛИЖЧОГО кінця речення від таймкоду LLM
  (розширення обмежене max_extend), стеля — hard_max.
- Хвіст кліпу мусить бути завершеним реченням: заборонено обриватись на
  висячому сполучнику/прийменнику ("і", "але", "що", "тому що", ...).
- Кліпи коротші за floor_dur відкидаються; ціль — набрати min_dur.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

# Слова, на яких речення завершитись не може (RU + UA транскрипти)
DANGLING_TAILS = frozenset({
    # союзи RU
    "и", "но", "а", "или", "либо", "что", "чтобы", "потому", "поэтому",
    "который", "которая", "которые", "которое", "если", "когда", "же",
    "ведь", "то", "не", "ни", "ли", "б", "бы",
    # прийменники RU
    "в", "во", "на", "за", "до", "по", "из", "от", "у", "о", "об", "про",
    "для", "под", "над", "без", "через", "между", "с", "со", "при",
    # UA
    "і", "й", "але", "що", "щоб", "бо", "тому", "який", "яка", "які",
    "яке", "якщо", "коли", "або", "та", "ні", "ж", "би",
    "від", "з", "із", "зі", "під", "між",
})

_STRONG_ENDS = (".", "!", "?")
_STRIP_CHARS = ".,!?;:—–-\"'«»()[]„“…"


def _tail_word(text: str) -> str:
    words = text.split()
    if not words:
        return ""
    return words[-1].strip().strip(_STRIP_CHARS).lower()


def is_tail_complete(text: str) -> bool:
    """True, якщо останнє слово тексту — не висячий сполучник/прийменник."""
    tail = _tail_word(text)
    if not tail:
        return False
    return tail not in DANGLING_TAILS


def build_sentences(
    segments: List[Dict[str, Any]],
    soft_pause: float = 0.6,
    max_run_on: float = 10.0,
) -> List[Dict[str, Any]]:
    """
    Групує word-таймстампи Whisper у речення:
    {"start", "end", "text", "words", "strong"}.

    Сильна межа — розділовий знак у кінці слова. М'яка — пауза між словами
    довша за soft_pause (на випадок, коли Whisper губить крапки). Суцільний
    монолог без пауз і пунктатури примусово ріжеться кожні ~max_run_on секунд
    у найближчому природному мікропроваллі.
    """
    flat: List[Dict[str, Any]] = []
    for seg in segments:
        words = seg.get("words") or []
        if not words:
            txt = (seg.get("text") or "").strip()
            if txt:
                flat.append({
                    "word": txt,
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "seg_end": True,
                })
            continue
        for wi, w in enumerate(words):
            wt = (w.get("word") or "").strip()
            if not wt:
                continue
            flat.append({
                "word": wt,
                "start": float(w["start"]),
                "end": float(w["end"]),
                "seg_end": wi == len(words) - 1,
            })

    sentences: List[Dict[str, Any]] = []
    cur: List[Dict[str, Any]] = []

    def flush(strong: bool) -> None:
        if not cur:
            return
        sentences.append({
            "start": cur[0]["start"],
            "end": cur[-1]["end"],
            "text": " ".join(x["word"] for x in cur).strip(),
            "words": list(cur),
            "strong": strong,
        })
        cur.clear()

    for i, x in enumerate(flat):
        cur.append(x)
        nxt = flat[i + 1] if i + 1 < len(flat) else None
        gap = (nxt["start"] - x["end"]) if nxt else None
        if x["word"].rstrip().endswith(_STRONG_ENDS):
            flush(True)
        elif gap is None or gap > soft_pause:
            flush(False)
        elif (x["end"] - cur[0]["start"]) > max_run_on and (gap > 0.25 or x["seg_end"]):
            flush(False)

    flush(False)

    # Сверхкороткі осколки ("Так.", "Ну.") приклеюємо до попереднього речення
    merged: List[Dict[str, Any]] = []
    for s in sentences:
        tiny = len(s["words"]) < 2 or (s["end"] - s["start"]) < 0.5
        if tiny and merged:
            p = merged[-1]
            p["end"] = s["end"]
            p["text"] = (p["text"] + " " + s["text"]).strip()
            p["words"].extend(s["words"])
            p["strong"] = p["strong"] or s["strong"]
        else:
            merged.append(s)
    return merged


def snap_clip_window(
    sentences: List[Dict[str, Any]],
    q_start: float,
    q_end: float,
    *,
    min_dur: float,
    floor_dur: float,
    max_dur: float,
    hard_max: float,
    max_extend: float,
) -> Optional[Dict[str, Any]]:
    """
    Прив'язує сире вікно LLM [q_start, q_end] до меж речень.

    Повертає {"first_idx", "last_idx", "start", "end"} або None, якщо
    завершену думку в це вікно покласти не вдалось. Кінець максимально
    близько до q_end (сильні межі мають перевагу), хвіст мусить бути
    завершеним, тривалість — у межах [floor_dur, hard_max].
    """
    if not sentences:
        return None

    def span(a: int, b: int) -> float:
        return sentences[b]["end"] - sentences[a]["start"]

    # Старт: перше речення, що перетинає q_start. Якщо q_start падає в паузу —
    # беремо наступне речення, щоб кліп не відкривався тишиною.
    start_i = len(sentences) - 1
    for i, s in enumerate(sentences):
        if s["end"] > q_start:
            start_i = i
            break

    # Кінець: найближча межа речення до q_end (сильним межам бонус 0.5с)
    end_i = min(
        range(len(sentences)),
        key=lambda i: abs(sentences[i]["end"] - q_end) - (0.5 if sentences[i]["strong"] else 0.0),
    )
    end_i = max(end_i, start_i)

    # Не вилазимо за жорстку стелю
    while span(start_i, end_i) > hard_max and end_i > start_i:
        end_i -= 1
    if span(start_i, end_i) > hard_max:
        return None

    # Набираємо до min_duration цілими реченнями (м'яка стеля max_dur)
    while span(start_i, end_i) < min_dur and end_i + 1 < len(sentences) and span(start_i, end_i + 1) <= max_dur:
        end_i += 1

    # Хвіст не може висіти на сполучнику: тягнемо вперед (в межах max_extend
    # та hard_max) або відкатуємось до останньої завершеної межі
    walked_back = False
    while not is_tail_complete(sentences[end_i]["text"]):
        can_extend = (
            not walked_back
            and end_i + 1 < len(sentences)
            and span(start_i, end_i + 1) <= hard_max
            and (sentences[end_i]["end"] - q_end) <= max_extend
        )
        if can_extend:
            end_i += 1
        elif end_i > start_i:
            end_i -= 1
            walked_back = True
        else:
            return None

    # Занадто короткий обривок — підтягуємо контекст реченням раніше
    while span(start_i, end_i) < floor_dur and start_i > 0 and span(start_i - 1, end_i) <= hard_max:
        start_i -= 1

    dur = span(start_i, end_i)
    if dur < floor_dur or dur > hard_max:
        return None

    return {
        "first_idx": start_i,
        "last_idx": end_i,
        "start": sentences[start_i]["start"],
        "end": sentences[end_i]["end"],
    }


def sentence_windows(
    sentences: List[Dict[str, Any]],
    lo: int,
    hi: int,
    *,
    min_dur: float,
    floor_dur: float,
    max_dur: float,
) -> List[Tuple[int, int]]:
    """
    Ріже діапазон речень [lo..hi] на послідовні вікна завершених думок:
    ціль min_dur, стеля max_dur, кожне вікно закінчується завершеним
    реченням. Повертає список пар індексів (first_idx, last_idx).
    """
    def span(a: int, b: int) -> float:
        return sentences[b]["end"] - sentences[a]["start"]

    windows: List[Tuple[int, int]] = []
    cur = lo
    while cur <= hi:
        j = cur
        while j <= hi and span(cur, j) < min_dur:
            j += 1

        if j > hi:
            # Остаток коротший за min_duration: хвіст має бути завершеним,
            # далі приклеюємо до попереднього вікна або відкидаємо
            last = hi
            while last >= cur and not is_tail_complete(sentences[last]["text"]):
                last -= 1
            if last < cur:
                break
            if windows and span(windows[-1][0], last) <= max_dur:
                windows[-1] = (windows[-1][0], last)
            elif span(cur, last) >= floor_dur:
                windows.append((cur, last))
            break

        # Завершуємо думку: подовжуємо висячий хвіст, поки влазить у max_dur
        while not is_tail_complete(sentences[j]["text"]) and j + 1 <= hi and span(cur, j + 1) <= max_dur:
            j += 1
        # Ужимаємо назад у max_dur
        while j > cur and span(cur, j) > max_dur:
            j -= 1
        # Після стиснення хвіст знову не має висіти
        while j > cur and not is_tail_complete(sentences[j]["text"]):
            j -= 1

        if span(cur, j) >= floor_dur:
            windows.append((cur, j))
            cur = j + 1
        else:
            break
    return windows


def slice_words_and_text(
    sentences: List[Dict[str, Any]],
    first_idx: int,
    last_idx: int,
) -> Tuple[List[Dict[str, Any]], str]:
    """Слова та текст кліпу для діапазону речень [first_idx..last_idx]."""
    words: List[Dict[str, Any]] = []
    texts: List[str] = []
    for i in range(first_idx, last_idx + 1):
        words.extend(sentences[i]["words"])
        texts.append(sentences[i]["text"])
    return words, " ".join(texts).strip()
