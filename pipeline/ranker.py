"""
Candidate generation & deterministic scoring (WOW-CUTTER Step 2).

Ключова ідея: не просити LLM «знайти моменти» (3B вгадує), а згенерувати ВСІ
валідні вікна-кандидати (кожен старт × 15-28с, з гарантіями юнітів та
завершених хвостів) і проженути їх через детермінований скоринг. LLM лишається
тільки на останній милі — порівняти топ-20 готових текстів (analyzer_qwen).

Критерії (макс 100 балів, ваги в config.yaml → ranking.weights):
    hook_open    25  — захоплення уваги з перших секунд
    payoff       25  — вердикт/результат у кліпі (бонус — в останній третині)
    interest     20  — карта цікавості від labeler + пік на початку
    dynamics     10  — темп мови, відсутність проваллянь
    cleanliness  10  — відсутність філерів («ну», «типу», «короче»)
    cs2_terms    10  — щільність конкретики (карти, гранати, числа)
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from pipeline.sentences import is_tail_complete, sentence_windows, slice_words_and_text
from pipeline.units import (
    detect_news_units,
    detect_units,
    looks_like_news_transition,
    looks_like_myth_intro,
    looks_like_verdict,
    plan_unit_parts,
)
from pipeline.labeler import FILLER_RE, HOOK_RE, RESULT_RE, CS2_TERMS_RE

# Максимальний бал кожного критерію (для нормалізації total до 100)
SCORE_MAXES = {
    "hook_open": 25.0,
    "payoff": 25.0,
    "interest": 20.0,
    "dynamics": 10.0,
    "cleanliness": 10.0,
    "cs2_terms": 10.0,
}

DEFAULT_WEIGHTS = {k: 1.0 for k in SCORE_MAXES}

DIGIT_RE = re.compile(r"\d")
LEADING_DIGIT_RE = re.compile(r"^\s*(?:топ|top)?\s*\d", re.IGNORECASE)


def _words(text: str) -> List[str]:
    return [w for w in re.split(r"\s+", text.strip()) if w]


# ---------------------------------------------------------------- хук-старт

def sentence_hook_strength(text: str, label_role: Optional[str] = None) -> float:
    """0..25: наскільки речення здатне зачепити увагу на першій секунді."""
    t = text.strip()
    score = 0.0
    if t.endswith("?"):
        score = 20.0
    if looks_like_myth_intro(t):
        score = max(score, 18.0)
    if HOOK_RE.search(t):
        score = max(score, 16.0)
    if LEADING_DIGIT_RE.match(t):
        score = max(score, 14.0)
    if label_role == "hook":
        score = min(25.0, score + 6.0) if score else 16.0
    return score


# ---------------------------------------------------------------- кандидати

def _make_candidate(sentences, first_idx: int, last_idx: int, *, source: str, unit_type: Optional[str] = None, part: int = 1, parts_total: int = 1) -> Optional[Dict[str, Any]]:
    s0, s1 = sentences[first_idx], sentences[last_idx]
    duration = s1["end"] - s0["start"]
    if duration < 3.0:
        return None
    words, text = slice_words_and_text(sentences, first_idx, last_idx)
    if not words or not text:
        return None
    return {
        "first_idx": first_idx,
        "last_idx": last_idx,
        "start": s0["start"],
        "end": s1["end"],
        "duration": duration,
        "text": text,
        "words": words,
        "source": source,
        "unit_type": unit_type,
        "part": part,
        "parts_total": parts_total,
        "scores": {},
    }


def _myth_window_valid(sentences, first_idx: int, last_idx: int) -> bool:
    """Якщо всередині вікна є теза міфа — вердикт теж мусить бути всередині (після тези)."""
    verdict_after = False
    for i in range(last_idx, first_idx - 1, -1):
        t = sentences[i]["text"]
        if looks_like_myth_intro(t) and not verdict_after:
            return False
        if looks_like_verdict(t):
            verdict_after = True
    return True


def _news_window_valid(sentences, first_idx: int, last_idx: int) -> bool:
    """Вікно не стрибає між новинами: перехід всередині (після старту) заборонений."""
    return not any(
        looks_like_news_transition(sentences[i]["text"])
        for i in range(first_idx + 1, last_idx + 1)
    )


def _question_window_valid(sentences, first_idx: int, last_idx: int, unit_type: Optional[str]) -> bool:
    """
    Питання не відриваються від відповідей (розмовний контент):
      - кліп не може ЗАКІНЧУВАТИСЬ на питанні без відповіді (обрубок-дразнилка)
        — виняток: Ч.1 міфа, де питання-кліфгенгер навмисне;
      - кліп, що СТАРТУЄ з питання, мусить мати щонайменше 2 речення продовження.
    """
    if unit_type == "myth":
        return True
    if sentences[last_idx]["text"].rstrip().endswith("?"):
        return False
    if sentences[first_idx]["text"].rstrip().endswith("?") and (last_idx - first_idx) < 2:
        return False
    return True


def generate_candidates(
    sentences: List[Dict[str, Any]],
    *,
    min_dur: float,
    max_dur: float,
    hard_max: float,
) -> List[Dict[str, Any]]:
    """
    Усі валідні вікна-кандидати:
      1) слайдинг: кожен старт × тривалості [min_dur .. max_dur], крок 1 речення;
      2) юнітні плани: міфи (теза→вердикт, з розбиттям на частини).
    """
    candidates: List[Dict[str, Any]] = []
    seen: set = set()
    n = len(sentences)

    def span(a: int, b: int) -> float:
        return sentences[b]["end"] - sentences[a]["start"]

    def add(first_idx: int, last_idx: int, source: str, unit_type: Optional[str] = None, part: int = 1, parts_total: int = 1, cross_news: bool = False):
        key = (first_idx, last_idx)
        if key in seen:
            return
        if not is_tail_complete(sentences[last_idx]["text"]):
            return
        if not _myth_window_valid(sentences, first_idx, last_idx):
            return
        if not cross_news and not _news_window_valid(sentences, first_idx, last_idx):
            return
        if not _question_window_valid(sentences, first_idx, last_idx, unit_type):
            return
        cand = _make_candidate(sentences, first_idx, last_idx, source=source, unit_type=unit_type, part=part, parts_total=parts_total)
        if cand:
            seen.add(key)
            candidates.append(cand)

    # 1) Слайдинг-вікна
    for s in range(n):
        e = s
        while e < n and span(s, e) < min_dur:
            e += 1
        if e < n:
            add(s, e, "sliding")
        while e + 1 < n and span(s, e + 1) <= max_dur:
            e += 1
            add(s, e, "sliding")

    # 2) Міф-юніти (гарантія вердикту + розбиття довгих на частини)
    units = detect_units(sentences, 0, n - 1)
    for u in units:
        parts = plan_unit_parts(
            sentences, u["first_idx"], u["last_idx"],
            min_dur=min_dur, floor_dur=min(18.0, min_dur), max_dur=max_dur, hard_max=hard_max,
        )
        for p_i, (i0, i1) in enumerate(parts, 1):
            add(i0, i1, "unit", unit_type="myth", part=p_i, parts_total=len(parts))

    # 3) Новинні юніти: кліп = одна новина від заголовка до кінця
    news_units = detect_news_units(sentences, 0, n - 1)
    for k, nu in enumerate(news_units):
        span_nu = span(nu["first_idx"], nu["last_idx"])
        if span_nu <= max_dur:
            add(nu["first_idx"], nu["last_idx"], "unit", unit_type="news")
        else:
            win = sentence_windows(
                sentences, nu["first_idx"], nu["last_idx"],
                min_dur=min_dur, floor_dur=min(18.0, min_dur), max_dur=max_dur,
            )
            total = len(win)
            for p_i, (i0, i1) in enumerate(win, 1):
                add(i0, i1, "unit", unit_type="news", part=p_i, parts_total=total)

    # 4) Короткі новини склеюються в дайджест (2-3 штуки), поки не набереться min_dur —
    #    свідомий перетин переходів (cross_news), бо це і є сенс дайджеста
    for k, nu in enumerate(news_units):
        if span(nu["first_idx"], nu["last_idx"]) >= min_dur:
            continue
        last_idx = nu["last_idx"]
        m = k + 1
        while m < len(news_units) and span(nu["first_idx"], news_units[m]["last_idx"]) < min_dur:
            last_idx = news_units[m]["last_idx"]
            m += 1
        if m < len(news_units) or last_idx != nu["last_idx"]:
            if last_idx > nu["last_idx"] and span(nu["first_idx"], last_idx) <= max_dur:
                # додаємо ще ОДНУ новину поверх мінімуму, щоб не обривати останню
                if m < len(news_units) and span(nu["first_idx"], news_units[m]["last_idx"]) <= max_dur:
                    last_idx = news_units[m]["last_idx"]
                add(nu["first_idx"], last_idx, "unit", unit_type="news", cross_news=True)

    return candidates


# ---------------------------------------------------------------- скоринг

def score_candidate(
    cand: Dict[str, Any],
    sentences: List[Dict[str, Any]],
    labels: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Наповнює cand["scores"] і повертає розкладку. Детерміновано, без LLM."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        for k, v in weights.items():
            if k in w:
                try:
                    w[k] = float(v)
                except (TypeError, ValueError):
                    pass

    first_idx, last_idx = cand["first_idx"], cand["last_idx"]
    sents = sentences[first_idx:last_idx + 1]
    texts = [s["text"] for s in sents]
    joined = " ".join(texts)
    words = cand["words"]
    n_words = max(1, len(words))
    duration = max(0.1, cand["duration"])

    # --- Hook-Open (0-25): сила першого речення, бонус якщо хук на 2-му в перші 2с
    label_first = labels[first_idx]["role"] if first_idx < len(labels) else None
    hook = sentence_hook_strength(texts[0], label_first)
    if hook == 0.0 and len(sents) > 1:
        label_second = labels[first_idx + 1]["role"] if first_idx + 1 < len(labels) else None
        second = sentence_hook_strength(texts[1], label_second)
        if second > 0.0 and (sents[1]["start"] - sents[0]["start"]) <= 2.0:
            hook = second * 0.6
    hook_open = round(min(25.0, hook), 1)

    # --- Payoff (0-25): вердикт/результат; бонус за позицію в останній третині
    payoff_pos: Optional[int] = None
    for k, text in enumerate(texts):
        role = labels[first_idx + k]["role"] if first_idx + k < len(labels) else None
        if role in ("verdict", "result") or looks_like_verdict(text) or RESULT_RE.search(text):
            payoff_pos = k
            break
    payoff = 0.0
    if payoff_pos is not None:
        payoff = 15.0
        if payoff_pos >= max(1, (len(texts) - 1) * 2 // 3):
            payoff += 10.0
    payoff = round(min(25.0, payoff), 1)

    # --- Interest (0-20): середній інтерес + пік у першій третині
    interests = []
    for k in range(len(sents)):
        i = first_idx + k
        interests.append(float(labels[i]["interest"]) if i < len(labels) else 5.0)
    mean_i = sum(interests) / max(1, len(interests))
    first_third = interests[: max(1, len(interests) // 3)]
    peak_i = max(first_third) if first_third else 0.0
    interest = round(min(20.0, mean_i / 10.0 * 12.0 + peak_i / 10.0 * 8.0), 1)

    # --- Dynamics (0-10): темп + паузи
    wps = n_words / duration
    dynamics = max(0.0, min(1.0, (wps - 2.0) / 3.0)) * 10.0
    pause_penalty = 0.0
    for k in range(1, len(words)):
        gap = words[k]["start"] - words[k - 1]["end"]
        if gap > 0.8:
            pause_penalty += 1.0
        if pause_penalty >= 3.0:
            break
    dynamics = round(max(0.0, dynamics - pause_penalty), 1)

    # --- Cleanliness (0-10): філери
    fillers = len(FILLER_RE.findall(joined))
    density_pct = fillers * 100.0 / n_words
    cleanliness = round(max(0.0, 10.0 - density_pct * 2.2), 1)

    # --- CS2-конкретика (0-10): терміни + числа на 10 слів
    hits = len(CS2_TERMS_RE.findall(joined)) + len(DIGIT_RE.findall(joined)) * 0.5
    cs2 = round(min(10.0, hits * 10.0 / n_words * 3.5), 1)

    scores = {
        "hook_open": hook_open,
        "payoff": payoff,
        "interest": interest,
        "dynamics": dynamics,
        "cleanliness": cleanliness,
        "cs2_terms": cs2,
    }
    weighted_total = sum(scores[k] * w[k] for k in scores)
    max_total = sum(SCORE_MAXES[k] * w[k] for k in SCORE_MAXES)
    scores["total"] = round(weighted_total / max(0.01, max_total) * 100.0, 1)
    cand["scores"] = scores
    return scores


def score_all(
    candidates: List[Dict[str, Any]],
    sentences: List[Dict[str, Any]],
    labels: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> None:
    for c in candidates:
        score_candidate(c, sentences, labels, weights)


# ---------------------------------------------------------------- відбір

def _time_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    inter = min(a["end"], b["end"]) - max(a["start"], b["start"])
    if inter <= 0:
        return 0.0
    return inter / max(0.1, min(a["duration"], b["duration"]))


def select_top(
    candidates: List[Dict[str, Any]],
    *,
    top_n: int,
    min_score: float = 0.0,
    max_overlap: float = 0.4,
    max_per_source_unit: int = 3,
) -> List[Dict[str, Any]]:
    """
    Сортування за total + жадібний відбір:
      - перетин із уже вибраними > max_overlap → skip (анти-дубль);
      - не більше max_per_source_unit частин одного міф-юніта (різноманіття);
      - кліпи нижче min_score відсіюються.
    """
    ranked = sorted(candidates, key=lambda c: c["scores"].get("total", 0.0), reverse=True)
    selected: List[Dict[str, Any]] = []
    unit_parts_count: Dict[Tuple[int, int], int] = {}

    for c in ranked:
        if c["scores"].get("total", 0.0) < min_score:
            continue
        if c["source"] == "unit" and c["parts_total"] > 1:
            key = (c["first_idx"], c["last_idx"])
            if unit_parts_count.get(key, 0) >= max_per_source_unit:
                continue
        if any(_time_overlap(c, s) > max_overlap for s in selected):
            continue
        selected.append(c)
        if c["source"] == "unit" and c["parts_total"] > 1:
            key = (c["first_idx"], c["last_idx"])
            unit_parts_count[key] = unit_parts_count.get(key, 0) + 1
        if len(selected) >= top_n:
            break

    selected.sort(key=lambda c: c["start"])
    return selected


# ---------------------------------------------------------------- cold-open

LEADING_FILLER_RE = re.compile(
    r"^(ну|короче|типу|типа|значит|значить|вот|так|ось|просто|в общем|взагалі|до речі|кстати)[,.\s]+",
    re.IGNORECASE,
)


def apply_cold_open(
    cand: Dict[str, Any],
    sentences: List[Dict[str, Any]],
    labels: List[Dict[str, Any]],
    *,
    floor_dur: float = 12.0,
    max_shift: float = 4.0,
) -> Dict[str, Any]:
    """
    «Захват уваги з першої секунди»:
      1) лід-філери першого речення зрізаються (старт із першого змістовного слова);
      2) якщо старт не хуковий — зсув до найближчого hook-речення в перших max_shift
         секундах (міфи не зсуваємо — губиться теза).
    Мутує candidate (start/first_idx/words/text) і повертає його.
    floor_dur — підлога тривалості після зрізів (не цільова мінімальна).
    """
    first_idx = cand["first_idx"]

    # 1) лід-філери першого речення
    words = cand["words"]
    first_text = sentences[first_idx]["text"]
    n_first_words = len(_words(first_text))
    cut = 0
    for w in words[: min(4, n_first_words)]:
        if LEADING_FILLER_RE.match(w["word"]) and len(words) - cut > 3:
            cut += 1
        else:
            break
    if cut and (words[-1]["end"] - words[cut]["start"]) >= floor_dur:
        cand["words"] = words[cut:]
        cand["start"] = round(words[cut]["start"], 3)
        cand["duration"] = round(cand["end"] - cand["start"], 2)
        cand["text"] = re.sub(LEADING_FILLER_RE, "", cand["text"], count=1).strip() or cand["text"]

    # 2) зсув старту до hook-речення
    if cand.get("unit_type") != "myth":
        for k in (1, 2):
            idx = first_idx + k
            if idx > cand["last_idx"]:
                break
            role = labels[idx]["role"] if idx < len(labels) else None
            strength = sentence_hook_strength(sentences[idx]["text"], role)
            if strength <= 0.0:
                continue
            new_start = sentences[idx]["start"]
            if new_start - cand["start"] > max_shift:
                break
            if (cand["end"] - new_start) >= floor_dur:
                new_words, new_text = slice_words_and_text(sentences, idx, cand["last_idx"])
                if new_words:
                    cand["first_idx"] = idx
                    cand["start"] = round(new_start, 3)
                    cand["duration"] = round(cand["end"] - cand["start"], 2)
                    cand["words"] = new_words
                    cand["text"] = new_text
            break

    cand["start"] = round(cand["start"], 3)
    cand["duration"] = round(cand["end"] - cand["start"], 2)
    return cand
