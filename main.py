#!/usr/bin/env python3
"""
CS2 Clips & Shorts Auto-Cutter
Main entrypoint for processing CS2 guide/lineup videos into vertical TikTok/Shorts clips.
"""

import os
import re
import sys
import json
import argparse
import threading
import glob
import time
import shutil
import yaml
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Fix compatibility between PyAV >= 19.0.0 and faster-whisper <= 1.2.1
try:
    import faster_whisper.audio as _fwa
    _p = _fwa.__file__
    with open(_p, "r", encoding="utf-8") as _f:
        _src = _f.read()
    if ', metadata_errors="ignore"' in _src:
        _src = _src.replace(', metadata_errors="ignore"', '')
        with open(_p, "w", encoding="utf-8") as _f:
            _f.write(_src)
except Exception:
    pass

try:
    import av
    _orig_av_open = av.open

    def _safe_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        kwargs.pop("metadata_encoding", None)
        return _orig_av_open(*args, **kwargs)

    av.open = _safe_av_open
except ImportError:
    pass

from pipeline.transcriber import Transcriber, extract_audio, get_video_duration
from pipeline.segmenter import Segmenter
from pipeline.analyzer import ContextAnalyzer
from pipeline.analyzer_qwen import QwenAnalyzer
from pipeline.banner import create_or_get_banner
from pipeline.subtitles import SubtitleGenerator
from pipeline.renderer import Renderer
from pipeline.metadata import generate_clip_metadata
from pipeline.downloader import fetch_channel_videos


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv"}

DEFAULT_CONFIG_YAML = """# ====================================================================
# MUSORZALIV: TIKTOK CS2 CLIPPER BOT CONFIGURATION
# Оптимізовано під алгоритми TikTok / Shorts (Retention, Hook, Completion)
# Аудиторія: СНД (російськомовні описи, заголовки, хуки та теги)
# ====================================================================

paths:
  input_dir: "input_videos"       # Сюди закидати вихідні відео
  output_dir: "output_clips"      # Готові змонтовані вертикальні кліпи
  temp_dir: "temp"                # Тимчасові файли
  assets_dir: "assets"
  auto_archive: true              # Автоматично переміщувати оброблені відео в archive

video:
  target_width: 1080
  target_height: 1920
  crop_mode: "center_crop"        # Динамічний 9:16 кріп по центру дій/прицілу
  zoom: 1.08                      # 108% зум (105-110%) для фокусу на дії та зміни хешу пікселів
  speed: 1.04                     # Прискорення для унікалізації хешу
  color_grade:
    enabled: true
    contrast: 1.03
    brightness: 0.01
    saturation: 1.05
  use_nvenc: true                 # NVENC GPU
  render_workers: 2               # скільки кліпів рендериться паралельно (NVENC тримає декілька сесій)
  nvenc_preset: "p4"
  video_bitrate: "6M"

  # Невидима унікалізація: для ока відео те саме, але перцептивні/аудіо хеші нові
  uniquize:
    enabled: true
    crop_px: [2, 5]            # випадковий мікрокроп з кожного краю
    noise_strength: [1, 2]     # невидимий часовий шум
    hue_shift: [-0.8, 0.8]     # мікрозсув відтінку (градуси)
    gamma: [0.99, 1.01]        # мікрозсув гамми
    speed_range: [1.03, 1.05]  # випадкове прискорення на кліп
    bitrate_jitter: 0.12       # ±12% до бітрейту
    gop_range: [96, 240]       # випадковий GOP
    tail_trim_max: 0.08        # випадкове обрізання хвоста (сек)
    audio_tone: true           # нечутний тон 16-19kHz (ламає аудіо-відбиток)
    tone_freq: [16000, 19000]
    tone_amp: [0.002, 0.005]

  # Динамічний зум-панч на перших 1.5 секунд (hook zoom — привертає увагу миттєво)
  hook_zoom:
    enabled: true
    scale: 1.12                    # +12% зум поверх базового на старті
    duration: 1.5                  # 1.5 секунди punch-in ефекту

  # Видимі покращення картинки
  beautify:
    enabled: true
    sharpen: 0.35              # легка чіткість геймплею
    vignette: true             # делікатна вінєтка
    bg_resolution: 270         # проміжна роздільна здатність фону
    bg_blur: 10
    bg_darken: 0.06
    bg_saturation: 1.12

# Аудіо-прибирка голосу (перед loudnorm)
audio:
  enhance: true
  highpass: 70                # зріз низького гулу
  compressor: "threshold=-20dB:ratio=2.5:attack=10:release=150:makeup=1.5"

# Параметри нарізки кліпів: довжину диктує контекст (семантичний юніт),
# рамки жорсткі: не коротше 20с і не довше 35 секунд (sweet-spot TikTok retention)
clips:
  sentence_based: true
  min_duration: 20.0              # Не коротше 20с
  max_duration: 35.0              # Не довше 35с (sweet-spot утримання)
  hard_max_duration: 38.0         # Абсолютна стеля
  target_duration: 28.0           # Орієнтир для fallback-нарізки
  silence_trim: true
  pad_start: -0.5                 # Старт 0.5с ДО першого слова — захопити екшен
  pad_end: 0.05                   # Зріз одразу після останнього слова
  start_jitter: 0.0
  max_internal_pause: 0.8
  silence_split_threshold: 0.8

  # Семантичне різання: межі кліпів чіпляються за речення Whisper, не за сирі таймкоди LLM
  sentence_snap: true
  snap_max_extend: 8.0            # макс. подовження хвоста до кінця речення (сек)
  sentence_soft_pause: 0.6        # пауза (сек), яка теж вважається межею речення
  min_clip_duration: 18.0         # підлога: не коротше 18с (сек)

whisper:
  model_size: "base"
  device: "auto"                   # auto = CUDA з відкатом на CPU
  compute_type: "auto"             # auto = float16 на GPU, int8 на CPU
  batch_size: 8                    # батч-транскрибація на GPU (~4x)
  language: null

banner:
  enabled: false

promo_midroll:
  enabled: false
  video_path: "assets/baner.mp4"

subtitles:
  enabled: true
  font_name: "DejaVu Sans"
  font_size: 54
  style: "box"
  words_per_group: 2              # 2-3 слова на екрані за раз
  bottom_margin: 680              # По центру екрану під прицілом
  active_word_highlight: true     # Анімація підсвічування активного слова
  active_word_color: "&H0000FFFF" # Жовтий акцентний колір
  font_color: "&H00FFFFFF"
  
  hook_overlay:
    enabled: true
    duration: 3.5                 # 0.0 - 3.5 секунди
    font_size: 66
    font_color: "&H0000FFFF"      # Яскраво-жовтий текст
    outline_color: "&H00000000"   # Товста чорна обводка
    outline_width: 8
    top_margin: 360               # Верхня третина кадру

analysis:
  enabled: true
  use_llm: true
  llm_url: "http://localhost:11434"
  llm_model: "qwen2.5:3b"
  llm_timeout: 180.0
  language: "ru"                  # Російська мова для СНГ
  min_score: 4.0
  max_clips: 0
  drop_sponsor_segments: true

# WOW-CUTTER: ранжований відбір (розмітка → кандидати → скоринг → реранк)
ranking:
  enabled: true
  llm_model: "qwen2.5:7b"         # краще розуміння; авто-пул, відкат на analysis.llm_model
  auto_pull: true
  top_n: 8                        # скільки кліпів віддавати на рендер
  rerank_top: 20                  # скільки фіналістів показувати Qwen
  min_total_score: 35.0           # відсік слабких кандидатів (0-100)
  review_clips: true              # огляд кліпів перед рендером
  weights:
    hook_open: 1.0                # захоплення уваги в перші 2с (макс 25)
    payoff: 1.0                   # вердикт/результат (макс 25)
    interest: 1.0                 # карта цікавості LLM (макс 20)
    dynamics: 1.0                 # темп мови (макс 10)
    cleanliness: 1.0              # без філерів (макс 10)
    cs2_terms: 1.0                # конкретика CS2 (макс 10)

tiktok:
  auto_upload: false
  webhook_url: ""
  delay_between_uploads: 900
"""


def ensure_project_structure(config: dict = None, config_path: str = "config.yaml"):
    """
    Ensures that all required directories and default configuration exist.
    Called automatically on every startup so the tool works out-of-the-box on fresh installs.
    """
    # 1. Create default config if missing
    if not os.path.exists(config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG_YAML)
        print(f"[✓] Створено новий файл конфігурації: {config_path}")

    # 2. Collect and create required directories
    base_dirs = [
        "input_videos",
        "input_videos/archive",
        "output_clips",
        "temp",
        "assets",
        "bin"
    ]
    if config:
        p_cfg = config.get("paths", {})
        for k in ["input_dir", "output_dir", "temp_dir", "assets_dir"]:
            if p_cfg.get(k):
                base_dirs.append(p_cfg[k])
                if k == "input_dir":
                    base_dirs.append(os.path.join(p_cfg[k], "archive"))

    for d in base_dirs:
        os.makedirs(d, exist_ok=True)


def load_config(config_path: str = "config.yaml") -> dict:
    ensure_project_structure(config_path=config_path)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            ensure_project_structure(config=cfg, config_path=config_path)
            return cfg
    return {}


def analyze_video(
    video_path: str,
    config: dict,
    transcriber: Transcriber,
    segmenter: Segmenter,
) -> tuple:
    """
    Steps 1-3: audio extraction, transcription, ranked clip selection.
    Returns (clips, video_duration). Без рендеру — використовується і для --preview.
    """
    video_file = Path(video_path)
    print("\n" + "=" * 65)
    print(f"🎬 ОБРОБКА ВІДЕО: {video_file.name}")
    print("=" * 65)

    duration = get_video_duration(str(video_file))
    print(f"[*] Тривалість відео: {duration:.1f} сек (~{duration / 60:.1f} хв)")

    temp_dir = config.get("paths", {}).get("temp_dir", "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Step 1: Extract Audio
    audio_path = os.path.join(temp_dir, f"{video_file.stem}_audio.wav")
    print(f"[*] Видобування аудіодоріжки...")
    extract_audio(str(video_file), audio_path)

    # Step 2: Whisper Local Transcription (з кешем: повторний запуск — миттєво)
    segments_cache = os.path.join(temp_dir, f"{video_file.stem}_segments.json")
    segments = None
    if os.path.exists(segments_cache) and os.path.getmtime(segments_cache) > os.path.getmtime(str(video_file)):
        try:
            with open(segments_cache, "r", encoding="utf-8") as f:
                segments = json.load(f).get("segments")
            if segments:
                print(f"[✓] Транскрипт взято з кешу ({len(segments)} сегментів) — Whisper пропущено")
        except Exception:
            segments = None

    if segments is None:
        print(f"[*] Локальна транскрибація через Whisper (прогрес у логах)...")
        t0 = time.time()
        segments, audio_dur, lang = transcriber.transcribe(audio_path)
        t_whisper = time.time() - t0
        print(f"[✓] Транскрибовано за {t_whisper:.2f} сек. Виявлена мова: {lang.upper()}, знайдено {len(segments)} сегментів мови.")
        try:
            with open(segments_cache, "w", encoding="utf-8") as f:
                json.dump({"segments": segments}, f, ensure_ascii=False)
        except Exception:
            pass

    # Step 3: Ranked clip selection (labeling → candidates → scoring → rerank)
    print(f"[*] Ранжований відбір моментів (розмітка → кандидати → скоринг)...")
    raw_clips = segmenter.find_clips(segments, duration, cache_key=video_file.stem)

    # Automated check: duration bounds з конфігу (жодних зашитих 28с/12с)
    clips_cfg = config.get("clips", {})
    hard_max_cfg = float(clips_cfg.get("hard_max_duration", 120.0))
    max_dur_cfg = float(clips_cfg.get("max_duration", 120.0))
    floor_cfg = float(clips_cfg.get("min_clip_duration", 30.0))

    clips = []
    for c in raw_clips:
        dur = float(c.get("duration", 0.0))
        if dur > hard_max_cfg:
            print(f"[⚠️] Автоматичний контроль: кліп #{c.get('index', '?')} ({dur:.1f}с > {hard_max_cfg:.0f}с). Обрізаємо до {max_dur_cfg:.0f}с...")
            c["end"] = c["start"] + max_dur_cfg
            c["duration"] = max_dur_cfg
            dur = max_dur_cfg
        if dur >= floor_cfg:
            clips.append(c)
        else:
            print(f"[⚠️] Відкинуто занадто короткий кліп ({dur:.1f}с < {floor_cfg:.0f}с).")

    print(f"[✓] Відібрано {len(clips)} кліпів ({floor_cfg:.0f}-{max_dur_cfg:.0f}с, довжину диктує контекст):")
    for i, c in enumerate(clips, 1):
        c["index"] = i
        hook_str = f" | Хук: «{c.get('hook_title', '')}»" if c.get("hook_title") else ""
        score_str = f" | Бал: {c.get('score_breakdown', {}).get('total', '?')}" if c.get("score_breakdown") else ""
        print(f"    • Кліп #{c['index']:02d}: [{c['start']:.1f}s -> {c['end']:.1f}s] ({c['duration']:.1f}с){hook_str}{score_str}")

    # Cleanup audio (render_clips перечитує відео напряму)
    if os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except Exception:
            pass

    return clips, duration


def print_preview_table(video_name: str, clips: list) -> None:
    """--preview: таблиця відібраних кліпів з розкладкою скорів, без рендеру."""
    from rich.console import Console as _Console
    from rich.table import Table as _Table
    from rich import box as _box

    console = _Console()
    console.print(f"\n[bold #ffe600]🔎 PREVIEW: {video_name}[/bold #ffe600] — {len(clips)} кліпів (без рендеру)\n")
    if not clips:
        console.print("[yellow]Кліпів не відібрано.[/yellow]")
        return

    t = _Table(box=_box.SIMPLE_HEAVY, header_style="bold #ff8c00")
    t.add_column("#", justify="center", width=3)
    t.add_column("Час", width=13)
    t.add_column("Трив", justify="right", width=5)
    t.add_column("Бал", justify="right", width=5)
    t.add_column("Хук", width=26, overflow="fold")
    t.add_column("Розкладка (hook/payoff/int/dyn/clean/cs2)", width=26)
    t.add_column("Перше речення", overflow="fold")

    for c in clips:
        sb = c.get("score_breakdown", {})
        layout = "/".join(str(sb.get(k, "?")) for k in ("hook_open", "payoff", "interest", "dynamics", "cleanliness", "cs2_terms"))
        first_sent = (c.get("text") or "").split(".")[0][:60]
        t.add_row(
            str(c.get("index", "?")),
            f"{c['start']:.0f}s-{c['end']:.0f}s",
            f"{c['duration']:.0f}с",
            f"[bold green]{sb.get('total', '?')}[/bold green]",
            (c.get("hook_title") or "")[:26],
            layout,
            first_sent,
        )
    console.print(t)


def cli_review_callback(video_name: str, clips: list, analyzer=None) -> list:
    """--review у консолі: таблиця + виключення за номерами."""
    print_preview_table(video_name, clips)
    if not clips:
        return clips
    try:
        raw = input("Виключити кліпи (номери через кому, Enter = лишити всі): ").strip()
    except (EOFError, KeyboardInterrupt):
        return clips
    if not raw:
        return clips
    drop = {int(x) for x in re.findall(r"\d+", raw) if 1 <= int(x) <= len(clips)}
    kept = [c for c in clips if c.get("index") not in drop]
    print(f"[✓] Залишено {len(kept)} з {len(clips)} кліпів")
    return kept


_PRINT_LOCK = threading.Lock()


def _p(msg: str) -> None:
    """Атомарний друк для паралельних рендер-потоків (без розривів рядків у логах)."""
    with _PRINT_LOCK:
        sys.stdout.write(str(msg) + "\n")
        sys.stdout.flush()


def render_clips(
    video_path: str,
    config: dict,
    clips: list,
    sub_generator,
    renderer,
    banner_img,
    analyzer,
) -> list:
    """Step 4: рендер вибраних кліпів (NVENC + 9:16 + субтитри + унікалізація).

    Паралельний: video.render_workers кліпів кодуються одночасно (NVENC тримає
    декілька сесій). Прогрес і мілстоуни — через атомарні принти.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    video_file = Path(video_path)
    temp_dir = config.get("paths", {}).get("temp_dir", "temp")
    output_dir = config.get("paths", {}).get("output_dir", "output_clips")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    render_workers = max(1, int(config.get("video", {}).get("render_workers", 2)))
    promo_v = renderer.get_promo_video()

    _p(f"\n[*] Апаратний рендер та унікалізація: {len(clips)} кліпів, {render_workers} паралельно...")
    created_clips = []

    def _render_one(c):
        clip_slug = c["title"]
        base_name = f"{video_file.stem}_clip_{c['index']:02d}_{clip_slug}"
        out_filename = f"{base_name}.mp4"
        out_clip_path = os.path.join(output_dir, out_filename)
        out_json_path = os.path.join(output_dir, f"{base_name}.json")

        # Subtitles generation (standard or split for mid-roll)
        clip_speed = renderer.get_clip_speed(c)
        ass_path = os.path.join(temp_dir, f"{video_file.stem}_clip_{c['index']:02d}.ass")
        ass_p1 = os.path.join(temp_dir, f"{video_file.stem}_clip_{c['index']:02d}_p1.ass")
        ass_p2 = os.path.join(temp_dir, f"{video_file.stem}_clip_{c['index']:02d}_p2.ass")

        sub_file = None
        sub_split = None
        try:
            if promo_v:
                p1, p2 = sub_generator.generate_ass_split(c, c["duration"] / 2.0, ass_p1, ass_p2, speed=clip_speed)
                sub_split = (p1, p2)
            else:
                sub_file = sub_generator.generate_ass(c, ass_path, speed=clip_speed)

            _p(f"[+] Рендер кліпу {c['index']}/{len(clips)}: {out_filename}")
            _p(f"    Таймкод: {c['start']:.1f}s - {c['end']:.1f}s | Довжина: {c['duration']:.1f}s")
            _p(f"    🔁 Унікалізація: {renderer.get_uniq_summary(c)}")

            t_render_start = time.time()
            ok = renderer.render_clip(
                input_video=str(video_file),
                clip=c,
                output_path=out_clip_path,
                banner_path=banner_img,
                ass_subtitles_path=sub_file,
                ass_subtitles_split=sub_split
            )
            t_render = time.time() - t_render_start

            if ok and os.path.exists(out_clip_path):
                file_mb = os.path.getsize(out_clip_path) / (1024 * 1024)
                _p(f"    [✓] Кліп #{c['index']} готово за {t_render:.1f}s! Розмір: {file_mb:.1f} MB -> {out_clip_path}")

                # Generate JSON metadata for n8n automation (context + optional LLM)
                generate_clip_metadata(c, video_file.stem, out_json_path, config, analyzer)
                _p(f"    [✓] Збережено метадані для n8n: {out_json_path}")
                return out_clip_path
            else:
                _p(f"    [✗] Помилка рендеру кліпу #{c['index']}.")
                return None
        except Exception as e:
            _p(f"    [✗] Кліп #{c['index']}: {e}")
            return None
        finally:
            for p in [ass_path, ass_p1, ass_p2]:
                if os.path.exists(p):
                    try: os.remove(p)
                    except: pass

    if render_workers <= 1 or len(clips) <= 1:
        for c in clips:
            res = _render_one(c)
            if res:
                created_clips.append(res)
    else:
        with ThreadPoolExecutor(max_workers=render_workers) as ex:
            futures = [ex.submit(_render_one, c) for c in clips]
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    created_clips.append(res)
        created_clips.sort()

    return created_clips


def process_video(
    video_path: str,
    config: dict,
    transcriber: Transcriber,
    segmenter: Segmenter,
    sub_generator: SubtitleGenerator,
    renderer: Renderer,
    banner_img: str,
    analyzer: ContextAnalyzer = None,
    review_callback=None,
):
    video_file = Path(video_path)

    clips, duration = analyze_video(video_path, config, transcriber, segmenter)

    # Огляд кліпів перед рендером (TUI-екран або консоль)
    if review_callback and clips:
        reviewed = review_callback(video_file.name, clips, analyzer)
        clips = reviewed if reviewed is not None else clips
        if not clips:
            print(f"[⚠️] Усі кліпи для {video_file.name} виключено в огляді — пропускаємо рендер.")
            return []

    created_clips = render_clips(video_path, config, clips, sub_generator, renderer, banner_img, analyzer)

    # Auto-archive original processed video if enabled
    auto_archive = config.get("paths", {}).get("auto_archive", False)
    if auto_archive and created_clips:
        archive_dir = os.path.join(os.path.dirname(video_path), "archive")
        os.makedirs(archive_dir, exist_ok=True)
        dest_path = os.path.join(archive_dir, video_file.name)
        shutil.move(video_path, dest_path)
        print(f"[*] Вихідне відео переміщено в архів: {dest_path}")

    output_dir = config.get("paths", {}).get("output_dir", "output_clips")
    print(f"\n🎉 Відео {video_file.name} успішно оброблено! Створено {len(created_clips)} кліпів у '{output_dir}'.")
    return created_clips


def get_input_videos(input_target: str = "input_videos") -> list:
    """Finds all video files in target path (ignoring archive directory)."""
    video_files = []
    if os.path.isfile(input_target):
        video_files = [input_target]
    elif os.path.isdir(input_target):
        for root, dirs, files in os.walk(input_target):
            if "archive" in dirs:
                dirs.remove("archive")
            for f in sorted(files):
                if Path(f).suffix.lower() in VIDEO_EXTENSIONS:
                    video_files.append(os.path.join(root, f))
    return video_files


def run_cutter(config: dict = None, input_target: str = None, review_callback=None, preview_only: bool = False) -> list:
    """
    Runs full pipeline on queued videos.
    review_callback(video_name, clips) -> clips: огляд кліпів перед рендером.
    preview_only=True: тільки аналіз + таблиця, без рендеру.
    """
    if config is None:
        config = load_config()

    input_target = input_target or config.get("paths", {}).get("input_dir", "input_videos")
    video_files = get_input_videos(input_target)

    if not video_files:
        return []

    print("=" * 65)
    print("🚀 ЗАПУСК CS2 AUTO-CUTTER ДЛЯ TIKTOK / SHORTS")
    print(f"Знайдено відео для обробки: {len(video_files)}")
    print(f"Режим кодування: {'NVENC (GPU)' if config.get('video', {}).get('use_nvenc', True) else 'CPU'}")
    print(f"Режим кропу: {config.get('video', {}).get('crop_mode', 'blur_background')}")
    print(f"Швидкість: {config.get('video', {}).get('speed', 1.04)}x")
    print(f"Ранжований відбір: {'УВІМКНЕНО' if config.get('ranking', {}).get('enabled', True) else 'вимкнено'}")
    print("=" * 65)

    transcriber = Transcriber(config)
    analyzer = ContextAnalyzer(config)
    if analyzer.enabled:
        if analyzer.llm_available():
            print(f"[🧠] Контекст-аналіз: УВІМКНЕНО + локальний LLM ({analyzer.qwen.llm_model})")
        else:
            print("[🧠] Контекст-аналіз: УВІМКНЕНО (локальні алгоритми; LLM не знайдено — Ollama опційна)")
    segmenter = Segmenter(config, analyzer)
    sub_generator = SubtitleGenerator(config)
    renderer = Renderer(config)

    banner_img = None
    if config.get("banner", {}).get("enabled", False):
        banner_img = create_or_get_banner(config)

    all_created = []
    t_all_start = time.time()

    for v in video_files:
        try:
            if preview_only:
                clips, _duration = analyze_video(v, config, transcriber, segmenter)
                print_preview_table(Path(v).name, clips)
                continue

            created = process_video(
                v, config, transcriber, segmenter, sub_generator, renderer, banner_img, analyzer,
                review_callback=review_callback,
            )
            all_created.extend(created)
        except Exception as e:
            print(f"[✗] Помилка обробки файлу {v}: {e}")
            import traceback
            traceback.print_exc()

    t_total = time.time() - t_all_start
    if preview_only:
        print("\n" + "=" * 65)
        print(f"🔎 PREVIEW завершено за {t_total:.1f} сек (без рендеру).")
        print("=" * 65)
        return []
    print("\n" + "=" * 65)
    print(f"✨ ВСІ ЗАВДАННЯ ВИКОНАНО ЗА {t_total:.1f} сек!")
    print(f"Усього згенеровано кліпів: {len(all_created)}")
    print(f"Папка з готовими кліпами: {os.path.abspath(config.get('paths', {}).get('output_dir', 'output_clips'))}")
    print("=" * 65)
    return all_created


def main():
    parser = argparse.ArgumentParser(description="CS2 Clips & Lineups Auto-Cutter for TikTok/Shorts")
    parser.add_argument("-i", "--input", help="Шлях до вхідного відео або папки з відео", default=None)
    parser.add_argument("-o", "--output", help="Шлях до вихідної папки для кліпів", default=None)
    parser.add_argument("-c", "--config", help="Шлях до файлу конфігурації", default="config.yaml")
    parser.add_argument("--fetch", action="store_true", help="Автоматично викачати нові відео з каналів-донорів перед запуском")
    parser.add_argument("--promo", help="Шлях до рекламного відео для вставки в середину кліпу (mid-roll)", default=None)
    parser.add_argument("--no-promo", action="store_true", help="Вимкнути рекламний ролик (чисті ролики для прогріву акаунтів)")
    parser.add_argument("--archive", action="store_true", help="Автоматично переміщувати оброблені відео в input_videos/archive/")
    parser.add_argument("--no-banner", action="store_true", help="Вимкнути накладання промо-банера")
    parser.add_argument("--no-subs", action="store_true", help="Вимкнути динамічні субтитри")
    parser.add_argument("--no-nvenc", action="store_true", help="Вимкнути апаратне кодування NVENC (використовувати CPU)")
    parser.add_argument("--crop", choices=["center_crop", "blur_background"], help="Режим кропу 9:16")
    parser.add_argument("--speed", type=float, help="Швидкість відео (наприклад, 1.04)")
    parser.add_argument("--preview", action="store_true", help="Тільки аналіз + таблиця відібраних кліпів з балами (без рендеру)")
    parser.add_argument("--review", action="store_true", help="Огляд кліпів у консолі перед рендером (виключити невдалі)")

    args = parser.parse_args()

    config = load_config(args.config)

    # CLI overrides
    if args.output:
        config.setdefault("paths", {})["output_dir"] = args.output
    if args.archive:
        config.setdefault("paths", {})["auto_archive"] = True
    if args.promo:
        config.setdefault("promo_midroll", {})["enabled"] = True
        config["promo_midroll"]["video_path"] = args.promo
    if args.no_promo:
        config.setdefault("promo_midroll", {})["enabled"] = False
    if args.no_banner:
        config.setdefault("banner", {})["enabled"] = False
    if args.no_subs:
        config.setdefault("subtitles", {})["enabled"] = False
    if args.no_nvenc:
        config.setdefault("video", {})["use_nvenc"] = False
    if args.crop:
        config.setdefault("video", {})["crop_mode"] = args.crop
    if args.speed:
        config.setdefault("video", {})["speed"] = args.speed

    # Check for channel fetching
    if args.fetch:
        print("[*] Запуск автоматичного завантаження з каналів-донорів...")
        fetch_channel_videos()

    input_target = args.input or config.get("paths", {}).get("input_dir", "input_videos")
    video_files = get_input_videos(input_target)

    if not video_files:
        print("\n" + "=" * 65)
        print("⚠️  НЕ ЗНАЙДЕНО ВІДЕО ДЛЯ ОБРОБКИ!")
        print("=" * 65)
        print(f"Будь ласка, закинь свої відео файли (.mp4, .mkv, .mov) у папку:")
        print(f"👉  {os.path.abspath(input_target)}")
        print("\nАбо запустіть з прапорцем автозавантаження:")
        print("👉  ./run.sh --fetch")
        print("=" * 65)
        sys.exit(0)

    # Огляд кліпів: --review примусово; ranking.review_clips — якщо є інтерактивний термінал
    review_callback = None
    review_enabled = args.review or bool(config.get("ranking", {}).get("review_clips", False))
    if review_enabled and not args.preview and sys.stdin.isatty():
        review_callback = cli_review_callback

    run_cutter(config, input_target, review_callback=review_callback, preview_only=args.preview)


if __name__ == "__main__":
    main()
