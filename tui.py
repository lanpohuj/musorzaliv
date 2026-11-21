#!/usr/bin/env python3
"""
MUSORZALIV v1.2.0 — Interactive TUI (Rich + readchar).
Меню в стилі системної консолі: запуск нарізки з живим прогресом,
статистика кліпів, налаштування config.yaml, реальні логи пайплайну.
"""

import os
import re
import sys
import glob
import time
import shutil
import threading
import urllib.request
from collections import deque
from pathlib import Path

import yaml
import readchar
from rich.console import Console, Group
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
from rich import box

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Ensure bin directory (ffmpeg/ffprobe/yt-dlp) is in PATH
os.environ["PATH"] = f"{BASE_DIR / 'bin'}:{os.environ.get('PATH', '')}"

# ==================== ДІАГНОСТИКА: ПОСТІЙНИЙ ЛОГ ====================
# Усе, що відбувається (включно з дампами падінь), пишеться в temp/musorzaliv.log —
# за ним можна подивитись ПОСЛЕ того, як програма закрилась.
import faulthandler

APP_VERSION = "v1.2.0"
LOG_PATH = os.path.join(BASE_DIR, "temp", "musorzaliv.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
_log_file = open(LOG_PATH, "a", buffering=1, encoding="utf-8")
faulthandler.enable(file=_log_file)  # сегфолти/аборти нативних бібліотек теж потраплять у лог
_log_file.write(f"\n===== MUSORZALIV TUI {APP_VERSION} старт {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")

def _thread_excepthook(args):
    import traceback as _tb
    _log_file.write(f"[THREAD-CRASH] {args.thread}: {args.exc_type.__name__}: {args.exc_value}\n")
    _log_file.write("".join(_tb.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))

threading.excepthook = _thread_excepthook

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

from main import run_cutter, load_config, get_input_videos, ensure_project_structure
from pipeline.downloader import download_youtube_video
from pipeline.transcriber import get_video_duration

console = Console()

# ==================== НАЛАШТУВАННЯ ТЕМИ ====================
START_RGB = (255, 140, 0)   # Amber Orange
END_RGB   = (255, 230, 0)   # Cyber Yellow

PRIMARY   = "#ffe600"       # Яскраво-жовтий акцент
SECONDARY = "#ff8c00"       # Оранжевий
SELECTED_BG = "bold black on #ffe600"
SELECTED_SUB = "#ffaa00"
# ===========================================================

# ---- Лого MUSORZALIV (ANSI Shadow) ----
GLYPHS = {
    "M": ["███╗   ███╗", "████╗ ████╗", "██╔████╔██║", "██║╚███╔██║", "██║ ╚═══██║", "██║     ██║", "╚═╝     ╚═╝"],
    "U": ["██╗   ██╗", "██║   ██║", "██║   ██║", "██║   ██║", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
    "S": ["███████╗", "██╔════╝", "███████╗", "╚════██║", "███████║", "╚══════╝"],
    "O": [" ██████╗ ", "██╔═══██╗", "██║   ██║", "██║   ██║", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
    "R": ["██████╗ ", "██╔══██╗", "██████╔╝", "██╔══██╗", "██║  ██║", "╚═╝  ╚═╝"],
    "Z": ["███████╗", " ╚══███╔╝", "   ███╔╝ ", "  ███╔╝  ", " ███████╗", "╚═══════╝"],
    "A": [" █████╗ ", "██╔══██╗", "███████║", "██╔══██║", "██║  ██║", "╚═╝  ╚═╝"],
    "L": ["██╗     ", "██║     ", "██║     ", "██║     ", "███████╗", "╚══════╝"],
    "I": ["██╗", "██║", "██║", "██║", "██║", "╚═╝"],
    "V": ["██╗   ██╗", "██║   ██║", "██║   ██║", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
}


def build_raw_logo():
    """MUSORZALIV у стилі ANSI Shadow, вирівняний по базовій лінії."""
    glyphs = [GLYPHS[ch] for ch in "MUSORZALIV"]
    height = max(len(g) for g in glyphs)
    rows = []
    for r in range(height):
        line = ""
        for g in glyphs:
            w = len(g[0])
            off = height - len(g)
            line += g[r - off] if r >= off else " " * w
        rows.append(line.rstrip() or " ")
    return rows


RAW_LOGO_LINES = build_raw_logo()
LOGO_WIDTH = max(len(l) for l in RAW_LOGO_LINES)


def generate_gradient_colors(start_rgb, end_rgb, steps):
    colors = []
    for i in range(steps):
        r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * i / max(1, steps - 1))
        g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * i / max(1, steps - 1))
        b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * i / max(1, steps - 1))
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors


def get_centered_logo_text():
    if shutil.get_terminal_size().columns < LOGO_WIDTH + 4:
        # вузький термінал: компактний бейдж замість повного лого
        t = Text(" M U S O R Z A L I V ", style=f"bold black on {PRIMARY}")
        return Align.center(t)
    char_colors = generate_gradient_colors(START_RGB, END_RGB, LOGO_WIDTH)
    rendered = Text()
    for line in RAW_LOGO_LINES:
        for idx, char in enumerate(line):
            rendered.append(char, style=f"bold {char_colors[min(idx, len(char_colors) - 1)]}")
        rendered.append("\n")
    return rendered


# ==================== СТАН ПАЙПЛАЙНУ ====================
STATE = {
    "running": False,
    "review_active": False,
    "thread": None,
    "logs": deque(maxlen=600),
    "results": [],
    "progress": {
        "phase": "idle",       # idle | working | done | error
        "video": "",           # поточне відео
        "video_i": 0,          # номер відео
        "video_n": 0,          # всього відео
        "clip_i": 0,           # поточний кліп
        "clip_n": 0,           # всього кліпів у відео
        "clips_done": 0,       # зрендерено кліпів
        "last_line": "",
        "error": "",
    },
}
_LOG_LOCK = threading.Lock()

# Мілстоуни з виводу пайплайну (main.py)
RE_VIDEO_START = re.compile(r"🎬\s+ОБРОБКА ВІДЕО:\s*(.+?)\s*$")
RE_CLIPS_PLANNED = re.compile(r"Сформовано\s+(\d+)\s+TikTok-кліпів|Відібрано\s+(\d+)\s+кліпів")
RE_CLIP_RENDER = re.compile(r"Рендер кліпу\s+(\d+)/(\d+)")
RE_CLIP_DONE = re.compile(r"✓\].*[Гг]отово за")
RE_VIDEO_DONE = re.compile(r"🎉\s+Відео\s+.+?успішно оброблено!\s+Створено\s+(\d+)")


class Tee:
    """Хапає вивід пайплайну: у буфер логів + мілстоуни прогресу. В термінал не форвардить."""

    def write(self, s):
        if not s:
            return len(s) if s else 0
        try:
            _log_file.write(s)
        except Exception:
            pass
        with _LOG_LOCK:
            logs = STATE["logs"]
            for line in s.splitlines():
                line = line.rstrip()
                if not line:
                    continue
                logs.append(line)
                p = STATE["progress"]
                p["last_line"] = line[-120:]
                m = RE_VIDEO_START.search(line)
                if m:
                    p["video"] = m.group(1).strip()
                    p["video_i"] += 1
                    p["clip_i"] = 0
                    p["clip_n"] = 0
                    p["clips_done"] = 0
                m = RE_CLIPS_PLANNED.search(line)
                if m:
                    p["clip_n"] = int(m.group(1) or m.group(2))
                    p["clips_done"] = 0
                m = RE_CLIP_RENDER.search(line)
                if m:
                    p["clip_i"] = int(m.group(1))
                    p["clip_n"] = int(m.group(2))
                if RE_CLIP_DONE.search(line):
                    p["clips_done"] += 1  # паралельний рендер: порядок завершення довільний
                m = RE_VIDEO_DONE.search(line)
                if m:
                    p["clips_done"] = int(m.group(1))
        return len(s)

    def flush(self):
        pass


def log_line(text, error=False):
    line = ("[ERROR] " if error else "") + str(text)
    with _LOG_LOCK:
        STATE["logs"].append(line)
    try:
        _log_file.write(line + "\n")
    except Exception:
        pass


def ollama_status():
    """Перевіряє Ollama раз на 30с, щоб не тормозити малювання меню."""
    global _OLLAMA_CACHE
    now = time.time()
    if now - _OLLAMA_CACHE[0] < 30:
        return _OLLAMA_CACHE[1]
    ok = False
    try:
        cfg = load_config()
        url = cfg.get("analysis", {}).get("llm_url", "http://localhost:11434")
        with urllib.request.urlopen(f"{url}/api/tags", timeout=0.7) as r:
            ok = r.status == 200
    except Exception:
        ok = False
    _OLLAMA_CACHE = (now, ok)
    return ok


_OLLAMA_CACHE = (0.0, False)


# ==================== ХЕДЕР / ФУТЕР ====================
def make_header():
    header_text = Text()
    header_text.append(" MUSORZALIV v1.2.0 ", style=f"bold black on {PRIMARY}")
    if ollama_status():
        header_text.append("  ● OLLAMA ", style="bold green")
    else:
        header_text.append("  ○ LOCAL MODE ", style="dim yellow")
    cfg = load_config()
    gpu = "NVENC GPU" if cfg.get("video", {}).get("use_nvenc", True) else "CPU"
    header_text.append(f" │ {gpu} ", style="bold cyan")
    return Align.center(header_text)


def make_footer(context="menu"):
    footer = Text()
    footer.append(" [↑/↓/J/K] ", style=f"bold {PRIMARY}")
    footer.append("Navigate  ", style="dim white")

    if context == "settings":
        footer.append("[ENTER] ", style=f"bold {SECONDARY}")
        footer.append("Change Value  ", style="dim white")
    else:
        footer.append("[ENTER] ", style=f"bold {SECONDARY}")
        footer.append("Select  ", style="dim white")
        footer.append("[L] ", style="bold cyan")
        footer.append("Logs  ", style="dim white")

    footer.append("[Q] ", style="bold red")
    footer.append("Quit", style="dim white")
    return Align.center(footer)


# ==================== ГОЛОВНЕ МЕНЮ ====================
MENU_ITEMS = [
    ("[+]", "Run Process",    "Нарізати всі відео в черзі на кліпи"),
    ("[↓]", "Download Video", "Завантажити ролик з YouTube (до 720p)"),
    ("[i]", "Clip Status",    "Черга, готові кліпи, статистика output"),
    ("[*]", "Settings",       "Банер, кроп, швидкість, NVENC, семантика"),
    ("[x]", "Exit",           "Завершити роботу і вийти"),
]


def render_frame(selected_index):
    term_height = shutil.get_terminal_size().lines
    logo = get_centered_logo_text()
    menu_content = Text()
    for idx, (icon, title, desc) in enumerate(MENU_ITEMS):
        title_f = f"{title:<14}"
        if idx == selected_index:
            menu_content.append(" > ", style=f"bold {PRIMARY}")
            menu_content.append(f" {icon} {title_f} ", style=SELECTED_BG)
            menu_content.append(f" │ {desc} ", style=f"italic {SELECTED_SUB}")
            menu_content.append("\n")
        else:
            menu_content.append("   ", style="dim")
            menu_content.append(f" {icon} {title_f} ", style="bold white")
            menu_content.append(f" │ {desc} ", style="dim gray")
            menu_content.append("\n")

    if STATE["running"]:
        status_line = Text()
        status_line.append("  ● PIPELINE ACTIVE ", style=f"bold black on {PRIMARY}")
        status_line.append(f"  {STATE['progress']['last_line'][:60]}", style="dim")
        status_block = Align.center(status_line)
    else:
        status_block = Text("")

    menu_panel = Panel(
        Align.center(menu_content),
        border_style=SECONDARY,
        padding=(1, 2),
        title=f"[bold {PRIMARY}] SYSTEM CONTROL [/bold {PRIMARY}]",
        subtitle=f"[dim {SECONDARY}]MUSORZALIV AUTO-CUTTER[/dim {SECONDARY}]",
        expand=False
    )

    content_height = 24
    top_padding = max(0, (term_height - content_height) // 2)
    screen = Group(
        Text("\n" * top_padding),
        make_header(),
        Text("\n"),
        Align.center(logo),
        Align.center(menu_panel),
        status_block,
        Text("\n"),
        make_footer("menu")
    )
    console.clear()
    console.print(screen)


def run_menu():
    selected_index = 0
    while True:
        render_frame(selected_index)
        key = readchar.readkey()

        if key == readchar.key.UP or key in ('k', 'K'):
            selected_index = (selected_index - 1) % len(MENU_ITEMS)
        elif key == readchar.key.DOWN or key in ('j', 'J'):
            selected_index = (selected_index + 1) % len(MENU_ITEMS)
        elif key in ('l', 'L'):
            return "logs"
        elif key in (readchar.key.ENTER, "\r", "\n"):
            return selected_index
        elif key in (readchar.key.CTRL_C, 'q', 'Q'):
            return len(MENU_ITEMS) - 1


# ==================== ЗАПУСК ПАЙПЛАЙНУ ====================
def _pipeline_worker(config):
    p = STATE["progress"]
    p.update({"phase": "working", "video": "", "video_i": 0,
              "clip_i": 0, "clip_n": 0, "clips_done": 0, "last_line": "", "error": ""})
    STATE["results"] = []

    review_cb = tui_review_callback if config.get("ranking", {}).get("review_clips", False) else None

    old_out, old_err = sys.stdout, sys.stderr
    tee = Tee()
    sys.stdout = tee
    sys.stderr = tee
    try:
        created = run_cutter(config, review_callback=review_cb)
        STATE["results"] = created or []
        p["phase"] = "done"
        log_line(f"✅ Пайплайн завершено. Створено кліпів: {len(STATE['results'])}")
    except Exception as e:
        p["phase"] = "error"
        p["error"] = str(e)
        log_line(f"Пайплайн упав з помилкою: {e}", error=True)
        import traceback
        for ln in traceback.format_exc().splitlines():
            log_line(ln, error=True)
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        STATE["running"] = False


def tui_review_callback(video_name, clips, analyzer=None):
    """Місток: run_cutter (у потоці) → інтерактивний екран огляду кліпів у TUI."""
    if not clips:
        return clips
    try:
        return review_clip_screen(video_name, clips, analyzer)
    except (EOFError, KeyboardInterrupt):
        # stdin закрився — не роняємо пайплайн, приймаємо всі кліпи
        log_line("Огляд перервано (EOF) — приймаю всі кліпи", error=True)
        return clips


def review_clip_screen(video_name, clips, analyzer=None):
    """
    Екран огляду кліпів перед рендером:
      ↑/↓ — навігація, SPACE — виключити/повернути, R — перегенерувати хук,
      A — прийняти всі, ENTER — рендерити вибрані, Q — пропустити відео.
    """
    STATE["review_active"] = True
    rc = Console(file=sys.__stdout__)

    def render(excluded, sel):
        logo = get_centered_logo_text()
        t = Table.grid(padding=(0, 2))
        t.add_column(width=3, justify="center")
        t.add_column(width=9, justify="right")
        t.add_column(width=5, justify="right")
        t.add_column(width=5, justify="right")
        t.add_column(width=24, overflow="ellipsis")
        t.add_column(overflow="ellipsis")

        for i, c in enumerate(clips):
            sb = c.get("score_breakdown", {})
            first_sent = (c.get("text") or "").split(".")[0][:52]
            num = f"{'>' if i == sel else ' '}{i + 1}"
            style = "dim strike" if i in excluded else ("bold black on #ffe600" if i == sel else "")
            t.add_row(
                num,
                f"{c['start']:.0f}s-{c['end']:.0f}s",
                f"{c['duration']:.0f}с",
                f"{sb.get('total', '?')}",
                (c.get("hook_title") or "")[:24],
                first_sent,
                style=style,
            )

        hints = Text()
        hints.append(" [SPACE] ", style=f"bold {PRIMARY}")
        hints.append("Вкл/викл  ", style="dim")
        hints.append("[R] ", style="bold cyan")
        hints.append("Новий хук  ", style="dim")
        hints.append("[A] ", style="bold green")
        hints.append("Всі  ", style="dim")
        hints.append("[ENTER] ", style="bold green")
        hints.append("Рендерити  ", style="dim")
        hints.append("[Q] ", style="bold red")
        hints.append("Пропустити відео", style="dim")

        panel = Panel(
            Group(Align.center(t), Text(""), Align.center(hints)),
            border_style=SECONDARY,
            padding=(1, 3),
            title=f"[bold {PRIMARY}] 📋 CLIP REVIEW — {video_name[:40]} [/bold {PRIMARY}]",
            subtitle=f"[dim {SECONDARY}]залишено {len(clips) - len(excluded)} з {len(clips)}[/dim {SECONDARY}]",
            expand=False,
        )
        rc.clear()
        rc.print(Group(
            Text("\n" * max(0, (shutil.get_terminal_size().lines - 24) // 2)),
            make_header(), Text("\n"), Align.center(logo),
            Align.center(panel),
        ))

    excluded = set()
    sel = 0
    try:
        while True:
            render(excluded, sel)
            try:
                key = readchar.readkey()
            except Exception:
                return clips  # stdin недоступний (EOF/термінал) — приймаємо все
            if key == readchar.key.UP or key in ('k', 'K'):
                sel = (sel - 1) % len(clips)
            elif key == readchar.key.DOWN or key in ('j', 'J'):
                sel = (sel + 1) % len(clips)
            elif key == ' ':
                if sel in excluded:
                    excluded.discard(sel)
                else:
                    excluded.add(sel)
            elif key in ('r', 'R') and analyzer is not None:
                new_hook = analyzer.regen_hook(clips[sel])
                clips[sel]["hook_title"] = new_hook
                log_line(f"Новий хук для кліпу #{sel + 1}: {new_hook}")
            elif key in ('a', 'A'):
                return clips
            elif key in (readchar.key.ENTER, "\r", "\n"):
                kept = [c for i, c in enumerate(clips) if i not in excluded]
                if not kept:
                    render(excluded, sel)
                return kept
            elif key in ('q', 'Q', readchar.key.ESC):
                return []
    finally:
        STATE["review_active"] = False


def bar(done, total, width=30):
    if total <= 0:
        return Text("░" * width, style="dim")
    n = int(width * min(1.0, done / max(1, total)))
    return Text("█" * n + "░" * (width - n), style=f"bold {PRIMARY}")


def run_process():
    config = load_config()
    input_dir = config.get("paths", {}).get("input_dir", "input_videos")

    if not STATE["running"]:
        queued = get_input_videos(input_dir)
        if not queued:
            console.clear()
            logo = get_centered_logo_text()
            console.print(Group(
                Text("\n" * max(0, (shutil.get_terminal_size().lines - 18) // 2)),
                make_header(), Text("\n"), Align.center(logo),
                Align.center(Panel(
                    "[bold yellow]⚠️  У черзі немає відео для нарізки![/bold yellow]\n\n"
                    f"[dim]Закинь файли в [bold]{input_dir}/[/bold] або скористайся пунктом "
                    "[bold][↓] Download Video[/bold].[/dim]\n\n[bold]Press ENTER to return...[/bold]",
                    border_style=SECONDARY, padding=(1, 4),
                    title=f"[bold {PRIMARY}] QUEUE EMPTY [/bold {PRIMARY}]",
                    expand=False,
                )),
            ))
            try:
                readchar.readkey()
            except (EOFError, KeyboardInterrupt):
                pass
            return

        STATE["running"] = True
        STATE["progress"]["video_n"] = len(queued)
        t = threading.Thread(target=_pipeline_worker, args=(config,), daemon=True)
        STATE["thread"] = t
        t.start()

    # Живий екран прогресу (консоль прив'язана до реального stdout, мимо Tee)
    live_console = Console(file=sys.__stdout__)
    logo = get_centered_logo_text()

    def get_progress_screen():
        p = STATE["progress"]
        term_height = shutil.get_terminal_size().lines
        top_padding = max(0, (term_height - 22) // 2)

        t = Table.grid(padding=(0, 2))
        t.add_column(justify="right", style="bold white")
        t.add_column(justify="left")
        if p["video_n"] > 0:
            t.add_row("Video:", Text(f"{p['video'] or '...'} ({min(p['video_i'], p['video_n'])}/{p['video_n']})", style="bold cyan"))
        else:
            t.add_row("Video:", Text("пошук відео в черзі...", style="dim"))
        if p["clip_n"] > 0:
            t.add_row("Clips:", Text.assemble(
                (f"{min(p['clips_done'], p['clip_n'])} / {p['clip_n']}", "bold green"),
                ("  ", ""),
                bar(p["clips_done"], p["clip_n"], 26),
            ))
        t.add_row("Status:", Text(p["last_line"] or "Ініціалізація...", style=f"italic {SECONDARY}"))

        spinner = Spinner("dots", text=Text("Pipeline engine active...", style=f"bold {PRIMARY}"), style=f"bold {PRIMARY}")
        content = Group(Align.center(t), Text(""), Align.center(spinner))
        panel = Panel(
            content,
            border_style=SECONDARY,
            padding=(1, 6),
            title=f"[bold {PRIMARY}] EXECUTING PIPELINE [/bold {PRIMARY}]",
            subtitle=f"[dim {SECONDARY}][L] logs у головному меню[/dim {SECONDARY}]",
            expand=False,
        )
        return Group(
            Text("\n" * top_padding),
            make_header(),
            Text("\n"),
            Align.center(logo),
            Align.center(panel),
        )

    live_console.clear()
    with Live(get_progress_screen(), console=live_console, refresh_per_second=10, screen=False,
              redirect_stdout=False, redirect_stderr=False) as live:
        notice_shown = False
        while STATE["running"]:
            if STATE.get("review_active"):
                # Огляд кліпів малює свій екран — Live завмирає, щоб не перемальовувати поверх
                if not notice_shown:
                    live.update(Align.center(Panel(
                        "[bold #ffe600]📋 ОГЛЯД КЛІПІВ — керуй у вікні огляду[/bold #ffe600]\n\n"
                        "[dim]SPACE — виключити │ R — новий хук │ ENTER — рендер вибраних[/dim]",
                        border_style=SECONDARY, padding=(1, 4),
                        title=f"[bold {PRIMARY}] REVIEW ACTIVE [/bold {PRIMARY}]",
                        expand=False,
                    )))
                    notice_shown = True
                time.sleep(0.1)
                continue
            if notice_shown:
                notice_shown = False
                live_console.clear()
            live.update(get_progress_screen())
            time.sleep(0.1)

    # Фінальний екран
    p = STATE["progress"]
    live_console.clear()
    term_height = shutil.get_terminal_size().lines
    top_padding = max(0, (term_height - 20) // 2)
    if p["phase"] == "error":
        body = f"[bold red]✗ Pipeline failed:[/bold red] [dim]{p['error'][:100]}[/dim]\n\n[dim]Дивіться [bold][L] Logs[/bold] для деталей. Press ENTER to return...[/dim]"
        color, title = "red", " STATUS: FAILED "
    else:
        n = len(STATE["results"])
        body = f"[bold green]✔ Pipeline finished! Створено кліпів: {n}[/bold green]\n\n[dim]Готові кліпи в output_clips/. Press ENTER to return to menu...[/dim]"
        color, title = "green", " STATUS: COMPLETED "
    live_console.print(Group(
        Text("\n" * top_padding),
        make_header(),
        Text("\n"),
        Align.center(get_centered_logo_text()),
        Align.center(Panel(
            Align.center(body),
            border_style=color,
            padding=(1, 4),
            title=f"[bold {color}]{title}[/bold {color}]",
            expand=False,
        )),
    ))
    try:
        readchar.readkey()
    except (EOFError, KeyboardInterrupt):
        pass


# ==================== ЗАВАНТАЖЕННЯ З YOUTUBE ====================
def run_download():
    config = load_config()
    input_dir = config.get("paths", {}).get("input_dir", "input_videos")

    console.clear()
    logo = get_centered_logo_text()
    console.print(Group(
        Text("\n" * max(0, (shutil.get_terminal_size().lines - 16) // 2)),
        make_header(), Text("\n"), Align.center(logo),
        Align.center(Panel(
            f"[dim]Встав URL відео YouTube (до 720p). Файл потрапить у [bold]{input_dir}/[/bold][/dim]\n"
            "[dim]Порожній ввод — повернення в меню.[/dim]",
            border_style=SECONDARY, padding=(1, 4),
            title=f"[bold {PRIMARY}] 📥 DOWNLOAD VIDEO [/bold {PRIMARY}]",
            expand=False,
        )),
    ))
    term_width = shutil.get_terminal_size().columns
    left_pad = " " * max(0, (term_width - 40) // 2)
    url = console.input(f"{left_pad}[bold {PRIMARY}]> [/bold {PRIMARY}]").strip()
    if not url:
        return

    STATE["progress"]["phase"] = "working"
    STATE["progress"]["last_line"] = f"Завантаження: {url[:70]}"
    STATE["running"] = True

    def worker():
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = Tee()
        sys.stderr = Tee()
        try:
            files = download_youtube_video(url, output_dir=input_dir, max_height=720)
            STATE["results"] = files or []
            STATE["progress"]["phase"] = "done"
            log_line(f"✅ Завантажено файлів: {len(STATE['results'])}")
        except Exception as e:
            STATE["progress"]["phase"] = "error"
            STATE["progress"]["error"] = str(e)
            log_line(f"Помилка завантаження: {e}", error=True)
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            STATE["running"] = False

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    live_console = Console(file=sys.__stdout__)
    live_console.clear()
    with Live(console=live_console, refresh_per_second=8, redirect_stdout=False, redirect_stderr=False) as live:
        while STATE["running"]:
            tail = list(STATE["logs"])[-1] if STATE["logs"] else "Підключення..."
            live.update(Panel(
                Align.center(Group(
                    Spinner("dots", text=Text("Завантаження відео...", style=f"bold {PRIMARY}"), style=f"bold {PRIMARY}"),
                    Text(""),
                    Text(tail[:100], style="dim"),
                )),
                border_style=SECONDARY, padding=(1, 6),
                title=f"[bold {PRIMARY}] 📥 DOWNLOADING [/bold {PRIMARY}]",
                expand=False,
            ))
            time.sleep(0.12)

    p = STATE["progress"]
    live_console.clear()
    if p["phase"] == "error":
        body = f"[bold red]✗ Download failed:[/bold red] [dim]{p['error'][:100]}[/dim]"
        color = "red"
    else:
        names = "\n".join(f"  👉 [cyan]{os.path.basename(f)}[/cyan]" for f in STATE["results"]) or "  [dim](без нових файлів)[/dim]"
        body = f"[bold green]✔ Завантажено![/bold green]\n{names}\n\n[dim]Можеш одразу тиснути [bold][+] Run Process[/bold]. Press ENTER...[/dim]"
        color = "green"
    live_console.print(Align.center(Panel(
        Align.center(body),
        border_style=color, padding=(1, 4),
        title=f"[bold {color}] DOWNLOAD {( 'OK' if color == 'green' else 'FAILED' )} [/bold {color}]",
        expand=False,
    )))
    try:
        readchar.readkey()
    except (EOFError, KeyboardInterrupt):
        pass


# ==================== СТАТУС КЛІПІВ ====================
def run_clip_status():
    config = load_config()
    input_dir = config.get("paths", {}).get("input_dir", "input_videos")
    output_dir = config.get("paths", {}).get("output_dir", "output_clips")

    queued = get_input_videos(input_dir)
    clips = glob.glob(os.path.join(output_dir, "*.mp4"))
    clips.sort(key=os.path.getmtime, reverse=True)
    total_mb = sum(os.path.getsize(c) for c in clips) / (1024 * 1024)

    rows = Text()
    if not clips:
        rows.append("  (порожньо — запусти [bold][+] Run Process[/bold])", style="dim")
    else:
        for c in clips[:6]:
            sz = os.path.getsize(c) / (1024 * 1024)
            age = time.strftime("%d.%m %H:%M", time.localtime(os.path.getmtime(c)))
            rows.append(f"  {Path(c).name[:58]:<58} ", style="white")
            rows.append(f"{sz:6.1f} MB  ", style="bold cyan")
            rows.append(f"{age}\n", style="dim")
        if len(clips) > 6:
            rows.append(f"  ... і ще {len(clips) - 6} кліпів", style="dim")

    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="bold white")
    table.add_column(justify="left")
    table.add_row("Queue:", Text(f"{len(queued)} відео в {input_dir}/", style="bold yellow"))
    table.add_row("Clips done:", Text(f"{len(clips)} ({total_mb:.1f} MB) у {output_dir}/", style="bold green"))
    if STATE["running"]:
        p = STATE["progress"]
        table.add_row("Now:", Text(f"{p['video'] or '...'} — кліп {p['clip_i']}/{p['clip_n'] or '?'}", style=f"bold {PRIMARY}"))

    panel = Panel(
        Group(Align.center(table), Text(""), rows),
        border_style=SECONDARY,
        padding=(1, 4),
        title=f"[bold {PRIMARY}] 📊 CLIP STATUS [/bold {PRIMARY}]",
        subtitle=f"[dim {SECONDARY}]Press Q or ENTER to return[/dim {SECONDARY}]",
        expand=False,
    )

    console.clear()
    console.print(Group(
        Text("\n" * max(0, (shutil.get_terminal_size().lines - 26) // 2)),
        make_header(), Text("\n"), Align.center(get_centered_logo_text()),
        Align.center(panel),
    ))
    while True:
        key = readchar.readkey()
        if key in ('q', 'Q', '\r', '\n', readchar.key.ESC, readchar.key.ENTER, readchar.key.BACKSPACE):
            break


# ==================== ЛОГИ ====================
def run_logs_viewer():
    lines = list(STATE["logs"])[-40:]
    while True:
        content = Text()
        if not lines:
            content.append("  (логи порожні — запусти [bold][+] Run Process[/bold])", style="dim")
        else:
            for ln in lines:
                if "[ERROR]" in ln:
                    content.append(ln + "\n", style="bold red")
                elif "✓" in ln or "✅" in ln:
                    content.append(ln + "\n", style="green")
                elif "⚠️" in ln:
                    content.append(ln + "\n", style="yellow")
                else:
                    content.append(ln + "\n", style="white")

        term_height = shutil.get_terminal_size().lines
        panel_height = max(10, term_height - 8)
        shown = content.split("\n")
        if len(shown) < panel_height:
            content.append("\n" * (panel_height - len(shown)))

        status = " ● LIVE " if STATE["running"] else " ○ IDLE "
        panel = Panel(
            content,
            border_style=SECONDARY,
            padding=(1, 2),
            title=f"[bold {PRIMARY}] 📋 PIPELINE LOGS [/bold {PRIMARY}]",
            subtitle=f"[dim {SECONDARY}]{status} [R] Refresh │ Q/ENTER — back[/dim {SECONDARY}]",
            expand=True,
        )
        console.clear()
        console.print(Group(Text("\n"), make_header(), Text("\n"), panel))

        key = readchar.readkey()
        if key in ('r', 'R'):
            lines = list(STATE["logs"])[-40:]
            continue
        if key in ('q', 'Q', '\r', '\n', readchar.key.ESC, readchar.key.ENTER, readchar.key.BACKSPACE):
            break


# ==================== НАЛАШТУВАННЯ ====================
def save_config(config: dict, path: str = "config.yaml"):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def build_settings_items(config: dict):
    promo = config.setdefault("promo_midroll", {})
    video = config.setdefault("video", {})
    whisper = config.setdefault("whisper", {})
    clips = config.setdefault("clips", {})
    analysis = config.setdefault("analysis", {})

    def apply_midroll():
        promo["enabled"] = not promo.get("enabled", False)

    def apply_crop():
        video["crop_mode"] = "blur_background" if video.get("crop_mode", "center_crop") == "center_crop" else "center_crop"

    def apply_speed():
        opts = [1.0, 1.03, 1.04, 1.05]
        cur = float(video.get("speed", 1.04))
        video["speed"] = opts[(opts.index(cur) + 1) % len(opts)] if cur in opts else opts[0]

    def apply_whisper():
        opts = ["tiny", "base", "small"]
        cur = whisper.get("model_size", "base")
        whisper["model_size"] = opts[(opts.index(cur) + 1) % len(opts)] if cur in opts else "base"

    def apply_nvenc():
        video["use_nvenc"] = not video.get("use_nvenc", True)

    def apply_render_workers():
        opts = [1, 2, 3]
        cur = int(video.get("render_workers", 2))
        video["render_workers"] = opts[(opts.index(cur) + 1) % len(opts)] if cur in opts else 2

    def apply_snap():
        clips["sentence_snap"] = not clips.get("sentence_snap", True)

    def apply_llm():
        analysis["use_llm"] = not analysis.get("use_llm", True)

    return [
        {"idx": "[1]", "key": "Mid-roll Ad (Banner)", "desc": "Вставка реклами посеред кліпу",
         "value": lambda: "ON (Active)" if promo.get("enabled") else "OFF (Warmup)", "apply": apply_midroll},
        {"idx": "[2]", "key": "Ad Video Path", "desc": "Шлях до рекламного ролика",
         "value": lambda: str(promo.get("video_path", "assets/baner.mp4")), "apply": "text"},
        {"idx": "[3]", "key": "9:16 Crop Format", "desc": "center_crop (зум) / blur_background",
         "value": lambda: str(video.get("crop_mode", "center_crop")), "apply": apply_crop},
        {"idx": "[4]", "key": "Video Speed", "desc": "Прискорення для збиття хешу (1.00-1.05)",
         "value": lambda: f"{float(video.get('speed', 1.04)):.2f}x", "apply": apply_speed},
        {"idx": "[5]", "key": "Whisper Model", "desc": "tiny (швидко) / base / small (точно)",
         "value": lambda: str(whisper.get("model_size", "base")), "apply": apply_whisper},
        {"idx": "[6]", "key": "Clip Duration", "desc": "Контекст сам вирішує: 30с - 2хв",
         "value": lambda: "Авто (за контекстом)", "apply": None},
        {"idx": "[7]", "key": "Hardware Render", "desc": "NVENC (GPU) або Software (CPU)",
         "value": lambda: "NVENC (NVIDIA GPU)" if video.get("use_nvenc", True) else "Software (CPU)", "apply": apply_nvenc},
        {"idx": "[8]", "key": "Render Streams", "desc": "Скільки кліпів кодується одночасно",
         "value": lambda: f"{int(video.get('render_workers', 2))} паралельно", "apply": apply_render_workers},
        {"idx": "[9]", "key": "Semantic Cut", "desc": "Різання за реченнями + вердикти міфів",
         "value": lambda: "ON (Semantic)" if clips.get("sentence_snap", True) else "OFF (Raw timecodes)", "apply": apply_snap},
        {"idx": "[10]", "key": "Qwen LLM (Ollama)", "desc": "Локальний LLM для вибору моментів",
         "value": lambda: "ON (Ollama)" if analysis.get("use_llm", True) else "OFF (Heuristic)", "apply": apply_llm},
        {"idx": "[0]", "key": "⬅ Back to Main Menu", "desc": "Повернутись у головне меню",
         "value": lambda: "", "apply": None},
    ]


def prompt_text_input(item, config):
    term_height = shutil.get_terminal_size().lines
    console.clear()
    console.print(Group(
        Text("\n" * max(0, (term_height - 16) // 2)),
        make_header(),
        Text("\n"),
        Align.center(Panel(
            f"Current Value: [cyan]{item['value']()}[/cyan]\n\nEnter new value for [bold {PRIMARY}]{item['key']}[/bold {PRIMARY}]:",
            border_style=SECONDARY,
            padding=(1, 4),
            title=f"[bold {PRIMARY}] EDIT PARAMETER [/bold {PRIMARY}]"
        )),
        Text("\n")
    ))
    term_width = shutil.get_terminal_size().columns
    left_pad = " " * max(0, (term_width - 40) // 2)
    new_val = console.input(f"{left_pad}[bold {PRIMARY}]> [/bold {PRIMARY}]").strip()
    if new_val:
        config.setdefault("promo_midroll", {})["video_path"] = new_val
        save_config(config)


def render_settings_frame(items, selected_index):
    logo = get_centered_logo_text()
    table = Table(box=box.SQUARE, border_style=SECONDARY, header_style=f"bold {PRIMARY}")
    table.add_column("#", justify="center", width=4)
    table.add_column("Parameter", width=24)
    table.add_column("Current Value", width=22)
    table.add_column("Description", style="dim white", overflow="ellipsis")

    last = len(items) - 1
    for i, item in enumerate(items):
        if i == selected_index:
            style = SELECTED_BG
            idx_display = f"> {item['idx']}" if i != last else item['idx']
        else:
            style = ""
            idx_display = f"  {item['idx']}" if i != last else item['idx']
        table.add_row(idx_display, item['key'], item['value'](), item['desc'], style=style)

    term_height = shutil.get_terminal_size().lines
    content_height = 26
    top_padding = max(0, (term_height - content_height) // 2)

    screen = Group(
        Text("\n" * top_padding),
        make_header(),
        Text("\n"),
        Align.center(logo),
        Align.center(table),
        Text("\n"),
        make_footer("settings")
    )
    console.clear()
    console.print(screen)


def run_settings():
    while True:
        config = load_config()
        items = build_settings_items(config)
        selected_index = 0
        redraw = True
        while True:
            if redraw:
                render_settings_frame(items, selected_index)
                redraw = False
            key = readchar.readkey()

            if key == readchar.key.UP or key in ('k', 'K'):
                selected_index = (selected_index - 1) % len(items)
                redraw = True
            elif key == readchar.key.DOWN or key in ('j', 'J'):
                selected_index = (selected_index + 1) % len(items)
                redraw = True
            elif key in (readchar.key.ENTER, "\r", "\n"):
                if selected_index == len(items) - 1:
                    return
                item = items[selected_index]
                if item["apply"] == "text":
                    prompt_text_input(item, config)
                elif callable(item["apply"]):
                    item["apply"]()
                    save_config(config)
                redraw = True
            elif key in (readchar.key.CTRL_C, 'q', 'Q', '0'):
                return


# ==================== ГОЛОВНИЙ ЦИКЛ ====================
def main():
    ensure_project_structure()

    while True:
        choice = run_menu()

        if choice == "logs":
            run_logs_viewer()
        elif choice == 0:
            run_process()
        elif choice == 1:
            run_download()
        elif choice == 2:
            run_clip_status()
        elif choice == 3:
            run_settings()
        elif choice == 4:
            console.clear()
            console.print(Align.center(Panel(
                f"[bold {SECONDARY}]👋 MUSORZALIV завершив роботу. До зустрічі![/bold {SECONDARY}]",
                border_style="dim")))
            sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        console.print("\n\n[bold yellow]Програму перервано. До зустрічі![/bold yellow]\n")
        sys.exit(0)
    except Exception:
        import traceback as _tb
        _log_file.write("\n[FATAL] " + _tb.format_exc() + "\n")
        _log_file.flush()
        console.print_exception()
        console.print(f"[bold red]Повний звіт у {LOG_PATH}[/bold red]")
        sys.exit(1)
