"""
Donor channel video downloader using yt-dlp.
Automatically fetches fresh CS2 guide videos for cutting.

YouTube тепер вимагає JavaScript-рушій (deno/node) для отримання валідних
посилань — без нього yt-dlp падає з HTTP 403. Модуль сам знаходить рушій
у системі і передає його yt-dlp явно.
"""

import os
import sys
import shutil
import subprocess
import json
from pathlib import Path
from typing import List


DONOR_CHANNELS = [
    "https://www.youtube.com/@NartOutHere/videos",
    "https://www.youtube.com/@Austincs/videos"
]

# Пошук JS-рушія: спочатку у профілі nix, потім у PATH
_RUNTIME_CANDIDATES = [
    ("deno", ["deno"]),
    ("node", ["node", "nodejs"]),
    ("bun", ["bun"]),
]


def find_js_runtime() -> str:
    """Повертає 'runtime:/шлях/до/бінарника' або '' якщо рушія нема."""
    nix_profile = os.path.expanduser("~/.nix-profile/bin")
    for runtime, names in _RUNTIME_CANDIDATES:
        for name in names:
            p = shutil.which(name)
            if p:
                return f"{runtime}:{p}"
            cand = os.path.join(nix_profile, name)
            if os.path.exists(cand) and os.access(cand, os.X_OK):
                return f"{runtime}:{cand}"
    return ""


def get_js_runtime_args() -> List[str]:
    """Аргументи yt-dlp для JS-рушія (якщо знайдений)."""
    rt = find_js_runtime()
    if rt:
        return ["--js-runtimes", rt]
    # Нічого не знайшли — yt-dlp спробує сам; при фейлі покажемо підказку
    return []


def get_ytdlp_cmd() -> List[str]:
    """
    Returns the command list to invoke yt-dlp reliably in any environment.
    Uses sys.executable -m yt_dlp if available to bypass any broken shebangs
    from copied virtual environments.
    """
    try:
        import yt_dlp
        return [sys.executable, "-m", "yt_dlp"]
    except ImportError:
        pass

    system_bin = shutil.which("yt-dlp")
    if system_bin:
        return [system_bin]

    return ["yt-dlp"]


def _js_hint() -> str:
    if command_exists("pacman"):
        pkg = "Manjaro/Arch:  sudo pacman -S deno   (або node)"
    elif command_exists("apt"):
        pkg = "Ubuntu/Debian: sudo apt install nodejs   (або curl -fsSL https://deno.land/install.sh | sh)"
    elif command_exists("dnf"):
        pkg = "Fedora:        sudo dnf install nodejs"
    elif command_exists("nix-build"):
        pkg = "NixOS:         nix profile install nixpkgs#deno"
    else:
        pkg = "Встанови deno (https://deno.land) або nodejs"
    return (
        "YouTube вимагає JavaScript-рушій для завантажень (yt-dlp без нього отримує 403).\n"
        f"    {pkg}"
    )


def command_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _run_download(cmd: List[str], retries: int = 2) -> subprocess.CompletedProcess:
    """Запуск yt-dlp з ретраями (YouTube інше віддає 403/429 мінливо)."""
    last = None
    for attempt in range(1, retries + 1):
        last = subprocess.run(cmd)
        if last.returncode == 0:
            return last
        if attempt < retries:
            print(f"[!] Спроба {attempt} не вдалась (код {last.returncode}). Повторюю...")
    return last


def get_ffmpeg_location_args() -> List[str]:
    """Показуємо yt-dlp де лежить ffmpeg (у нас він в bin/, а не в PATH)."""
    local_bin = os.path.abspath("bin/ffmpeg")
    if os.path.exists(local_bin) and os.access(local_bin, os.X_OK):
        return ["--ffmpeg-location", os.path.dirname(local_bin)]
    system_bin = shutil.which("ffmpeg")
    if system_bin:
        return ["--ffmpeg-location", os.path.dirname(system_bin)]
    return []


def download_youtube_video(
    url: str,
    output_dir: str = "input_videos",
    max_height: int = 720
) -> List[str]:
    """
    Downloads a video from YouTube URL into output_dir capped at max_height (default 720p).
    Returns list of newly downloaded file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    before_files = set(os.listdir(output_dir))

    cmd = get_ytdlp_cmd() + get_js_runtime_args() + get_ffmpeg_location_args() + [
        "--no-playlist",
        "-f", f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best",
        "--merge-output-format", "mp4",
        "-o", os.path.join(output_dir, "%(title).80B.%(ext)s"),
        url.strip()
    ]

    print(f"[*] Завантаження відео з YouTube (макс. якість {max_height}p)...")
    res = _run_download(cmd)
    if res.returncode != 0:
        if not find_js_runtime():
            print(f"[!] {_js_hint()}")
        raise RuntimeError(f"yt-dlp завершився з кодом {res.returncode}")

    after_files = set(os.listdir(output_dir))
    new_files = [os.path.join(output_dir, f) for f in (after_files - before_files) if not f.startswith(".")]
    return new_files


def fetch_channel_videos(
    channels: List[str] = None,
    output_dir: str = "input_videos",
    max_videos_per_channel: int = 2,
    archive_file: str = "assets/downloaded_archive.txt"
) -> List[str]:
    """
    Downloads the latest videos from donor channels using yt-dlp.
    Maintains an archive to never re-download the same video.
    """
    if channels is None:
        channels = DONOR_CHANNELS

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(archive_file)), exist_ok=True)

    downloaded = []

    for chan in channels:
        print(f"[*] Перевірка нових відео з каналу: {chan}...")
        cmd = get_ytdlp_cmd() + get_js_runtime_args() + get_ffmpeg_location_args() + [
            "--no-playlist",
            "--playlist-end", str(max_videos_per_channel),
            "--download-archive", archive_file,
            "-f", "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", f"{output_dir}/%(title)s.%(ext)s",
            chan
        ]
        try:
            res = _run_download(cmd)
            if res.returncode == 0:
                print(f"[✓] Завершено перевірку каналу {chan}")
            elif not find_js_runtime():
                print(f"[!] {_js_hint()}")
        except Exception as e:
            print(f"[!] Помилка завантаження з {chan}: {e}")

    return downloaded


if __name__ == "__main__":
    fetch_channel_videos()
