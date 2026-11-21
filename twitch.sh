#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

for tool in ffmpeg ffprobe; do
    if ! command -v "$tool" >/dev/null && [ ! -x "bin/$tool" ]; then
        echo "Требуется $tool с поддержкой libass. На macOS установите FFmpeg через Homebrew."
        exit 1
    fi
done

if [ ! -x .venv-twitch/bin/python ]; then
    python_bin="${TWITCH_PYTHON:-python3.12}"
    if ! command -v "$python_bin" >/dev/null; then
        echo "Нужен Python 3.12. Укажите путь через TWITCH_PYTHON или установите python@3.12."
        exit 1
    fi
    if ! "$python_bin" -c 'import pyexpat' 2>/dev/null; then
        echo "В $python_bin не загружается pyexpat. Укажите исправный Python 3.12 через TWITCH_PYTHON."
        exit 1
    fi
    "$python_bin" -m venv .venv-twitch
fi
if [ ! -f .venv-twitch/.installed ] || [ requirements-twitch.txt -nt .venv-twitch/.installed ] || [ requirements.txt -nt .venv-twitch/.installed ]; then
    .venv-twitch/bin/python -m pip install -r requirements-twitch.txt
    touch .venv-twitch/.installed
fi

exec .venv-twitch/bin/python -m pipeline.twitch_web "$@"
