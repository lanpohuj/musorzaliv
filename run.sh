#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OS="$(uname -s)"

if [ "${1:-}" = "--twitch" ]; then
    shift
    exec "$DIR/twitch.sh" "$@"
fi

# ── 1. Python 3 ──────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "[!] Python 3 не знайдено."
    if [ "$OS" = "Darwin" ]; then
        echo "    macOS: brew install python  або  https://www.python.org/downloads/"
    elif command -v pacman &>/dev/null; then
        echo "    Manjaro/Arch:  sudo pacman -S python"
    elif command -v apt &>/dev/null; then
        echo "    Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    elif command -v dnf &>/dev/null; then
        echo "    Fedora:        sudo dnf install python3"
    fi
    exit 1
fi

# ── 2. Віртуальне оточення + залежності ─────────────────────────────
if [ -d "$DIR/.venv" ] && ! "$DIR/.venv/bin/python3" -c "import sys" 2>/dev/null; then
    echo "[!] Виявлено неробочий/чужий .venv. Видаляємо для чистого налаштування..."
    rm -rf "$DIR/.venv"
fi

if [ ! -d "$DIR/.venv" ]; then
    echo "[*] Створення віртуального оточення .venv..."
    python3 -m venv "$DIR/.venv"
    "$DIR/.venv/bin/pip" install --upgrade pip
    echo "[*] Встановлення залежностей (faster-whisper, yt-dlp, rich, ...)..."
    "$DIR/.venv/bin/pip" install -r "$DIR/requirements.txt"
fi
PY_RUNNER="$DIR/.venv/bin/python3"

# ── 3. FFmpeg ────────────────────────────────────────────────────────
# На macOS: bin/ffmpeg — мертвий symlink на nix store. Видаляємо його,
# щоб скрипт пішов шукати системний ffmpeg або підказав brew.
if [ "$OS" = "Darwin" ]; then
    # Якщо в bin/ є NixOS symlink — не використовуємо
    if [ -L "$DIR/bin/ffmpeg" ]; then
        LINK_TARGET="$(readlink "$DIR/bin/ffmpeg" || true)"
        if echo "$LINK_TARGET" | grep -q '/nix/store'; then
            # nix-symlink не працює на mac — ігноруємо
            true
        else
            export PATH="$DIR/bin:$PATH"
        fi
    elif [ -x "$DIR/bin/ffmpeg" ]; then
        export PATH="$DIR/bin:$PATH"
    fi

    if ! command -v ffmpeg &>/dev/null; then
        echo "[!] FFmpeg не знайдено. Встанови через Homebrew:"
        echo "    brew install ffmpeg"
        echo ""
        echo "    Або завантаж статичний білд: https://evermeet.cx/ffmpeg/"
        echo "    і поклади ffmpeg + ffprobe у папку bin/ поруч з run.sh"
        exit 1
    fi
else
    # Linux
    if [ -x "$DIR/bin/ffmpeg" ] && [ ! -L "$DIR/bin/ffmpeg" ]; then
        export PATH="$DIR/bin:$PATH"
    elif [ -L "$DIR/bin/ffmpeg" ] && [ -e "$DIR/bin/ffmpeg" ]; then
        # symlink існує і таргет живий
        export PATH="$DIR/bin:$PATH"
    fi

    if ! command -v ffmpeg &>/dev/null; then
        echo "[!] FFmpeg не знайдено — потрібен для нарізки й рендеру."
        if command -v pacman &>/dev/null; then
            echo "[*] Manjaro/Arch виявлено. Встановити ffmpeg зараз (sudo pacman -S --needed ffmpeg)? [y/N]"
            read -r ans
            if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
                sudo pacman -S --needed ffmpeg
            else
                echo "    Потім сам: sudo pacman -S ffmpeg"
            fi
        elif command -v apt &>/dev/null; then
            echo "    Ubuntu/Debian: sudo apt install ffmpeg"
        elif command -v dnf &>/dev/null; then
            echo "    Fedora:        sudo dnf install ffmpeg"
        fi
    fi

    # ── 4. NixOS: гарантуємо доступ до системних C-бібліотек ────────
    if grep -qi '^ID=nixos' /etc/os-release 2>/dev/null; then
        export LD_LIBRARY_PATH="${NIX_LD_LIBRARY_PATH:-/run/current-system/sw/share/nix-ld/lib}:$LD_LIBRARY_PATH"
        if ! command -v ffmpeg &>/dev/null && [ ! -x "$DIR/bin/ffmpeg" ]; then
            mkdir -p "$DIR/bin"
            FFMPEG_STORE=$(nix-build '<nixpkgs>' -A ffmpeg --no-out-link 2>/dev/null || true)
            if [ -n "$FFMPEG_STORE" ]; then
                ln -sf "$FFMPEG_STORE/bin/ffmpeg" "$DIR/bin/ffmpeg"
                ln -sf "$FFMPEG_STORE/bin/ffprobe" "$DIR/bin/ffprobe"
                export PATH="$DIR/bin:$PATH"
            fi
        fi
    fi
fi

# ── 5. macOS: підказка про config ────────────────────────────────────
if [ "$OS" = "Darwin" ]; then
    if grep -q 'use_nvenc: true' "$DIR/config.yaml" 2>/dev/null; then
        echo "[i] macOS: NVENC недоступний. У config.yaml змінено use_nvenc: false автоматично."
        sed -i '' 's/use_nvenc: true/use_nvenc: false/' "$DIR/config.yaml"
    fi
fi

# ── 6. Запуск: без аргументів — TUI, інакше CLI ──────────────────────
if [ $# -eq 0 ] || [ "$1" == "--tui" ]; then
    exec "$PY_RUNNER" "$DIR/tui.py"
else
    exec "$PY_RUNNER" "$DIR/main.py" "$@"
fi
