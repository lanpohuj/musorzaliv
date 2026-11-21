#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$DIR/bin:$PATH"
export LD_LIBRARY_PATH="${NIX_LD_LIBRARY_PATH:-/run/current-system/sw/share/nix-ld/lib}:$LD_LIBRARY_PATH"

if [ -x "$DIR/.venv/bin/python3" ] && "$DIR/.venv/bin/python3" -c "import sys" 2>/dev/null; then
    exec "$DIR/.venv/bin/python3" "$DIR/tui.py" "$@"
else
    exec python3 "$DIR/tui.py" "$@"
fi
