#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
applications="${1:-/Applications}"
name="Twitch Shorts"
target="$applications/$name.app"
work="$(mktemp -d "${TMPDIR:-/tmp}/twitch-shorts.XXXXXXXX")"
trap 'rm -rf "$work"' EXIT

app="$work/$name.app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources" "$applications"
plist="$app/Contents/Info.plist"
cat > "$plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key><string>TwitchShorts</string>
    <key>CFBundleIdentifier</key><string>local.musorzaliv.twitch-shorts</string>
    <key>CFBundleName</key><string>Twitch Shorts</string>
    <key>CFBundleDisplayName</key><string>Twitch Shorts</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>CFBundleIconFile</key><string>TwitchShorts.icns</string>
    <key>LSMinimumSystemVersion</key><string>12.0</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST
/usr/bin/plutil -insert TwitchWorkspace -string "$root" "$plist"
/usr/bin/swiftc -parse-as-library -O -target "$(uname -m)-apple-macos12" \
    "$root/macos/TwitchShorts.swift" -o "$app/Contents/MacOS/TwitchShorts"
"$root/.venv-twitch/bin/python" "$root/macos/make_icon.py" "$work/TwitchShorts.iconset"
/usr/bin/iconutil -c icns "$work/TwitchShorts.iconset" -o "$app/Contents/Resources/TwitchShorts.icns"
/usr/bin/plutil -lint "$plist"
/usr/bin/codesign --force --sign - "$app"

if [ -e "$target" ]; then
    echo "Уже существует $target. Переместите предыдущую версию перед переустановкой." >&2
    exit 1
fi
/usr/bin/ditto "$app" "$target"
/usr/bin/codesign --verify "$target"
echo "$target"
