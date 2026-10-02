#!/bin/sh
set -eu

cd "$(dirname "$0")"
app_dir="stt_binary.app"
mkdir -p "$app_dir/Contents/MacOS"
cp Info.plist "$app_dir/Contents/Info.plist"
swiftc stt_binary.swift -o "$app_dir/Contents/MacOS/stt_binary" \
    -framework Speech -framework AVFoundation
codesign --force --sign - "$app_dir"
printf '%s\n' "Built $app_dir."
