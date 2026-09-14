#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party"
mkdir -p "$THIRD_PARTY"

if [ ! -d "$THIRD_PARTY/libsm64" ]; then
  git clone --depth 1 https://github.com/libsm64/libsm64.git "$THIRD_PARTY/libsm64"
fi

make -C "$THIRD_PARTY/libsm64" lib -j"$(getconf _NPROCESSORS_ONLN)"
ls -la "$THIRD_PARTY/libsm64/dist"

if [ ! -f "$ROOT/roms/baserom.us.z64" ]; then
  echo
  echo "Place a Super Mario 64 US ROM at roms/baserom.us.z64 before running experiments."
  echo "libsm64 reads Mario's animation and texture data from it at runtime."
fi
