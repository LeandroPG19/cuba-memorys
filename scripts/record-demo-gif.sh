#!/usr/bin/env bash
# Record scripts/demo.sh → assets/demo.gif (asciinema + agg)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAST="${ROOT}/assets/demo.cast"
GIF="${ROOT}/assets/demo.gif"
export DATABASE_URL="${DATABASE_URL:-postgresql://cuba:memorys2026@127.0.0.1:5488/brain}"
export CUBA_BINARY_PATH="${CUBA_BINARY_PATH:-$ROOT/rust/target/release/cuba-memorys}"

if [[ ! -x "$CUBA_BINARY_PATH" ]]; then
  echo "Building release binary..."
  (cd "$ROOT/rust" && cargo build --release)
fi

command -v asciinema >/dev/null || { echo "asciinema required"; exit 1; }
command -v agg >/dev/null || { echo "agg required (asciinema-agg)"; exit 1; }

mkdir -p "$ROOT/assets"
rm -f "$CAST"

echo "Recording terminal demo → $CAST"
asciinema rec "$CAST" \
  --command "bash -lc 'export TERM=xterm-256color; bash \"$ROOT/scripts/demo.sh\"'" \
  --idle-time-limit 0.4 \
  --overwrite \
  --title "cuba-memorys v0.10.0 demo"

echo "Rendering GIF → $GIF"
agg "$CAST" "$GIF" \
  --cols 88 \
  --rows 32 \
  --theme dracula \
  --font-family "JetBrains Mono,Fira Code,Consolas,DejaVu Sans Mono" \
  --speed 1.15 \
  --fps-cap 12 \
  --last-frame-duration 2.5

ls -lh "$GIF"
file "$GIF"