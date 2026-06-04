#!/usr/bin/env bash
# Record scripts/demo.sh → assets/demo-v0.10.gif (asciinema + agg)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAST="${ROOT}/assets/demo.cast"
GIF="${ROOT}/assets/demo-v0.10.gif"
LEGACY="${ROOT}/assets/demo.gif"
export DATABASE_URL="${DATABASE_URL:-postgresql://cuba:memorys2026@127.0.0.1:5488/brain}"
export CUBA_BINARY_PATH="${CUBA_BINARY_PATH:-$ROOT/rust/target/release/cuba-memorys}"
export RECORD_DEMO=1

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
  --command "bash -lc 'export TERM=xterm-256color; export RECORD_DEMO=1; bash \"$ROOT/scripts/demo.sh\"'" \
  --idle-time-limit 1.2 \
  --overwrite \
  --title "cuba-memorys v0.10.0 demo"

echo "Rendering GIF → $GIF"
agg "$CAST" "$GIF" \
  --cols 96 \
  --rows 38 \
  --theme monokai \
  --font-family "JetBrains Mono,Fira Code,Consolas,DejaVu Sans Mono" \
  --speed 0.55 \
  --fps-cap 8 \
  --last-frame-duration 5 \
  --no-loop

# Keep legacy path for older links; README uses demo-v0.10.gif
cp -f "$GIF" "$LEGACY"

ls -lh "$GIF" "$LEGACY"
file "$GIF"
echo "Done. Commit assets/demo-v0.10.gif + assets/demo.gif"