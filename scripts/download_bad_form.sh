#!/usr/bin/env bash
# download_bad_form.sh
# Downloads 25 bad-form videos per exercise from YouTube.
# Re-runnable: skips already-downloaded files.
# Usage: bash download_bad_form.sh
#
# If YouTube rate-limits you, wait ~10 minutes and re-run — it picks up where it left off.

set -uo pipefail

BASE="$(cd "$(dirname "$0")" && pwd)/videos"
COOKIE_ARGS=""

# ── Try browser cookies to reduce rate-limit risk ───────────────────────────
for BROWSER in chrome firefox chromium; do
  if command -v "$BROWSER" &>/dev/null || [ -d "/Applications/$(echo "$BROWSER" | sed 's/./\u&/').app" ] || [ -d "/Applications/Google Chrome.app" ]; then
    if yt-dlp --cookies-from-browser "$BROWSER" --skip-download \
         --quiet "https://www.youtube.com/watch?v=dQw4w9WgXcQ" 2>/dev/null; then
      echo "[info] Using $BROWSER cookies"
      COOKIE_ARGS="--cookies-from-browser $BROWSER"
      break
    fi
  fi
done

[ -z "$COOKIE_ARGS" ] && echo "[info] No browser cookies found — proceeding without (may hit rate limits)"

# ── Helper: rename yt-id-named files to exercise_bad_N.mp4 ──────────────────
rename_files() {
  local DIR="$1"
  local PREFIX="$2"
  local N=1
  for F in "$DIR"/*.mp4 "$DIR"/*.webm "$DIR"/*.mkv 2>/dev/null; do
    [ -f "$F" ] || continue
    BASENAME=$(basename "$F")
    # Skip files already named correctly
    [[ "$BASENAME" == ${PREFIX}_[0-9]*.* ]] && continue
    EXT="${F##*.}"
    mv "$F" "$DIR/${PREFIX}_${N}.${EXT}"
    (( N++ ))
  done
}

# ── Helper: download one batch ───────────────────────────────────────────────
download_batch() {
  local QUERY="$1"
  local OUTDIR="$2"
  local PREFIX="$3"
  local NEEDED=25

  local EXISTING
  EXISTING=$(ls "$OUTDIR"/*.mp4 "$OUTDIR"/*.webm "$OUTDIR"/*.mkv 2>/dev/null | wc -l | tr -d ' ')
  if [ "$EXISTING" -ge "$NEEDED" ]; then
    echo "[skip] $PREFIX already has $EXISTING/$NEEDED — nothing to do"
    return 0
  fi

  local REMAINING=$(( NEEDED - EXISTING ))
  echo ""
  echo "▶ $PREFIX: fetching $REMAINING more (have $EXISTING/$NEEDED)..."

  # Use video ID as filename — always unique, safe to re-run
  # shellcheck disable=SC2086
  yt-dlp "ytsearch$(( REMAINING + 10 )):${QUERY}" \
    $COOKIE_ARGS \
    --ignore-errors \
    --no-playlist \
    --socket-timeout 90 \
    --retries 5 \
    --retry-sleep linear=2:10 \
    --sleep-interval 3 \
    --max-sleep-interval 8 \
    -f "mp4/best[ext=mp4]/best[height<=720]/best" \
    --output "$OUTDIR/%(id)s.%(ext)s" \
    --max-downloads "$REMAINING" \
    --progress 2>&1 | grep -E "\[download\] Destination|ERROR|already been downloaded" || true

  rename_files "$OUTDIR" "$PREFIX"

  local AFTER
  AFTER=$(ls "$OUTDIR"/*.mp4 "$OUTDIR"/*.webm "$OUTDIR"/*.mkv 2>/dev/null | wc -l | tr -d ' ')
  echo "  → $PREFIX: $AFTER/$NEEDED complete"

  if [ "$AFTER" -lt "$NEEDED" ]; then
    echo "  ⚠  Only got $AFTER/$NEEDED. YouTube may be rate-limiting."
    echo "     Wait ~10 min then re-run this script — it will resume."
  fi
}

# ── Run downloads ────────────────────────────────────────────────────────────
mkdir -p "$BASE/deadlift_bad" "$BASE/squat_bad" "$BASE/pull_up_bad"

download_batch \
  "deadlift bad form rounded back incorrect technique common mistakes" \
  "$BASE/deadlift_bad" "deadlift_bad"

download_batch \
  "squat bad form knee cave butt wink heels rise incorrect technique" \
  "$BASE/squat_bad" "squat_bad"

download_batch \
  "pull up bad form kipping wrong technique common mistakes" \
  "$BASE/pull_up_bad" "pull_up_bad"

# ── Final summary ────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo " Dataset summary"
echo "══════════════════════════════════════════════════"
printf "  %-14s  %-4s  %s\n" "Class" "Type" "Files"
printf "  %-14s  %-4s  %s\n" "─────────────" "────" "─────"
for INFO in "deadlift:deadlift:good" "squat:squat:good" "pull Up:pull_up:good" \
            "deadlift_bad:deadlift:bad" "squat_bad:squat:bad" "pull_up_bad:pull_up:bad"; do
  DIR="${INFO%%:*}"
  REST="${INFO#*:}"
  EX="${REST%%:*}"
  TYPE="${REST##*:}"
  COUNT=$(ls "$BASE/$DIR/"*.mp4 "$BASE/$DIR/"*.MOV "$BASE/$DIR/"*.webm 2>/dev/null | wc -l | tr -d ' ')
  printf "  %-14s  %-4s  %s/25\n" "$EX" "$TYPE" "$COUNT"
done
echo "══════════════════════════════════════════════════"
