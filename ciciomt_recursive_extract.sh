#!/usr/bin/env bash
set -u

ROOT="${1:-$HOME/data/ciciomt2024}"
STATE_DIR="$HOME/.local/state/ciciomt-extract"
STATE_FILE="$STATE_DIR/extracted_state.tsv"
LOG_FILE="$HOME/data/ciciomt2024_extract.log"
LOCK_FILE="$STATE_DIR/extractor.lock"

mkdir -p "$ROOT" "$STATE_DIR"
touch "$STATE_FILE" "$LOG_FILE"

# Keep a single runner.
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "$(date -Is) another extractor is already running" >> "$LOG_FILE"
  exit 0
fi

log() { printf '%s %s\n' "$(date -Is)" "$*" >> "$LOG_FILE"; }

archive_sig() {
  local f="$1"
  # signature: path|size|mtime_epoch
  local sz mt
  sz=$(stat -c '%s' "$f" 2>/dev/null || echo 0)
  mt=$(stat -c '%Y' "$f" 2>/dev/null || echo 0)
  printf '%s|%s|%s' "$f" "$sz" "$mt"
}

already_done() {
  local sig="$1"
  grep -Fqx "$sig" "$STATE_FILE"
}

mark_done() {
  local sig="$1"
  printf '%s\n' "$sig" >> "$STATE_FILE"
}

extract_one() {
  local f="$1" d
  d="$(dirname "$f")"
  case "$f" in
    *.tar.gz|*.tgz|*.tar.xz|*.txz|*.tar.bz2|*.tar)
      tar -xf "$f" -C "$d"
      ;;
    *.zip)
      unzip -o -q "$f" -d "$d"
      ;;
    *.gz)
      [[ "$f" == *.tar.gz ]] && return 1
      gunzip -kf "$f"
      ;;
    *.bz2)
      [[ "$f" == *.tar.bz2 ]] && return 1
      bunzip2 -kf "$f"
      ;;
    *.xz)
      [[ "$f" == *.tar.xz ]] && return 1
      xz -dkf "$f"
      ;;
    *)
      return 1
      ;;
  esac
  return 0
}

log "extractor started for ROOT=$ROOT"

# Long-running loop: keep scanning for new/unseen archives.
while true; do
  # Build list each cycle so nested archives get picked up.
  mapfile -t archives < <(find "$ROOT" -type f \( -name '*.zip' -o -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' -o -name '*.tar.xz' -o -name '*.txz' -o -name '*.gz' -o -name '*.bz2' -o -name '*.xz' \) | sort)

  if [[ ${#archives[@]} -eq 0 ]]; then
    log "no archives found; sleeping 60s"
    sleep 60
    continue
  fi

  extracted_this_cycle=0
  skipped_this_cycle=0
  errors_this_cycle=0

  for f in "${archives[@]}"; do
    sig=$(archive_sig "$f")
    if already_done "$sig"; then
      skipped_this_cycle=$((skipped_this_cycle+1))
      continue
    fi

    log "extracting: $f"
    if extract_one "$f"; then
      mark_done "$sig"
      extracted_this_cycle=$((extracted_this_cycle+1))
      log "done: $f"
    else
      errors_this_cycle=$((errors_this_cycle+1))
      log "error: $f"
      # avoid tight-fail loops for unreadable archives
      mark_done "$sig"
    fi
  done

  log "cycle summary: found=${#archives[@]} extracted=$extracted_this_cycle skipped=$skipped_this_cycle errors=$errors_this_cycle"

  # If we extracted anything, immediately rescan to catch new nested files quickly.
  if [[ $extracted_this_cycle -gt 0 ]]; then
    sleep 2
  else
    sleep 30
  fi
done
