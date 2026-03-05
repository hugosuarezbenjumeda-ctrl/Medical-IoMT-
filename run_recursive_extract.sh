#!/usr/bin/env bash
set -euo pipefail
ROOT="$HOME/data/ciciomt2024"
LOG="$HOME/data/ciciomt2024_extract.log"

mkdir -p "$ROOT"
exec >>"$LOG" 2>&1

echo "===== $(date -Is) start recursive extraction ====="
declare -A done_map

extract_one() {
  local f="$1"
  local d
  d="$(dirname "$f")"
  case "$f" in
    *.tar.gz|*.tgz|*.tar.xz|*.txz|*.tar.bz2|*.tar)
      echo "[tar] $f"
      tar -xf "$f" -C "$d"
      ;;
    *.zip)
      echo "[zip] $f"
      unzip -o -q "$f" -d "$d"
      ;;
    *.gz)
      [[ "$f" == *.tar.gz ]] && return 1
      echo "[gz] $f"
      gunzip -kf "$f"
      ;;
    *.bz2)
      [[ "$f" == *.tar.bz2 ]] && return 1
      echo "[bz2] $f"
      bunzip2 -kf "$f"
      ;;
    *.xz)
      [[ "$f" == *.tar.xz ]] && return 1
      echo "[xz] $f"
      xz -dkf "$f"
      ;;
    *)
      return 1
      ;;
  esac
  return 0
}

for pass in $(seq 1 60); do
  mapfile -t archives < <(find "$ROOT" -type f \( -name '*.tar.gz' -o -name '*.tgz' -o -name '*.tar.xz' -o -name '*.txz' -o -name '*.tar.bz2' -o -name '*.tar' -o -name '*.zip' -o -name '*.gz' -o -name '*.bz2' -o -name '*.xz' \) | sort)
  echo "pass $pass: found ${#archives[@]} archive(s)"

  extracted=0
  new_seen=0
  for f in "${archives[@]}"; do
    if [[ -n "${done_map[$f]+x}" ]]; then
      continue
    fi
    done_map[$f]=1
    new_seen=$((new_seen+1))
    if extract_one "$f"; then
      extracted=$((extracted+1))
    fi
  done

  echo "pass $pass: newly seen=$new_seen extracted=$extracted"
  if [[ "$new_seen" -eq 0 ]]; then
    break
  fi
done

echo "remaining archives after run:"
find "$ROOT" -type f \( -name '*.tar.gz' -o -name '*.tgz' -o -name '*.tar.xz' -o -name '*.txz' -o -name '*.tar.bz2' -o -name '*.tar' -o -name '*.zip' -o -name '*.gz' -o -name '*.bz2' -o -name '*.xz' \) | sed -n '1,200p'

du -sh "$ROOT"
echo "===== $(date -Is) end recursive extraction ====="
