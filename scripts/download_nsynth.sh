#!/usr/bin/env bash
# Download NSynth dataset splits from Google Magenta public storage.
# Extracts only the bass instrument subset needed for SC-VAE training.
#
# Usage:
#   bash scripts/download_nsynth.sh --out /scratch/ig2671/datasets/nsynth
#
# Output layout:
#   <out>/nsynth-train/   — training split (~289k clips, all instruments)
#   <out>/nsynth-valid/   — validation split (~12k clips)
#   <out>/nsynth-test/    — test split (~4k clips)
#
# NSynthBass filters to bass instruments at load time — no pre-filtering needed.
# Total disk: ~22 GB train + ~2.7 GB valid + ~2.6 GB test = ~27 GB.
#
# Environment:
#   NSYNTH_OUT   override output dir (same as --out)

set -euo pipefail

# ---------------------------------------------------------------------------
# Args
OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out|-o) OUT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done
OUT="${NSYNTH_OUT:-${OUT}}"
if [[ -z "$OUT" ]]; then
  echo "Usage: $0 --out <dir>" >&2
  exit 1
fi

mkdir -p "$OUT"
cd "$OUT"

# Magenta download mirror (GCS bucket is dead as of 2026).
BASE_URL="http://download.magenta.tensorflow.org/datasets/nsynth"

SPLITS=(train valid test)
declare -A SIZES=([train]="22.4 GB" [valid]="2.7 GB" [test]="2.6 GB")

# ---------------------------------------------------------------------------
download_split() {
  local split="$1"
  local tarball="nsynth-${split}.jsonwav.tar.gz"
  local url="${BASE_URL}/${tarball}"
  local dest_dir="nsynth-${split}"

  if [[ -f "${dest_dir}/.complete" ]]; then
    echo "[skip] ${dest_dir} already complete"
    return
  fi

  echo "[download] ${split} (~${SIZES[$split]}) → ${tarball}"
  if command -v wget &>/dev/null; then
    wget -c --show-progress -O "${tarball}" "${url}"
  elif command -v curl &>/dev/null; then
    curl -L -C - --progress-bar -o "${tarball}" "${url}"
  else
    echo "ERROR: neither wget nor curl found" >&2
    exit 1
  fi

  echo "[extract] ${tarball}"
  tar -xzf "${tarball}"
  rm -f "${tarball}"
  touch "${dest_dir}/.complete"
  echo "[done] ${split} → ${dest_dir}/"
}

for split in "${SPLITS[@]}"; do
  download_split "$split"
done

echo ""
echo "[nsynth] Download complete. Layout:"
du -sh "${OUT}"/nsynth-*/  2>/dev/null || true
echo ""
echo "Set env vars before training:"
echo "  export NSYNTH_ROOT=${OUT}"
echo "  export NSYNTH_CACHE_TRAIN=${OUT}/cache/train.h5"
echo "  export NSYNTH_CACHE_VALID=${OUT}/cache/valid.h5"
echo ""
echo "Then run mel cache:"
echo "  python scripts/prepare_nsynth_cache.py --src \$NSYNTH_ROOT --split train --out \$NSYNTH_CACHE_TRAIN"
echo "  python scripts/prepare_nsynth_cache.py --src \$NSYNTH_ROOT --split valid --out \$NSYNTH_CACHE_VALID"
