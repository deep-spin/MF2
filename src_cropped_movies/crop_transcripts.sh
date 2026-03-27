#!/bin/bash

set -euo pipefail

# Usage:
#   bash run_crop_transcripts_from_claims.sh
#   bash run_crop_transcripts_from_claims.sh <csv_path> <transcripts_full_dir> <cropped_transcripts_dir> <report_path>

PY_SCRIPT="/mnt/data-poseidon/manos/projects/f3-project/official_repo/MF2/src_cropped_movies/crop_transcripts_from_claims.py"

CSV_PATH="/mnt/data-poseidon/manos/projects/f3-project/official_repo/data/cropped_movies_final/single_only/successfully_processed_claims.csv"
TRANSCRIPTS_FULL_DIR="/mnt/data-poseidon/manos/projects/f3-project/official_repo/data/cropped_movies_final/single_only/transcripts_full"
CROPPED_TRANSCRIPTS_DIR="/mnt/data-poseidon/manos/projects/f3-project/official_repo/data/cropped_movies_final/single_only/cropped_transcripts"
REPORT_PATH="/mnt/data-poseidon/manos/projects/f3-project/official_repo/data/cropped_movies_final/single_only/cropped_transcripts/cropping_report.md"

echo "Running transcript cropping..."
echo "CSV: $CSV_PATH"
echo "Full transcripts: $TRANSCRIPTS_FULL_DIR"
echo "Output cropped transcripts: $CROPPED_TRANSCRIPTS_DIR"
echo "Report: $REPORT_PATH"

python3 "$PY_SCRIPT" \
  --csv_path "$CSV_PATH" \
  --transcripts_full_dir "$TRANSCRIPTS_FULL_DIR" \
  --cropped_transcripts_dir "$CROPPED_TRANSCRIPTS_DIR" \
  --report_path "$REPORT_PATH"

echo "Done."
