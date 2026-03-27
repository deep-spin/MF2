#!/bin/bash

# VENV_PATH="/mnt/data-poseidon/manos/venvs/venv-f3"
# VENV_PATH="/mnt/scratch-artemis/manos/venvs_v2/venv-litellm-gemini"
VENV_PATH="/mnt/scratch-artemis/manos/venvs_v2/venv-litellm"
source "$VENV_PATH/bin/activate"

export PYTHONPATH="/mnt/data-poseidon/manos/projects/f3-project/official_repo/MF2:$PYTHONPATH"

MODEL_NAME="gemini-2.5-pro" #model name from registry (see models.closed_source.registry.py)
MODEL_PROVIDER="openrouter/google/gemini-2.5-pro" #provider used.
# MODEL_PROVIDER="openrouter/google-vertex/global/gemini-2.5-pro" #provider used.
API_KEY=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 # open-router key
API_BASE=https://openrouter.ai/api/v1 # open-router base url

# Output layout
BASE_OUTPUT_DIR="/mnt/data-poseidon/manos/projects/f3-project/official_repo/results_open_source_cropped_movies_draft"

# Modality and prompt templates
MODALITY="video_only"  # video_only | transcripts_only | video_and_transcripts | statement_only
USER_PROMPT_TEMPLATE_NAME="explanation_free"
SYSTEM_PROMPT_TEMPLATE_NAME="default"
MODE="independent"

# Cropped assets roots (per-claim files live under {root}/{movie_id}/{movie_id}_claim_{claim_id}.ext)
VIDEOS_ROOT="/mnt/data-poseidon/manos/projects/f3-project/official_repo/data/cropped_movies_final/single_only/cropped_mp4"
TRANSCRIPTS_ROOT="/mnt/data-poseidon/manos/projects/f3-project/official_repo/data/cropped_movies_final/single_only/transcripts_full"
SYNOPSIS_ROOT="/mnt/data-poseidon/manos/projects/f3-project/official_repo/MF2/data/synopsis"  

# Claims CSV (must include columns: movie_id, claim_id, true_claim, false_claim)
DATA_PATH="/mnt/data-poseidon/manos/projects/f3-project/official_repo/data/cropped_movies_final/single_only/successfully_processed_claims.csv"

# Frame sampling per clip (only used if modality includes video)
NUM_FRAMES=120

# Pick specific movies to run
# MOVIE_IDS=(5 9 13 15 17 29 30 34 37 38 39 47 54)
MOVIE_IDS=(5)

echo "Running $MODEL_NAME on movies: ${MOVIE_IDS[*]} (modality=$MODALITY)"

for MOVIE_ID in "${MOVIE_IDS[@]}"; do
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODALITY}/${USER_PROMPT_TEMPLATE_NAME}/"

  python /mnt/data-poseidon/manos/projects/f3-project/official_repo/MF2/src_cropped_movies/run_closed_source_cropped_movies.py \
    --model_name "$MODEL_NAME" \
    --mode "$MODE" \
    --modality "$MODALITY" \
    --movie_id "$MOVIE_ID" \
    --data_path "$DATA_PATH" \
    --videos_root "$VIDEOS_ROOT" \
    --transcripts_root "$TRANSCRIPTS_ROOT" \
    --synopsis_root "$SYNOPSIS_ROOT" \
    --user_prompt_template_name "$USER_PROMPT_TEMPLATE_NAME" \
    --system_prompt_template_name "$SYSTEM_PROMPT_TEMPLATE_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --API_KEY "$API_KEY" \
    --api_base "$API_BASE" \
    --model_provider "$MODEL_PROVIDER" \
    --num_frames "$NUM_FRAMES"
    # --shuffle_frames "$SHUFFLE_FRAMES" # not implemented yet for closed-source models

done