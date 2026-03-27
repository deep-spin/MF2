#!/bin/bash

# VENV_PATH="/mnt/data-poseidon/manos/venvs/venv-f3"
# VENV_PATH="/mnt/scratch-hades/manos/venvs/venv-f3-vllm"
VENV_PATH="/mnt/scratch-hades/manos/venvs/venv-f3-transformers"
# VENV_PATH="/mnt/scratch-dionysus/manos/venvs/venv-f3-proj-vllm"
source "$VENV_PATH/bin/activate"


# export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/mnt/data-poseidon/manos/projects/f3-project/official_repo/MF2:$PYTHONPATH"

# MODEL= #OpenGVLab/InternVL3_5-38B-Instruct Qwen/Qwen3-VL-30B-A3B-Instruct # OpenGVLab/InternVL3-78B # google/gemma-3-27b-it # "Qwen/Qwen2.5-VL-72B-Instruct"

MODELS=(
  # "Qwen/Qwen3-VL-30B-A3B-Instruct"
  # "Qwen/Qwen2.5-VL-72B-Instruct"
  "google/gemma-3-27b-it"
  "OpenGVLab/InternVL3-78B"
  "OpenGVLab/InternVL3_5-38B-Instruct"
  
)

# Output layout
BASE_OUTPUT_DIR="/mnt/data-poseidon/manos/projects/f3-project/official_repo/results_open_source_cropped_movies_cropped_trans"

# Modality and prompt templates
MODALITY="video_and_transcripts"  # video_only | transcripts_only | video_and_transcripts | statement_only
USER_PROMPT_TEMPLATE_NAME="direct_free"
SYSTEM_PROMPT_TEMPLATE_NAME="default"
mode="independent"

# Cropped videos, transcripts, and synopsis root directories
VIDEOS_ROOT="/mnt/data-poseidon/manos/projects/f3-project/official_repo/data/cropped_movies_final/single_only/cropped_mp4"
TRANSCRIPTS_ROOT="/mnt/data-poseidon/manos/projects/f3-project/official_repo/data/cropped_movies_final/single_only/cropped_transcripts"
SYNOPSIS_ROOT="None"

# Claims CSV file (must include columns: movie_id, claim_id, true_claim, false_claim)
DATA_PATH="/mnt/data-poseidon/manos/projects/f3-project/official_repo/data/cropped_movies_final/single_only/successfully_processed_claims.csv"

MOVIE_IDS=(5 9 13 15 16 17 29 30 34 37 38 39 47 54)
# MOVIE_IDS=(5)


for MODEL in "${MODELS[@]}"; do
  echo "Running $MODEL on movies: ${MOVIE_IDS[*]} (modality=$MODALITY)"

  for MOVIE_ID in "${MOVIE_IDS[@]}"; do
    OUTPUT_DIR=${BASE_OUTPUT_DIR}/${MODALITY}/${USER_PROMPT_TEMPLATE_NAME}

    python /mnt/data-poseidon/manos/projects/f3-project/official_repo/MF2/src_cropped_movies/run_open_source_cropped_movies.py \
      --model "$MODEL" \
      --mode "$mode" \
      --modality "$MODALITY" \
      --movie_id "$MOVIE_ID" \
      --data_path "$DATA_PATH" \
      --videos_root "$VIDEOS_ROOT" \
      --transcripts_root "$TRANSCRIPTS_ROOT" \
      --synopsis_root "$SYNOPSIS_ROOT" \
      --user_prompt_template_name "$USER_PROMPT_TEMPLATE_NAME" \
      --system_prompt_template_name "$SYSTEM_PROMPT_TEMPLATE_NAME" \
      --output_dir "$OUTPUT_DIR"
      # --host "hades" \
      # --port "8003"
  done
done