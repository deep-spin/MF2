#!/bin/bash

MODEL=$1
BASE_OUTPUT_DIR=$2
MODALITY=$3
USER_PROMPT_TEMPLATE_NAME=$4

MOVIE_IDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 42 43 44 45 46 47 48 49 50 51 52 53 54)


SYSTEM_PROMPT_TEMPLATE_NAME="default"
# Base paths for videos, transcripts and synopsis
VIDEO_BASE_PATH="data/MF2/test"
TRANSCRIPTS_BASE_PATH="data/transcripts"
SYNOPSIS_BASE_PATH="data/synopsis"

# path with claims files
DATA_PATH="data/MF2/test/metadata.csv"

echo "Running inference in movies: ${MOVIE_IDS[@]}"
echo "Running inference for model: $MODEL in movies with modality: $MODALITY"
for MOVIE_ID in "${MOVIE_IDS[@]}"; do

    # Extend output if you want
    OUTPUT_DIR=${BASE_OUTPUT_DIR}/${MODALITY}

    # Set video and transcript paths for this movie
    VIDEO_PATH="${VIDEO_BASE_PATH}/${MOVIE_ID}.mp4"
    TRANSCRIPTS_PATH="${TRANSCRIPTS_BASE_PATH}/${MOVIE_ID}.srt"
    SYNOPSIS_PATH="${SYNOPSIS_BASE_PATH}/${MOVIE_ID}.synopsis"

    # Run the inference script
    python ./run_open_vlm.py \
        --model "$MODEL" \
        --movie_id "$MOVIE_ID" \
        --data_path "$DATA_PATH" \
        --user_prompt_template_name "$USER_PROMPT_TEMPLATE_NAME" \
        --system_prompt_template_name "$SYSTEM_PROMPT_TEMPLATE_NAME" \
        --modality "$MODALITY" \
        --output_dir "$OUTPUT_DIR" \
        --video_path "$VIDEO_PATH" \
        --transcripts_path "$TRANSCRIPTS_PATH" \
        --synopsis_path "$SYNOPSIS_PATH"
done