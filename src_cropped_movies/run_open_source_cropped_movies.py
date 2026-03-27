#!/usr/bin/env python3
import argparse
import os
import json
import base64
import cv2
from typing import List, Tuple, Optional
import pandas as pd
from tqdm import tqdm
from loguru import logger
import re

from models.registry import MODELS_MAPPING
from templates.prompt_utils import build_prompts 

from utils import validate_paths_for_mode, load_transcripts, load_synopsis


def parse_args():
    p = argparse.ArgumentParser(description="Run LiteLLM on cropped per-claim segments")
    p.add_argument("--model", required=True, help="Model name (see your LiteLLM registry)")
    p.add_argument("--data_path", required=True, help="CSV with columns: movie_id,claim_id,true_claim,false_claim")
    p.add_argument("--movie_id", type=int, required=True, help="Movie ID to process")
    p.add_argument("--videos_root", default=None, help="Root dir for cropped videos")
    p.add_argument("--transcripts_root", default=None, help="Root dir for cropped transcripts")
    p.add_argument("--synopsis_root", default=None, help="Root dir for movie synopsis")
    p.add_argument("--movie_id_column", default="movie_id", help="Column name for movie ID.")
    p.add_argument("--claim_id_column", default="claim_id", help="Column name for claim ID.")
    p.add_argument("--true_claim_column", default="true_claim", help="Column name for true claims.")
    p.add_argument("--false_claim_column", default="false_claim", help="Column name for false claims.")
    p.add_argument("--user_prompt_template_name", default="default", help="Template to be used for the prompt.")
    p.add_argument("--system_prompt_template_name", default="default", help="Template to be used for system prompt.")
    p.add_argument("--output_dir", default="./results", help="Output directory for JSON results")
    p.add_argument("--mode",
        choices=["independent", "multiple_choice"],
        default="independent",
        help="Mode to be used. Whether to run inference for true and false claims independently or together(multiple-choice).",
    )    
    p.add_argument(
        "--modality",
        choices=["video_only", "transcripts_only", "video_and_transcripts", "statement_only"],
        default="video_and_transcripts",
        help="What inputs to include in the prompt per claim",
    )    
    p.add_argument("--shuffle_frames", action="store_true",default=False, help="Shuffle frames of the video before inference.")
    p.add_argument("--port", type=int, default=None, help="To be used if model is launched with vLLM server")
    p.add_argument("--host", type=str, default=None, help="To be used if model is launched with vLLM server")


    return p.parse_args()


def model_inference(args, input_prompts, video_path=None, system_prompt=None, shuffle_frames=False):
    inference_fn = MODELS_MAPPING[args.model]
    responses = inference_fn(args, input_prompts, video_path, system_prompt, shuffle_frames=shuffle_frames)
    #responses is a list of responses. We return the element at index 0 since we pass a single prompt.
    return responses[0]


def main():
    args = parse_args()

    # Prepare output path
    out_dir = os.path.join(args.output_dir, args.model.replace("/", "__"))
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"{args.movie_id}-results.json")
    if os.path.exists(out_json):
        logger.info(f"Results {out_json} already exists, skipping...")
        return

    # Load claims
    logger.info(f"Loading claims for movie {args.movie_id}.")
    df = pd.read_csv(args.data_path)
    df = df[df[args.movie_id_column] == args.movie_id]
    if df.empty:
        logger.error(f"No rows for movie_id={args.movie_id} in {args.data_path}")
        return

    
    if args.mode == "independent":
        logger.info(f"Building prompts for true and false claims independently...")
    elif args.mode == "multiple_choice":
        logger.error(f"Multiple choice mode is not implemented yet.")
        raise NotImplementedError("Multiple choice mode is not implemented yet.")
    else:
        logger.error(f"Invalid mode: {args.mode}")
        raise ValueError(f"Invalid mode: {args.mode}")

    # 1) Build data for inference
    true_prompts, false_prompts = [], []
    video_paths, claim_ids = [], []
    for idx, row in df[[args.movie_id_column,args.claim_id_column,args.true_claim_column,args.false_claim_column]].dropna(subset=[args.claim_id_column]).iterrows():
        movie_id = int(row[args.movie_id_column])
        claim_id = int(row[args.claim_id_column])
        true_claim = str(row[args.true_claim_column]) if pd.notna(row[args.true_claim_column]) else ""
        false_claim = str(row[args.false_claim_column]) if pd.notna(row[args.false_claim_column]) else ""
        
        video_path_candidate = os.path.join(args.videos_root, str(movie_id), f"{movie_id}_claim_{claim_id}.mp4") if args.videos_root else None
        video_path = video_path_candidate if (video_path_candidate and os.path.exists(video_path_candidate)) else None
        transcripts_path_candidate = os.path.join(args.transcripts_root, str(movie_id), f"{movie_id}_claim_{claim_id}.srt") if args.transcripts_root else None
        transcripts_path = transcripts_path_candidate if (transcripts_path_candidate and os.path.exists(transcripts_path_candidate)) else None
        synopsis_path_candidate = os.path.join(args.synopsis_root, f"{movie_id}.synopsis") if args.synopsis_root else None
        synopsis_path = synopsis_path_candidate if (synopsis_path_candidate and os.path.exists(synopsis_path_candidate)) else None

        # Validate against the selected modality (non-used paths will be nulled by the validator)
        video_path, transcripts_path, synopsis_path = validate_paths_for_mode(
            args.modality,
            video_path=video_path,
            transcripts_path=transcripts_path,
            synopsis_path=synopsis_path
        )

        # Load transcript, synopsis if needed
        transcript = load_transcripts(transcripts_path)
        synopsis = load_synopsis(synopsis_path)

        # Build prompts for true and false claims
        true_prompt, system_prompt_true = build_prompts([true_claim], args.user_prompt_template_name, args.system_prompt_template_name, video_path, transcript, synopsis, movie_id=movie_id)
        false_prompt, system_prompt_false = build_prompts([false_claim], args.user_prompt_template_name, args.system_prompt_template_name, video_path, transcript, synopsis, movie_id=movie_id)
        if isinstance(true_prompt, list):
            true_prompt = true_prompt[0]
        if isinstance(false_prompt, list):
            false_prompt = false_prompt[0]

        # Show example prompt
        if idx == 0:
            if system_prompt_true:
                logger.info(f"System prompt used: {system_prompt_true}")
            logger.info(f"Example user prompt used: {true_prompt[:400]} .... (truncated) {true_prompt[-100:]}")

        
        true_prompts.append(true_prompt)
        false_prompts.append(false_prompt)
        video_paths.append(video_path)
        claim_ids.append(claim_id)
    
    # Run inference
    logger.info(f"About to run inference for true and false claims...")

    responses_true = {}
    responses_false = {}
    pairs = list(zip(true_prompts, false_prompts, video_paths, claim_ids))
    for true_prompt, false_prompt, video_path, claim_id in tqdm(pairs, desc="Running inference for claims", total=len(pairs)):
        true_response = model_inference(args, [true_prompt], video_path, system_prompt=system_prompt_true, shuffle_frames=args.shuffle_frames)
        false_response = model_inference(args, [false_prompt], video_path, system_prompt=system_prompt_false, shuffle_frames=args.shuffle_frames)
        responses_true[claim_id] = true_response
        responses_false[claim_id] = false_response
    
    #Saving results 
    result_data = {
        str(claim_id): {
            args.true_claim_column: responses_true.get(claim_id, "") or "",
            args.false_claim_column: responses_false.get(claim_id, "") or ""
        }
        for claim_id in claim_ids
    }

    logger.info(f"Saving results to {out_json}.")
    with open(out_json, 'w') as f:
        json.dump(result_data, f, indent=4)


if __name__ == "__main__":
    main()