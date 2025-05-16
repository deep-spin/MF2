import argparse
import cv2
import time
import base64
import os
import pandas as pd
import json

from openai import OpenAI
import time
from tqdm import tqdm

from loguru import logger

from templates.prompt_utils import build_prompts
from utils import validate_paths_for_mode, load_transcripts, load_synopsis


def parse_args():
    parser = argparse.ArgumentParser(description="Video Claim Evaluation")
    parser.add_argument("--model", required=True, help="Model name. Check models.registry.py for available models.")
    parser.add_argument("--video_path", default=None, help="Path to the video file.")
    parser.add_argument("--transcripts_path", default=None, help="Path to the transcripts file.")
    parser.add_argument("--synopsis_path", default=None, help="Path to the synopsis file.")
    parser.add_argument("--movie_id", type=int, required=True, help="Movie ID to filter.")
    parser.add_argument("--data_path", required=True, help="Path to the dataset CSV file.")
    parser.add_argument("--movie_id_column", default="movie_id", help="Column name for movie ID.")
    parser.add_argument("--true_claim_column", default="true_claim", help="Column name for true claims.")
    parser.add_argument("--false_claim_column", default="false_claim", help="Column name for false claims.")
    parser.add_argument("--user_prompt_template_name", default="default", help="Template to be used for the prompt.")
    parser.add_argument("--system_prompt_template_name", default="default", help="Template to be used for system prompt.")
    parser.add_argument("--modality", required=True, choices=["video_only", "transcripts_only", "video_and_transcripts", "video_and_synopsis", "video_transcripts_and_synopsis", "statement_only" , "synopsis_only"], help="Inference mode to be used.")
    parser.add_argument("--output_dir", default="./results", help="Output directory.")
    parser.add_argument("--API_KEY", default="None", help="API_KEY.")
    return parser.parse_args()


def process_video(video_path, num_frames=256):
    base64Frames = []
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    interval = max(total_frames // num_frames, 1)
    for i in range(0, total_frames, interval):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        if len(base64Frames) >= num_frames:
            break

    video.release()
    print(f"Extracted {len(base64Frames)} frames")
    return base64Frames


def model_inference(args, input_prompts, video_path, system_prompt):
    responses = []
    base64Frames = process_video(args.video_path, num_frames=256)
    for input_prompt in tqdm(input_prompts):
        if video_path:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": input_prompt},
                        *[
                            { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{img}", "detail": "auto"} }                            
                            for img in base64Frames
                        ]
                    ]}
                ],
                temperature=0
            )
        else:
                response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": input_prompt},
                    ]}
                ],
                temperature=0
            )

        responses.append(response.choices[0].message.content)
    return responses


if __name__ == '__main__':
    args = parse_args()
    global client
    client = OpenAI(api_key=args.API_KEY)
    claims = pd.read_csv(args.data_path)

    output_dir = os.path.join(args.output_dir, args.model.replace("/", "__"))
    os.makedirs(output_dir, exist_ok=True) 
    output_file = os.path.join(output_dir, f'{args.movie_id}-results.json')
    if os.path.exists(output_file):
        logger.info(f"Results file {output_file} already exists, skipping...")
        exit()

    logger.info(f"Loading claims for movie {args.movie_id}.")
    true_claims = claims[claims[args.movie_id_column] == args.movie_id][args.true_claim_column].dropna().tolist()
    false_claims = claims[claims[args.movie_id_column] == args.movie_id][args.false_claim_column].dropna().tolist()


    # Validate paths based on inference mode
    video_path, transcripts_path, synopsis_path = validate_paths_for_mode(args.modality, args.video_path, args.transcripts_path, args.synopsis_path)

    # load transcripts, synopsis if needed
    transcripts = load_transcripts(transcripts_path)
    synopsis = load_synopsis(synopsis_path)
    

    # Build (text) prompts for true and false claims
    input_prompts_true,system_prompt_true = build_prompts(true_claims, args.user_prompt_template_name, args.system_prompt_template_name,video_path, transcripts, synopsis, movie_id=args.movie_id)
    input_prompts_false,system_prompt_false = build_prompts(false_claims, args.user_prompt_template_name, args.system_prompt_template_name,video_path, transcripts, synopsis, movie_id=args.movie_id) 
    # system prompt for true and false claims should be the same!

    # Show example prompt
    logger.info(f"Example user prompt used: {input_prompts_true[0]}")
    if system_prompt_true:
        logger.info(f"System prompt used: {system_prompt_true}")
    else:
        logger.info("Model's default system prompt used.")


    # Run inference
    logger.info(f"About to run inference for true and false claims...")
    responses = model_inference(args, input_prompts_true+input_prompts_false, video_path, system_prompt=system_prompt_true)

    responses_true = responses[:len(input_prompts_true)]
    responses_false = responses[len(input_prompts_true):]
    
    result_data = {
        idx: {
            args.true_claim_column: responses_true[idx] if idx < len(responses_true) else "",
            args.false_claim_column: responses_false[idx] if idx < len(responses_false) else ""
        }
        for idx in range(max(len(responses_true), len(responses_false)))
    }

    
    logger.info(f"Saving results to {output_file}.")
    with open(output_file, 'w') as f:
        json.dump(result_data, f, indent=4)

    print(f"Results saved to {output_file}.")