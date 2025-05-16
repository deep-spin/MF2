from google import genai
import argparse
import pandas as pd
import json
import os
import time
from tqdm import tqdm
import cv2
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
    
TARGET_MAX_FRAMES = 1

def downsample_video_if_needed(original_video_path, temp_video_dir):
    logger.info(f"Checking video for exact-frame downsampling: {original_video_path}")
    
    try:
        # Open the original video using OpenCV
        cap = cv2.VideoCapture(original_video_path)
        
        if not cap.isOpened():
            logger.warning(f"Could not open video {original_video_path}. Skipping.")
            return original_video_path, False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        logger.info(f"Video properties: FPS={fps}, Total frames={total_frames}, Duration={duration:.2f}s")
        
        if total_frames <= TARGET_MAX_FRAMES:
            logger.info(f"Video has {total_frames} frames, which is less than or equal to {TARGET_MAX_FRAMES}. No downsampling needed.")
            return original_video_path, False
        
        # Calculate frame intervals
        frame_interval = total_frames / TARGET_MAX_FRAMES
        frames_to_extract = []

        # Extract frames at even intervals
        for i in range(TARGET_MAX_FRAMES):
            frame_number = int(i * frame_interval)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                frames_to_extract.append(frame)
            else:
                logger.warning(f"Failed to extract frame {frame_number} from video.")
        
        cap.release()

        # Create the output video path
        os.makedirs(temp_video_dir, exist_ok=True)
        base_name = os.path.basename(original_video_path)
        name, ext = os.path.splitext(base_name)
        temp_video_path = os.path.join(temp_video_dir, f"{name}_exact_{TARGET_MAX_FRAMES}frames.mp4")

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 format
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frames_to_extract[0].shape[1], frames_to_extract[0].shape[0]))

        # Write frames to output video
        for frame in frames_to_extract:
            out.write(frame)
        
        out.release()

        logger.info(f"Downsampled video saved: {temp_video_path}")
        return temp_video_path, True

    except Exception as e:
        logger.error(f"Error during exact-frame downsampling for {original_video_path}: {e}", exc_info=True)
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        return original_video_path, False

def model_inference(args, input_prompts, video_path, system_prompt):
    logger.info(f"Running {args.model} on {len(input_prompts)} claims for movie {args.movie_id}.")
    temp_video_file, _ = downsample_video_if_needed(video_path, ".")
    video_file = client.files.upload(file=temp_video_file)
    while video_file.state.name == 'PROCESSING':
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)


    responses = []
    for input_prompt in tqdm(input_prompts):
        if video_path:
            response = client.models.generate_content(
                model=args.model,
                contents=[
                    {"role": "system", "parts": [system_prompt]},
                    {"role": "user", "parts": [video_file, input_prompt]},
                ],
            )
        else:
            response = client.models.generate_content(
                model=args.model,
                contents=[
                    {"role": "system", "parts": [system_prompt]},
                    {"role": "user", "parts": [input_prompt]},
                ],
            )
        responses.append(response.text)
    #client.caches.delete(name=cache.name) #uncomment to delete cache
    os.remove(temp_video_file)
    return responses

if __name__ == '__main__':
    args = parse_args()
    global client
    client = genai.Client(api_key=args.API_KEY)
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

    logger.info(f"Example user prompt used: {input_prompts_true[0]}")
    if system_prompt_true:
        logger.info(f"System prompt used: {system_prompt_true}")
    else:
        logger.info("Model's default system prompt used.")

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