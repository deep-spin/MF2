#!/usr/bin/env python3
"""
Cropping pipeline for videos and transcripts.

- Loads MF2_test_all_info.csv and filters rows with Correct Timestamp == True.
- Supports Single and Multi granularity:
  - Single: one range; apply single_buffer_seconds on both sides.
  - Multi: comma-separated ranges; apply multi_buffer_seconds per range, merge overlaps, crop/concat.
- Produces:
  - Videos: {saving_dir}/cropped_mp4/{movie_id}/{video_basename}_claim_{claim_id}.mp4
  - Transcripts: {saving_dir}/cropped_transcripts/{movie_id}/{video_basename}_claim_{claim_id}.srt
"""

import os
import argparse
import tempfile
import pandas as pd
from loguru import logger

from utils_cropped import general_crop_function, validate_timestamps

def parse_args():
    p = argparse.ArgumentParser(description='Crop videos and transcripts for Single/Multi claims.')
    p.add_argument('--claims_csv', default='/mnt/data-poseidon/manos/projects/f3-project/official_repo/data/MF2_test_all_info.csv', help='Path to MF2_test_all_info.csv')
    p.add_argument('--movies_dir', default='/mnt/scratch-artemis/manos/data/movies-facts-and-fibs/MF2/test/', help='Input videos dir')
    p.add_argument('--transcripts_dir', default='/mnt/data-poseidon/manos/projects/f3-project/official_repo/MF2/data/transcripts', help='Input transcripts dir (movie_id.srt)')
    p.add_argument('--process_granularity', choices=['single', 'multi', 'single_multi'], default='single',
                   help='Process Single only, Multi only, or both')

    p.add_argument('--apply_buffer', type=float, default=None, help='Whether to apply a buffer window around the given timeframe. \
        If x=80, the window will be extended by x/2 seconds before and x/2 seconds after the given timeframe. If None, no buffer will be applied.')
    
    # Single timestamp handling
    p.add_argument('--handle_single_timestamps', action='store_true',
                   help='If set, the script handles single timestamps (HH:MM:SS), by treating them as midpoints for a window of interest and creating a window around them.\
                    The window will be extended by apply_buffer/2 seconds before and apply_buffer/2 seconds after the given timestamp.\
                    If not set, the script will skip single timestamps (HH:MM:SS) and only process single-scene ranges (HH:MM:SS-HH:MM:SS) or multi-scene ranges [HH:MM:SS-HH:MM:SS,HH:MM:SS-HH:MM:SS,...].')
    
    p.add_argument('--saving_dir', default=None,required=True, help='Directory to save the cropped data')
    return p.parse_args()



# ---------- Main ----------

def should_process(granularity_value, mode):
    g = str(granularity_value).strip().lower()
    if mode == 'single':
        return g == 'single'
    if mode == 'multi':
        return g == 'multi'
    if mode == 'single_multi':
        return g in ('single', 'multi')
    else:
        logger.error(f"Invalid granularity mode: {mode}. Filtering not implemented for this mode.")
        raise ValueError(f"Invalid granularity mode: {mode}. Filtering not implemented for this mode.")

def main():
    args = parse_args()

    movies_dir = os.path.abspath(args.movies_dir)
    transcripts_dir = os.path.abspath(args.transcripts_dir)
    saving_dir = os.path.abspath(args.saving_dir)
    out_video_dir = os.path.join(saving_dir, 'cropped_mp4')
    out_srt_dir = os.path.join(saving_dir, 'cropped_transcripts')

    logger.info(f"CSV with movies and claims: {args.claims_csv}")
    
    if args.apply_buffer is not None:
        logger.info(f"Applied buffer: {args.apply_buffer}")
    else:
        logger.info("Applied buffer: None")
    if args.handle_single_timestamps:
        logger.info("Handle single timestamps: True")
    else:
        logger.info("Handle single timestamps: False")
    logger.info(f"Granularity to process: {args.process_granularity}")
    logger.info(f"Saving dir for cropped videos: {out_video_dir}")
    logger.info(f"Saving dir for cropped transcripts: {out_srt_dir}")

    df = pd.read_csv(args.claims_csv)
    # Filter out claims with no timestamps
    df = df[df['Timestamps'].notna() & (df['Timestamps'].astype(str).str.strip() != '')].copy()

    # Also filter by granularity
    if args.process_granularity == 'single':
        df = df[df['granularity'].astype(str).str.strip().str.lower() == 'single'].copy()
    elif args.process_granularity == 'multi':
        df = df[df['granularity'].astype(str).str.strip().str.lower() == 'multi'].copy()
    elif args.process_granularity == 'single_multi':
        df = df[df['granularity'].astype(str).str.strip().str.lower().isin(['single', 'multi'])].copy()


    if not args.handle_single_timestamps:
        # filter out claims with single timestamps (keep only those with '-')
        df = df[df['Timestamps'].astype(str).str.strip().str.strip('"').str.contains('-', na=False)].copy()

    # Save filtered dataframe
    os.makedirs(saving_dir, exist_ok=True)
    filtered_csv_path = os.path.join(saving_dir, 'filtered_claims_checked.csv')
    df.to_csv(filtered_csv_path, index=False)
    logger.info(f"Saved filtered dataframe to: {filtered_csv_path}")
    logger.info(f"Remaining claims after filtering: {len(df)}")


    #validate the timestamps format.
    validate_timestamps(df['Timestamps'])
    logger.info("Timestamps format validated.")

    # Start processing claims
    processed_claims = []
    for _, row in df.iterrows():
        movie_id = row['movie_id']
        video = row['video']
        claim_id = row['claim_id']
        g = str(row['granularity']).strip().lower()
        timestamps = row['Timestamps']
        ts = str(timestamps).strip().strip('"')
        video_path = os.path.join(movies_dir, f'{video}')
        base = str(video).replace('.mp4', '')
        video_out = os.path.join(out_video_dir, str(movie_id), f"{base}_claim_{claim_id}.mp4")
        transcript_in = os.path.join(transcripts_dir, f"{movie_id}.srt")
        transcript_out = os.path.join(out_srt_dir, str(movie_id), f"{base}_claim_{claim_id}.srt")
        

        

        # Process the video and transcript for this claim
        try:
            general_crop_function(video_path, movie_id, claim_id, ts, g, video_out, transcript_in, transcript_out, args.apply_buffer)
        except Exception as e:
            logger.error(f"[Video] {movie_id}:{claim_id} -> ERROR: {e}")
            continue

        processed_claims.append(row)
    

    # Save successfully processed claims
    if processed_claims:
        processed_df = pd.DataFrame(processed_claims)
        processed_csv_path = os.path.join(saving_dir, 'successfully_processed_claims.csv')
        processed_df.to_csv(processed_csv_path, index=False)
        logger.info(f"Saved {len(processed_df)} successfully processed claims to: {processed_csv_path}")
    else:
        logger.info("No claims were successfully processed")




if __name__ == '__main__':
    main()