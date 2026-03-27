import pandas as pd
import re
import subprocess
import os
import tempfile
import shutil
from loguru import logger


def general_crop_function(video_path: str,movie_id: str, claim_id: str, ts: str, granularity: str, video_out_path: str, transcript_input_path: str, transcript_output_path: str, buffer_seconds=None):
    """
    General function to crop a video and save the cropped video for each claim. Also saves the transcript for each claim.
    """
    # First we try to crop the video for this claim.
    try:
        max_video_duration = get_video_duration(video_path)
        if granularity == 'single':
            # Process single-scene: single timestamp (HH:MM:SS) or single-scene range (HH:MM:SS-HH:MM:SS)
            #  single timestamp (HH:MM:SS) are processed only if args.handle_single_timestamps is set, otherwise they are skipped.
            range = get_ranges_in_seconds(ts, max_video_duration=max_video_duration, buffer_seconds=buffer_seconds)
            start_s, end_s = range[0]
            # crop and save the video for this claims
            crop_video_segment_save_video_single(video_path, start_s, end_s, video_out_path)     
        elif granularity == 'multi':
            # Process multi-scene: multi-scene range [HH:MM:SS-HH:MM:SS,HH:MM:SS-HH:MM:SS,...]
            #  multi-scene ranges are processed by splitting them into single-scene ranges and processing each one separately.
            #assume that the multi-scene range is in valid format HH:MM:SS-HH:MM:SS,HH:MM:SS-HH:MM:SS,...
            ranges = get_ranges_in_seconds(ts, max_video_duration=max_video_duration, buffer_seconds=buffer_seconds)
            tmp_dir = tempfile.mkdtemp(prefix=f"clips_{movie_id}_{claim_id}_")
            # crop and save the video for this claim
            crop_video_segment_save_video_multi(video_path, ranges, tmp_dir, video_out_path)

        # Then we save the transcript for this claim.
        os.makedirs(os.path.dirname(transcript_output_path), exist_ok=True)
        shutil.copy(transcript_input_path, transcript_output_path)
    except Exception as e:
        logger.error(f"[Video] {movie_id}:{claim_id} -> ERROR: {e}")
        raise Exception(f"[Video] {movie_id}:{claim_id} -> ERROR: {e}") from e
    
    



    

def crop_video_segment_save_video_single(video_path: str, start_s: float, end_s: float, out_path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        duration = max(0.0, end_s - start_s)
        cmd = ['ffmpeg', '-ss', str(start_s), '-i', video_path, '-t', str(duration), '-c', 'copy', out_path, '-y']
        return run_ffmpeg(cmd)
    except Exception as e:
        logger.error(f"Error at crop_segment_save_video: {e}")
        raise Exception(f"Error at crop_segment_save_video: {e}") from e

def crop_video_segment_save_video_multi(video_path: str, ranges: list[tuple[float, float]],tmp_dir: str, out_path: str):
    """
    Crops a video into multiple segments based on the given ranges and saves them in a temporary directory.
    Then concatenates the segments into a single video and saves it to the given output path.
    """
    segs = []
    try:
        for i, (s, e) in enumerate(ranges):
            seg_path = os.path.join(tmp_dir, f"seg_{i:02d}.mp4")
            crop_video_segment_save_video_single(video_path, s, e, seg_path)
            segs.append(seg_path)
        if segs:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            concat_segments(segs, out_path)
    except Exception as e:
        logger.error(f"Failed crop and concatenate video segments for multi-scene ranges in tmp_dir: {tmp_dir} -> ERROR: {e}")
        raise Exception(f"Failed crop and concatenate video segments for multi-scene ranges in tmp_dir: {tmp_dir} -> ERROR: {e}") from e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def get_ranges_in_seconds(ts: str, max_video_duration: float, buffer_seconds: float = None) -> list[tuple[float, float]]:
    """
    Gets a timestamp: HH:MM:SS(single) or HH:MM:SS-HH:MM:SS(range) or HH:MM:SS-HH:MM:SS,HH:MM:SS-HH:MM:SS,...(multi)
    and returns a list of non-overlapping ranges (start_s, end_s).
    If buffer_seconds is None, no buffer will be applied and the window will be the original range.
    If buffer_seconds is provided, the window will be extended by buffer_seconds/2 seconds before and after the original range.
    """
    if ',' in ts:
        # multi-scene range: """HH:MM:SS-HH:MM:SS,HH:MM:SS-HH:MM:SS,..."""
        single_ranges = multi_range_to_single_ranges(ts)
        new_ranges = []
        for a, b in single_ranges:
            start_s = timestamp_to_seconds(a)
            end_s = timestamp_to_seconds(b)
            # apply buffer if needed
            if buffer_seconds is not None:
                start_s, end_s = apply_buffer_on_range(start_s, end_s, buffer_seconds, max_video_duration)
            new_ranges.append((start_s, end_s))
        # merge the new ranges
        new_ranges = merge_intervals(new_ranges)
        return new_ranges
    else:
        # single-scene parsing: HH:MM:SS (single timestamp) or HH:MM:SS-HH:MM:SS (range)
        if '-' in ts:
            # range: HH:MM:SS-HH:MM:SS
            start_str, end_str = [t.strip() for t in ts.split('-', 1)]
            start_s = timestamp_to_seconds(start_str)
            end_s = timestamp_to_seconds(end_str)
            # apply buffer if needed
            if buffer_seconds is not None:
                start_s, end_s = apply_buffer_on_range(start_s, end_s, buffer_seconds, max_video_duration)
        else:
            # single timestamp: HH:MM:SS -> we apply the window around it
            if buffer_seconds is None:
                raise IOError("Buffer seconds (apply_buffer) must be provided for single timestamps (HH:MM:SS). Otherwise set handle_single_timestamps to False.")
            else:
                midpoint_s = timestamp_to_seconds(ts)
                start_s, end_s = apply_buffer_on_single_timestamp(midpoint_s, buffer_seconds, max_video_duration)
    return [(start_s, end_s)]


def apply_buffer_on_range(start_s: float, end_s: float, buffer_seconds: float, max_video_duration: float) -> tuple[float, float]:
    """
    Applies a buffer window around the given range (HH:MM:SS-HH:MM:SS).
    """
    return max(0.0, start_s - buffer_seconds/2.0), min(end_s + buffer_seconds/2.0, max_video_duration)

def apply_buffer_on_single_timestamp(midpoint_s: float, buffer_seconds: float, max_video_duration: float) -> tuple[float, float]:
    """
    Applies a buffer window around the given single timestamp (HH:MM:SS).
    """
    return max(0.0, midpoint_s - buffer_seconds/2.0), min(midpoint_s + buffer_seconds/2.0, max_video_duration)


# ---------- Multi-scene range helpers ----------
def multi_range_to_single_ranges(multi_range: str) -> list[tuple[float, float]]:
    """
    Converts a multi-scene range (HH:MM:SS-HH:MM:SS,HH:MM:SS-HH:MM:SS,...) to a list of single-scene ranges.
    """
    parts = [p.strip() for p in str(multi_range).split(',')]
    single_ranges = []
    for p in parts:
        a, b = [t.strip() for t in p.split('-', 1)]
        single_ranges.append((a, b))
    return single_ranges

def merge_intervals(intervals):
    """
    Merges a list of ranges (start_s, end_s) into a list of non-overlapping ranges.
    Example:
    [(1.5, 3.2), (2.8, 5.1)] → [(1.5, 5.1)] (overlapping)
    [(1.5, 3.2), (1.3, 1.6)] → [(1.3, 3.2)] (overlapping)
    [(1.0, 3.0), (3.0, 5.0)] → [(1.0, 5.0)] (touching at boundary)
    [(1.0, 3.0), (4.0, 6.0)] → [(1.0, 3.0), (4.0, 6.0)] (non-overlapping)
    """
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged

def concat_segments(segment_paths, out_path):
    """
    Conatenates a list of segmented videos into a single video.
    """
    tmp_dir = tempfile.mkdtemp(prefix='concat_')
    try:
        list_path = os.path.join(tmp_dir, 'list.txt')
        with open(list_path, 'w') as f:
            for p in segment_paths:
                f.write(f"file '{os.path.abspath(p)}'\n")
        cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', list_path, '-c', 'copy', out_path, '-y']
        run_ffmpeg(cmd)
    except Exception as e:
        logger.error(f"Failed to concatenate segments for out_path: {out_path} -> ERROR: {e}")
        raise Exception(f"Failed to concatenate segments for out_path: {out_path} -> ERROR: {e}") from e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ---------- Video helpers ----------

def timestamp_to_seconds(ts_str) -> float:
    """
    Converts a timestamp string (HH:MM:SS) to seconds.
    """
    h, m, s = [int(x) for x in ts_str.split(':')]
    return h * 3600 + m * 60 + s

def get_video_duration(video_path) -> float:
    if not os.path.exists(video_path):
        logger.error(f"Video not found: {video_path}")
        raise FileNotFoundError(f"Video not found: {video_path}")
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, check=True, timeout=10
        )
        duration_str = r.stdout.strip()
        if not duration_str:
            logger.error(f"ffprobe returned empty output for {video_path}")
            raise ValueError(f"ffprobe returned empty output for {video_path}")
        return float(duration_str)
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe failed for {video_path}: {e.stderr}")
        raise e
    except FileNotFoundError:
        logger.error("ffprobe not found. Please install ffmpeg/ffprobe.")
        raise FileNotFoundError("ffprobe not found. Please install ffmpeg/ffprobe.")
    except ValueError as e:
        logger.error(f"Could not parse duration from ffprobe output for {video_path}: {e}")
        raise ValueError(f"Could not parse duration from ffprobe output for {video_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error getting duration for {video_path}: {e}")
        raise Exception(f"Unexpected error getting duration for {video_path}: {e}")


def run_ffmpeg(cmd) -> bool:
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        raise Exception(f"Error at run_ffmpeg: {e}") from e



def validate_timestamps(timestamps_series: pd.Series) -> None:
    """
    Validates that all timestamps are in valid format:
    - HH:MM:SS (single)
    - HH:MM:SS-HH:MM:SS (range, end > start)
    - HH:MM:SS-HH:MM:SS,HH:MM:SS-HH:MM:SS,... (multi, each end > start)
    
    Raises ValueError if any timestamp is invalid.
    """
    
    errors = []
    
    for idx, ts in timestamps_series.items():
        ts = str(ts).strip().strip('"')
        
        if not ts:
            continue
        
        if ',' in ts:
            # Multi-range format
            parts = [p.strip() for p in ts.split(',')]
            for i, p in enumerate(parts, start=1):
                if not p or p.count('-') != 1:
                    errors.append(f"Row {idx}: Invalid multi-range part '{p}' in '{ts}'")
                    continue
                
                a, b = [t.strip() for t in p.split('-', 1)]
                if not a or not b:
                    errors.append(f"Row {idx}: Empty timestamp in '{ts}'")
                    continue
                
                try:
                    start_s = timestamp_to_seconds(a)
                    end_s = timestamp_to_seconds(b)
                    if end_s <= start_s:
                        errors.append(f"Row {idx}: End <= start in '{p}' of '{ts}'")
                except (ValueError, AttributeError) as e:
                    errors.append(f"Row {idx}: Invalid timestamp format in '{p}' of '{ts}': {e}")
        
        elif '-' in ts:
            # Range format
            parts = ts.split('-', 1)
            if len(parts) != 2:
                errors.append(f"Row {idx}: Invalid range format '{ts}'")
                continue
            
            a, b = [t.strip() for t in parts]
            if not a or not b:
                errors.append(f"Row {idx}: Empty timestamp in '{ts}'")
                continue
            
            try:
                start_s = timestamp_to_seconds(a)
                end_s = timestamp_to_seconds(b)
                if end_s <= start_s:
                    errors.append(f"Row {idx}: End <= start in '{ts}'")
            except (ValueError, AttributeError) as e:
                errors.append(f"Row {idx}: Invalid timestamp format in '{ts}': {e}")
        
        else:
            # Single timestamp format
            try:
                timestamp_to_seconds(ts)
            except (ValueError, AttributeError) as e:
                errors.append(f"Row {idx}: Invalid timestamp format '{ts}': {e}")
    
    if errors:
        raise ValueError(f"Timestamp validation failed:\n" + "\n".join(errors))


# ---------- SRT utilities ----------
# SRT_RE = re.compile(r"^\s*(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})(?:\s+.*)?$")

# def load_srt_and_crop(path):
#     if not os.path.exists(path):
#         return []
#     with open(path, 'r', encoding='utf-8', errors='ignore') as f:
#         lines = f.read().splitlines()
#     blocks, i = [], 0
#     while i < len(lines):
#         if lines[i].strip().isdigit():
#             i += 1
#         if i >= len(lines):
#             break
#         m = SRT_RE.match(lines[i].strip())
#         if not m:
#             i += 1
#             continue
#         start = parse_srt_time(m.group(1))
#         end = parse_srt_time(m.group(2))
#         i += 1
#         text = []
#         while i < len(lines) and lines[i].strip() != '':
#             text.append(lines[i]); i += 1
#         while i < len(lines) and lines[i].strip() == '':
#             i += 1
#         blocks.append((start, end, text))
#     return blocks



# def parse_srt_time(t):
#     h, m, rest = t.split(':')
#     s, ms = rest.split(',')
#     return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0

# def format_srt_time(sec):
#     if sec < 0:
#         sec = 0
#     h = int(sec // 3600)
#     m = int((sec % 3600) // 60)
#     s = int(sec % 60)
#     ms = int(round((sec - int(sec)) * 1000))
#     return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

# def save_srt(blocks, out_path):
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     with open(out_path, 'w', encoding='utf-8') as f:
#         for idx, (start, end, text) in enumerate(blocks, start=1):
#             f.write(f"{idx}\n")
#             f.write(f"{format_srt_time(start)} --> {format_srt_time(end)}\n")
#             for line in text:
#                 f.write(f"{line}\n")
#             f.write("\n")

# def clip_srt_for_window(srt_blocks, clip_start_abs, clip_end_abs, shift_by):
#     out = []
#     for (s, e, text) in srt_blocks:
#         if e <= clip_start_abs or s >= clip_end_abs:
#             continue
#         ns = max(s, clip_start_abs) - shift_by
#         ne = min(e, clip_end_abs) - shift_by
#         out.append((ns, ne, text))
#     return out






