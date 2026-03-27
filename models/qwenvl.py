from transformers import AutoProcessor
import torch
from loguru import logger

# MAX_FRAMES = 1024
MAX_FRAMES = 180

from decord import VideoReader, cpu
import tempfile
import cv2
import numpy as np
import os

def extract_and_shuffle_frames(video_path, max_frames=MAX_FRAMES, shuffle=True):
    """Extract frames from video, optionally shuffle them, and return frame indices or temp video path."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    fps = float(vr.get_avg_fps())
    
    # Calculate frame indices (similar to QwenVL's sampling)
    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        # Sample evenly spaced frames
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
    
    # Shuffle frame indices if requested
    if shuffle:
        np.random.shuffle(frame_indices)
    
    # Create a temporary video with shuffled frames
    temp_video_path = None
    if shuffle:
        # Read frames in shuffled order
        frames = []
        for idx in frame_indices:
            frame = vr[idx].asnumpy()
            frames.append(frame)
        
        # Create temporary video file
        temp_fd, temp_video_path = tempfile.mkstemp(suffix='.mp4')
        os.close(temp_fd)
        
        # Write frames to temp video
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
        
        return temp_video_path, True  # Return temp path and flag that it should be cleaned up
    
    return video_path, False  # Return original path


def qwenvl_inference(args, input_prompts, video_path=None, system_prompt=None, shuffle_frames=False, max_frames=None):
    if max_frames is None:
        max_frames = MAX_FRAMES
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise RuntimeError(
            "qwenvl_inference requires vllm, but it is not installed in this environment."
        ) from e
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as e:
        raise RuntimeError(
            "qwenvl_inference requires qwen_vl_utils, but it is not installed in this environment."
        ) from e

    llm = LLM(
        model=args.model,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.9,
        dtype=torch.bfloat16,
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)

    video_inputs = None
    video_kwargs = None
    temp_video_path = None
    should_cleanup = False
    
    if video_path:
        # Extract and optionally shuffle frames
        if shuffle_frames:
            processed_video_path, should_cleanup = extract_and_shuffle_frames(
                video_path, max_frames=max_frames, shuffle=shuffle_frames
            )
            temp_video_path = processed_video_path if should_cleanup else None
        else:
            processed_video_path = video_path
        
        dummy_message = [{
            "role": "user",
            "content": [{"type": "video", "video": processed_video_path, "fps": 1.0, "max_frames": max_frames}]
        }]
        _, video_inputs, video_kwargs = process_vision_info(dummy_message, return_video_kwargs=True)

    batched_inputs = []
    for input_prompt in input_prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if video_path:
            processed_video_path = temp_video_path if temp_video_path else video_path
            messages.append({
                "role": "user",
                "content": [
                    {"type": "video", "video": processed_video_path, "max_pixels": 23520, "fps": 1.0, "max_frames": max_frames},
                    {"type": "text", "text": input_prompt}
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": input_prompt}]
            })

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        mm_data = {}
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data if mm_data else None,
            "mm_processor_kwargs": video_kwargs if mm_data else None,
        }

        batched_inputs.append(llm_inputs)

    try:
        outputs = llm.generate(batched_inputs, sampling_params=sampling_params)
        responses = [output.outputs[0].text.strip() for output in outputs]
    except Exception as e:
        logger.error(f"Batch generation failed: {str(e)}")
        responses = [None] * len(input_prompts)
    finally:
        # Clean up temporary video file if created
        if should_cleanup and temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    return responses