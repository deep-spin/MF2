from decord import VideoReader, cpu
import tempfile
import cv2
import numpy as np
import os
from tqdm import tqdm
import torch
from loguru import logger

MAX_FRAMES = 180

# -------- model cache --------
_MIMOVL_MODEL = None
_MIMOVL_PROCESSOR = None
# -----------------------------------

def load_model(args):
    global _MIMOVL_MODEL, _MIMOVL_PROCESSOR
    if _MIMOVL_MODEL is None:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        _MIMOVL_MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        ).eval()
        _MIMOVL_PROCESSOR = AutoProcessor.from_pretrained("XiaomiMiMo/MiMo-VL-7B-RL")
    return _MIMOVL_MODEL, _MIMOVL_PROCESSOR

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

def mimovl_inference(args, input_prompts, video_path=None, system_prompt=None, shuffle_frames=False):
    from qwen_vl_utils import process_vision_info
    model, processor = load_model(args)

    responses = []
    for input_prompt in tqdm(input_prompts, desc="Inference..."):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if video_path:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "fps": 1.0, "max_frames": MAX_FRAMES},
                    {"type": "text", "text": input_prompt}
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": input_prompt}]
            })

        # prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",)
        inputs = inputs.to(model.device)

        try:
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=1024,
                pad_token_id=processor.tokenizer.eos_token_id
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            output_text = [None] * len(input_prompts)

        responses.append(output_text[0])

    

    return responses

