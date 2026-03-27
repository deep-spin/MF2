
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np

MAX_FRAMES = 64  # Default max frames for video processing


# -------- Gemma3 model cache --------
_GEMMA3_MODEL = None
_GEMMA3_PROCESSOR = None
# -----------------------------------

def load_gemma3_model(args):
    global _GEMMA3_MODEL, _GEMMA3_PROCESSOR

    if _GEMMA3_MODEL is None:
        _GEMMA3_MODEL = Gemma3ForConditionalGeneration.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()

        _GEMMA3_PROCESSOR = AutoProcessor.from_pretrained(args.model)

    return _GEMMA3_MODEL, _GEMMA3_PROCESSOR

def load_video_frames(video_path, max_frames=MAX_FRAMES, shuffle_frames=False):
    """Extract frames from video and return as list of PIL Images."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    fps = float(vr.get_avg_fps())
    
    # Uniform sampling
    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
    
    # Shuffle frame indices if requested
    if shuffle_frames:
        np.random.shuffle(frame_indices)
    
    # Extract frames and convert to PIL Images
    frames = []
    for idx in frame_indices:
        frame = vr[idx].asnumpy()
        img = Image.fromarray(frame).convert('RGB')
        frames.append(img)
    
    return frames, fps, total_frames

def gemma3_inference(args, input_prompts, video_path=None, system_prompt=None, shuffle_frames=False):
    """Gemma3 inference following internvl pattern."""
    # Load model and processor
    # model = Gemma3ForConditionalGeneration.from_pretrained(
    #     args.model,
    #     device_map="auto",
    #     torch_dtype=torch.bfloat16,
    # ).eval()
    # processor = AutoProcessor.from_pretrained(args.model)

    model, processor = load_gemma3_model(args)

    # Load video frames if provided
    video_frames = None
    if video_path:
        video_frames, fps, total_frames = load_video_frames(
            video_path, max_frames=MAX_FRAMES, shuffle_frames=shuffle_frames
        )
    
    responses = []
    
    for input_prompt in tqdm(input_prompts):
        # Build messages following Gemma3 format
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        # Add user message with video frames (as images) or text
        if video_path and video_frames:
            # Add each frame as an image
            content = []
            for frame in video_frames:
                content.append({"type": "image", "image": frame})
            content.append({"type": "text", "text": input_prompt})
            messages.append({
                "role": "user",
                "content": content
            })
        else:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": input_prompt}]
            })
        
        # Apply chat template
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate response
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False
            )
            generation = generation[0][input_len:]
        
        # Decode response
        decoded = processor.decode(generation, skip_special_tokens=True)
        responses.append(decoded)
    
    return responses