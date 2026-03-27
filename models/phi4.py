from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np

MAX_FRAMES = 64  # Default max frames for video processing


# -------- Phi4 model cache --------
_PHI4_MODEL = None
_PHI4_PROCESSOR = None
_PHI4_GENERATION_CONFIG = None
# -----------------------------------

def load_phi4_model(args):
    global _PHI4_MODEL, _PHI4_PROCESSOR, _PHI4_GENERATION_CONFIG
    if _PHI4_MODEL is None:
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        _PHI4_MODEL = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation='flash_attention_2',
        ).eval()
        _PHI4_PROCESSOR = AutoProcessor.from_pretrained(args.model,trust_remote_code=True)
        _PHI4_GENERATION_CONFIG = GenerationConfig.from_pretrained(args.model, 'generation_config.json')
    return _PHI4_MODEL, _PHI4_PROCESSOR, _PHI4_GENERATION_CONFIG

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

def phi4_inference(args, input_prompts, video_path=None, system_prompt=None, shuffle_frames=False, max_frames=None):
    if max_frames is None:
        max_frames = MAX_FRAMES
    model, processor, generation_config = load_phi4_model(args)
    responses = []
    for input_prompt in tqdm(input_prompts):
        messages = []
        images = []
        placeholder = ''
        if video_path:
            frames, fps, total_frames = load_video_frames(video_path, max_frames=max_frames, shuffle_frames=shuffle_frames)
            for i, frame in enumerate(frames,start=1):
                images.append(frame)
                placeholder += f'<|image_{i}|>'
            messages.append({"role": "user", "content":placeholder + input_prompt})
        else:
            messages.append({"role": "user", "content": input_prompt})

        prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            prompt,
            images=images if images else None,
            return_tensors="pt",
        ).to(model.device)

        # Generation arguments -- according to the documentation of phi4
        generation_args = {
            'max_new_tokens': 1000,
            'temperature': 0.0,
            'do_sample': False,
        }
        with torch.inference_mode():
            generate_ids = model.generate(**inputs, **generation_args, generation_config=generation_config,)

        # remove input tokens
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        responses.append(response)
    return responses




def phi4_inference_fast(args, input_prompts, video_path=None, system_prompt=None, shuffle_frames=False, max_frames=None):
    """
    This function is a faster version of phi4_inference. It samples frames once from the video for all the prompts used.
    """
    if max_frames is None:
        max_frames = MAX_FRAMES

    model, processor, generation_config = load_phi4_model(args)
    responses = []

    # --- Preprocess video once ---
    if video_path:
        frames, fps, total_frames = load_video_frames(video_path, max_frames=max_frames, shuffle_frames=shuffle_frames)
        images = frames
        # Build placeholders for each frame once
        placeholder = ''.join([f'<|image_{i}|>' for i in range(1, len(frames)+1)])
    else:
        images = None
        placeholder = ''

    for input_prompt in tqdm(input_prompts):
        # --- Build message for this prompt ---
        messages = [{"role": "user", "content": placeholder + input_prompt}] if images else [{"role": "user", "content": input_prompt}]

        # --- Apply chat template ---
        prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # --- Tokenize + attach images ---
        inputs = processor(
            prompt,
            images=images if images else None,
            return_tensors="pt",
        ).to(model.device)

        # --- Generation arguments ---
        generation_args = {
            'max_new_tokens': 1000,
            'temperature': 0.0,
            'do_sample': False,
        }

        with torch.inference_mode():
            generate_ids = model.generate(
                **inputs,
                **generation_args,
                generation_config=generation_config,
            )

        # --- Remove input tokens ---
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

        # --- Decode response ---
        response = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        responses.append(response)

    return responses