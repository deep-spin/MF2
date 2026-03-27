from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from loguru import logger

MAX_FRAMES = 256  # Original inference setting used in the paper is 384 but with these on the video+sub setting we exceed context length!
# -------- model cache --------
_MOLMO2_MODEL = None
_MOLMO2_PROCESSOR = None
_PROCESS_VISION_INFO_FUNCTION = None
# -----------------------------------

def load_model(args):
    global _MOLMO2_MODEL, _MOLMO2_PROCESSOR, _PROCESS_VISION_INFO_FUNCTION

    if _MOLMO2_MODEL is None:
        # load the processor
        _MOLMO2_PROCESSOR = AutoProcessor.from_pretrained(
            args.model,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto"
        )

        # load the model
        _MOLMO2_MODEL = AutoModelForImageTextToText.from_pretrained(
            args.model,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto"
        ).eval()
        
        from molmo_utils import process_vision_info
        import molmo_utils
        logger.info(f"molmo_utils file: {molmo_utils.__file__}")
        _PROCESS_VISION_INFO_FUNCTION = process_vision_info

    return _MOLMO2_MODEL, _MOLMO2_PROCESSOR, _PROCESS_VISION_INFO_FUNCTION


def molmo2_inference(args, input_prompts, video_path=None, system_prompt=None, shuffle_frames=None, max_frames=None):
    if shuffle_frames:
        raise ValueError("shuffle_frames is not supported for Molmo2. You need to implement your own shuffle logic, and pass the frames: multi-image setting." )
    if max_frames is None:
        max_frames = MAX_FRAMES
    model, processor, process_vision_info = load_model(args)
    responses = []
    for input_prompt in tqdm(input_prompts):
        messages = []
        # if system_prompt:
        #     messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        if video_path:
            messages.append({"role": "user", "content": [{"type": "video", "video": video_path,"backend": "pyav","num_frames": max_frames}, {"type": "text", "text": input_prompt}]})
        else:
            messages.append({"role": "user", "content": [{"type": "text", "text": input_prompt}]})
    

        # process the video using `molmo_utils.process_vision_info`
        _, videos, video_kwargs = process_vision_info(messages)
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)

        # apply the chat template to the input messages
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # process the video and text
        inputs = processor(
            videos=videos,
            video_metadata=video_metadatas,
            text=text,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # generate output
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=2048)

        # only get generated tokens; decode them to text
        generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        responses.append(generated_text)
    
    return responses


def molmo2_inference_fast(args, input_prompts, video_path=None, system_prompt=None, shuffle_frames=None, max_frames=None):
    """
    This function is a faster version of molmo2_inference. It samples frames once from the video for all the prompts used.
    """
    if shuffle_frames:
        raise ValueError("shuffle_frames is not supported for Molmo2. You need to implement your own shuffle logic, and pass the frames in multi-image setting.")
    if max_frames is None:
        max_frames = MAX_FRAMES

    model, processor, process_vision_info = load_model(args)
    responses = []

    # --- Preprocess video once if provided ---
    if video_path:
        video_message = [{"role": "user", "content": [{"type": "video", "video": video_path, "backend": "pyav", "num_frames": max_frames}]}]
        _, videos_list, video_kwargs = process_vision_info(video_message)
        videos, video_metadatas = zip(*videos_list)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        videos = None
        video_metadatas = None
        video_kwargs = {}

    for input_prompt in tqdm(input_prompts):
        # --- Build user message with video + text together ---
        if video_path:
            messages = [{"role": "user", "content": [
                {"type": "video", "video": video_path, "backend": "pyav", "num_frames": max_frames},
                {"type": "text", "text": input_prompt}
            ]}]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": input_prompt}]}]

        # --- Apply chat template for text ---
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # --- Combine preprocessed video with text ---
        inputs = processor(
            videos=videos,
            video_metadata=video_metadatas,
            text=text,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # --- Generate output ---
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=2048)

        # --- Decode only newly generated tokens ---
        generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        responses.append(generated_text)

    return responses



# dont use the functions below. They are not used in the paper.


def load_video_frames(
    video_path,
    max_frames=MAX_FRAMES,
    shuffle_frames=False,
):
    from decord import VideoReader, cpu
    from PIL import Image
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    fps = float(vr.get_avg_fps())

    # Decide target indices
    if total_frames <= max_frames:
        target_indices = list(range(total_frames))
    else:
        target_indices = np.linspace(
            0, total_frames - 1, max_frames, dtype=int
        ).tolist()

    target_set = set(target_indices)
    max_needed = max(target_indices)

    frames = []
    for i, frame in enumerate(vr):
        if i in target_set:
            img = Image.fromarray(frame.asnumpy()).convert("RGB")
            frames.append(img)

        if i >= max_needed:
            break

    if shuffle_frames:
        np.random.shuffle(frames)

    return frames, fps, total_frames


def molmo2_inference_frames(
    args,
    input_prompts,
    video_path=None,
    system_prompt=None,
    shuffle_frames=False,
    max_frames=None,
):
    if max_frames is None:
        max_frames = MAX_FRAMES
    model, processor = load_model(args)
    responses = []

    # Pre-load frames once (important for efficiency)
    if video_path:
        frames, fps, total_frames = load_video_frames(
            video_path,
            max_frames=max_frames,
            shuffle_frames=shuffle_frames,
        )
    else:
        frames = None

    for input_prompt in tqdm(input_prompts):
        messages = []

        # User message
        user_content = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        # Add frames as multi-image input
        if video_path:
            for img in frames:
                user_content.append({
                    "type": "image",
                    "image": img
                })
        # Add text last 
        user_content.append({
            "type": "text",
            "text": input_prompt
        })

        messages.append({
            "role": "user",
            "content": user_content
        })

        # Apply chat template
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=2048
            )

        # Decode only newly generated tokens
        generated_tokens = generated_ids[0, inputs["input_ids"].size(1):]
        generated_text = processor.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        responses.append(generated_text)

    return responses