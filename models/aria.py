import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List
from decord import VideoReader

import torch
from transformers import AriaProcessor, AriaForConditionalGeneration
# from transformers import BitsAndBytesConfig


MAX_FRAMES = 64

def load_video(video_file, num_frames=MAX_FRAMES, cache_dir="cached_video_frames", verbosity="INFO", shuffle_frames=False):
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    video_basename = os.path.basename(video_file)
    cache_subdir = os.path.join(cache_dir, f"{video_basename}_{num_frames}")
    os.makedirs(cache_subdir, exist_ok=True)

    cached_frames = []
    missing_frames = []
    frame_indices = []
    
    for i in range(num_frames):
        frame_path = os.path.join(cache_subdir, f"frame_{i}.jpg")
        if os.path.exists(frame_path):
            cached_frames.append(frame_path)
        else:
            missing_frames.append(i)
            frame_indices.append(i) 
            
    vr = VideoReader(video_file)
    duration = len(vr)
    fps = vr.get_avg_fps()
            
    frame_timestamps = [int(duration / num_frames * (i+0.5)) / fps for i in range(num_frames)]
    
    if verbosity == "DEBUG":
        print("Already cached {}/{} frames for video {}, enjoy speed!".format(len(cached_frames), num_frames, video_file))
    # If all frames are cached, load them directly
    if not missing_frames:
        return [Image.open(frame_path).convert("RGB") for frame_path in cached_frames], frame_timestamps

    

    actual_frame_indices = [int(duration / num_frames * (i+0.5)) for i in missing_frames]


    missing_frames_data = vr.get_batch(actual_frame_indices).asnumpy()

    for idx, frame_index in enumerate(tqdm(missing_frames, desc="Caching rest frames")):
        img = Image.fromarray(missing_frames_data[idx]).convert("RGB")
        frame_path = os.path.join(cache_subdir, f"frame_{frame_index}.jpg")
        img.save(frame_path)
        cached_frames.append(frame_path)

    # Sort frames by index
    cached_frames.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    frames = [Image.open(fp).convert("RGB") for fp in cached_frames]

    # Shuffle frames and timestamps if requested
    if shuffle_frames:
        combined = list(zip(frames, frame_timestamps))
        np.random.shuffle(combined)
        frames, frame_timestamps = zip(*combined)

    return list(frames), list(frame_timestamps)

def create_image_gallery(images, columns=3, spacing=20, bg_color=(200, 200, 200)):
    """
    Combine multiple images into a single larger image in a grid format.
    
    Parameters:
        image_paths (list of str): List of file paths to the images to display.
        columns (int): Number of columns in the gallery.
        spacing (int): Space (in pixels) between the images in the gallery.
        bg_color (tuple): Background color of the gallery (R, G, B).
    
    Returns:
        PIL.Image: A single combined image.
    """
    # Open all images and get their sizes
    img_width, img_height = images[0].size  # Assuming all images are of the same size

    # Calculate rows needed for the gallery
    rows = (len(images) + columns - 1) // columns

    # Calculate the size of the final gallery image
    gallery_width = columns * img_width + (columns - 1) * spacing
    gallery_height = rows * img_height + (rows - 1) * spacing

    # Create a new image with the calculated size and background color
    gallery_image = Image.new('RGB', (gallery_width, gallery_height), bg_color)

    # Paste each image into the gallery
    for index, img in enumerate(images):
        row = index // columns
        col = index % columns

        x = col * (img_width + spacing)
        y = row * (img_height + spacing)

        gallery_image.paste(img, (x, y))

    return gallery_image


def get_placeholders_for_videos(frames: List, timestamps=[]):
    contents = []
    if not timestamps:
        for i, _ in enumerate(frames):
            contents.append({"text": None, "type": "image"})
        contents.append({"text": "\n", "type": "text"})
    else:
        for i, (_, ts) in enumerate(zip(frames, timestamps)):
            contents.extend(
                [
                    {"text": f"[{int(ts)//60:02d}:{int(ts)%60:02d}]", "type": "text"},
                    {"text": None, "type": "image"},
                    {"text": "\n", "type": "text"}
                ]
            )
    return contents


def aria_inference(
    args,
    input_prompts,
    video_path=None,  # keep argument name for consistency, but used for images
    system_prompt=None,
    shuffle_frames=False,
    max_frames=None
):
    """
    Inference for AriaForConditionalGeneration following Gemma3 pattern.
    
    Args:
        model_id_or_path: Hugging Face model ID or local path
        input_prompts: List of text prompts
        video_path: Optional list of images or single URL (kept same as Gemma3 arg)
        system_prompt: Optional system prompt text
        shuffle_frames: Whether to shuffle images
    Returns:
        List of generated responses
    """
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True) # or load_in_4bit=True
    if max_frames is None:
        max_frames = MAX_FRAMES
    # Load model and processor
    model = AriaForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        # quantization_config=quantization_config
    ).eval()
    processor = AriaProcessor.from_pretrained(args.model, trust_remote_code=True)

    if video_path:
        frames, frame_timestamps = load_video(video_path, num_frames=max_frames, shuffle_frames=shuffle_frames)
        if shuffle_frames:
            video_contents = get_placeholders_for_videos(frames)
        else:   
            video_contents = get_placeholders_for_videos(frames)
    else:
        frames = None
        video_contents = None


    responses = []
    for prompt in tqdm(input_prompts):
        # Construct messages
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        if video_contents:
            messages.append({
                "role": "user",
                "content": [
                    *video_contents,
                    {"text": prompt, "type": "text"},
                ],
            })
        else:
            messages.append({
                "role": "user",
                "content": [
                    {"text": prompt, "type": "text"},
                ],
            })

        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        if frames:
            inputs = processor(text=text, images=frames, return_tensors="pt")
        else:
            inputs = processor(text=text, return_tensors="pt")
        if "pixel_values" in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.9, stop_strings=["<|im_end|>"])
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = model.generate(
                **inputs,
                max_new_tokens=2048,
                stop_strings=["<|im_end|>"],
                tokenizer=processor.tokenizer,
                do_sample=False,
                temperature=0.,
            )
            output_ids = output[0][inputs["input_ids"].shape[1]:]
            result = processor.decode(output_ids, skip_special_tokens=True)
            if "<|im_end|>" in result:
                result = result.split("<|im_end|>")[0]
        responses.append(result)
    return responses