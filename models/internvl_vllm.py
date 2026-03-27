import math
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=12, num_segments=64, shuffle_frames=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    if shuffle_frames:
        np.random.shuffle(frame_indices)

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = torch.stack([transform(tile) for tile in tiles])
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def internvl3_inference_vllm(args, input_prompts, video_path=None, system_prompt=None, shuffle_frames=False):
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        top_k=-1,
        top_p=1.0,
        stop_token_ids=[],
    )

    if video_path:
        pixel_values, num_patches_list = load_video(
            video_path,
            num_segments=64,
            max_num=12,
            shuffle_frames=shuffle_frames,
        )
        pixel_values = pixel_values.to(torch.bfloat16)
        pixel_values_np = pixel_values.cpu().numpy()
        frames = []
        idx = 0
        for patches in num_patches_list:
            frame = pixel_values_np[idx:idx+patches]
            frames.append(frame)
            idx += patches
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(frames))])
    else:
        frames = None
        video_prefix = None

    batched_inputs = []
    for prompt in tqdm(input_prompts):
        full_prompt = video_prefix + prompt if (video_prefix and video_path) else prompt
        conversation = []
        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": full_prompt})
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        mm_data = {}
        if frames is not None:
            mm_data['image'] = frames  # treat each tile as image patches

        llm_inputs = {
            "prompt": text,
            "multi_modal_data": mm_data if mm_data else None,
        }
        batched_inputs.append(llm_inputs)

    outputs = llm.generate(batched_inputs, sampling_params=sampling_params)
    responses = [output.outputs[0].text.strip() for output in outputs]
    return responses