## The 3d-resampler compresses multiple frames into 64 tokens by introducing temporal_ids. 
# To achieve this, you need to organize your video data into two corresponding sequences: 
#   frames: List[Image]
#   temporal_ids: List[List[Int]].

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu    # pip install decord
from scipy.spatial import cKDTree
import numpy as np
import math
from loguru import logger
from tqdm import tqdm


MAX_NUM_FRAMES=120 # Indicates the maximum number of frames received after the videos are packed. The actual maximum number of valid frames is MAX_NUM_FRAMES * MAX_NUM_PACKING.
MAX_NUM_PACKING=3  # indicates the maximum packing number of video frames. valid range: 1-6
TIME_SCALE = 0.1 


def map_to_nearest_scale(values, scale):
    tree = cKDTree(np.asarray(scale)[:, None])
    _, indices = tree.query(np.asarray(values)[:, None])
    return np.asarray(scale)[indices]


def group_array(arr, size):
    return [arr[i:i+size] for i in range(0, len(arr), size)]

def encode_video(video_path, choose_fps=3, force_packing=None, shuffle_frames=False):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    video_duration = len(vr) / fps
        
    if choose_fps * int(video_duration) <= MAX_NUM_FRAMES:
        packing_nums = 1
        choose_frames = round(min(choose_fps, round(fps)) * min(MAX_NUM_FRAMES, video_duration))
        
    else:
        packing_nums = math.ceil(video_duration * choose_fps / MAX_NUM_FRAMES)
        if packing_nums <= MAX_NUM_PACKING:
            choose_frames = round(video_duration * choose_fps)
        else:
            choose_frames = round(MAX_NUM_FRAMES * MAX_NUM_PACKING)
            packing_nums = MAX_NUM_PACKING

    frame_idx = [i for i in range(0, len(vr))]      
    frame_idx =  np.array(uniform_sample(frame_idx, choose_frames))

    if force_packing:
        packing_nums = min(force_packing, MAX_NUM_PACKING)
    
    print(video_path, ' duration:', video_duration)
    print(f'get video frames={len(frame_idx)}, packing_nums={packing_nums}')
    
    frames = vr.get_batch(frame_idx).asnumpy()

    frame_idx_ts = frame_idx / fps
    scale = np.arange(0, video_duration, TIME_SCALE)

    frame_ts_id = map_to_nearest_scale(frame_idx_ts, scale) / TIME_SCALE
    frame_ts_id = frame_ts_id.astype(np.int32)

    assert len(frames) == len(frame_ts_id)

    frames = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in frames]
    frame_ts_id_group = group_array(frame_ts_id, packing_nums)

    if shuffle_frames:
        perm = np.random.permutation(len(frames))
        frames = [frames[i] for i in perm]

        # REMOVE temporal alignment entirely
        frame_ts_id = np.zeros(len(frames), dtype=np.int32)
        frame_ts_id_group = group_array(frame_ts_id, packing_nums)
    return frames, frame_ts_id_group


def minicpm_inference(args, input_prompts, video_path=None, system_prompt=None, shuffle_frames=False):
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,  # 'openbmb/MiniCPM-V-4_5' or openbmb/MiniCPM-o-2_6
    attn_implementation='flash_attention_2', dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)  # 'openbmb/MiniCPM-V-4_5' or openbmb/MiniCPM-o-2_6

    responses = []
    for input_prompt in tqdm(input_prompts, desc="Inference..."):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if video_path:
            fps = 5 # fps for video
            force_packing = None # You can set force_packing to ensure that 3D packing is forcibly enabled; otherwise, encode_video will dynamically set the packing quantity based on the duration.
            frames, frame_ts_id_group = encode_video(video_path, fps, force_packing=force_packing, shuffle_frames=shuffle_frames)

            messages.append({
                "role": "user",
                "content": frames+[input_prompt]})
        else:
            messages.append({
                "role": "user",
                "content": [input_prompt]})

        try:
            if video_path:
                answer = model.chat(msgs=messages,tokenizer=tokenizer, use_image_id=False, max_slice_nums=1, temporal_ids=frame_ts_id_group)
            else:
                answer = model.chat(msgs=messages,tokenizer=tokenizer, use_image_id=False, max_slice_nums=1)
        except Exception as e:
            logger.error(f"Chat failed: {str(e)}")
            answer = None
        
        responses.append(answer)
    return responses