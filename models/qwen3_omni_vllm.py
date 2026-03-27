import os
import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor
from vllm import LLM, SamplingParams
import re

os.environ.setdefault('VLLM_USE_V1', '0')
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")


# def qwen_omni_inference_vllm(args, input_prompts, video_path=None, system_prompt=None, shuffle_frames=False, use_audio=False):
#     """Qwen3-Omni inference using vLLM."""
    
#     # Load vLLM model
#     llm = LLM(
#         model=args.model,
#         trust_remote_code=True,
#         dtype=torch.bfloat16,
#         gpu_memory_utilization=0.95,
#         tensor_parallel_size=torch.cuda.device_count(),
#         limit_mm_per_prompt={'image': 1, 'video': 1, 'audio': 1},
#         max_num_seqs=1,
#         max_model_len=32768,
#         seed=1234,
#     )
    
#     processor = Qwen3OmniMoeProcessor.from_pretrained(args.model)
#     sampling_params = SamplingParams(
#         temperature=0.0,
#         max_tokens=3000,
#     )
    
#     batched_inputs = []
#     for input_prompt in input_prompts:
#         # Build messages
#         messages = []
#         # if system_prompt:
#         #     messages.append({
#         #         "role": "system",
#         #         "content": [{"type": "text", "text": system_prompt}]
#         #     })
        
#         user_content = []
#         if video_path:
#             user_content.append({
#                 "type": "video",
#                 "video": video_path
#             })
#         user_content.append({"type": "text", "text": input_prompt})
        
#         messages.append({
#             "role": "user",
#             "content": user_content
#         })
        
#         # Prepare inputs for vLLM
#         text = processor.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
        
#         audios, images, videos = process_mm_info(
#             messages,
#             use_audio_in_video=use_audio
#         )
        
#         inputs = {
#             'prompt': text,
#             'multi_modal_data': {},
#             "mm_processor_kwargs": {"use_audio_in_video": use_audio}
#         }
        
#         if images is not None:
#             inputs['multi_modal_data']['image'] = images
#         if videos is not None:
#             inputs['multi_modal_data']['video'] = videos
#         if audios is not None:
#             inputs['multi_modal_data']['audio'] = audios
        
#         batched_inputs.append(inputs)
    
#     # Generate responses
#     outputs = llm.generate(batched_inputs, sampling_params=sampling_params)
#     responses = [output.outputs[0].text.strip() for output in outputs]
    
#     return responses

from decord import VideoReader, cpu
from PIL import Image
import numpy as np


def extract_frames_uniformly(video_path, num_frames):
    """Extract frames uniformly from video and return as list of PIL Images."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    
    # Uniform sampling using np.linspace
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        # Sample evenly spaced frames
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    
    # Extract frames and convert to PIL Images
    frames = []
    for idx in frame_indices:
        frame = vr[idx].asnumpy()
        img = Image.fromarray(frame).convert('RGB')
        frames.append(img)
    
    return frames



def qwen_omni_inference_vllm(args, input_prompts, video_path=None, system_prompt=None, shuffle_frames=False, use_audio=False, num_frames=20):
    """Qwen3-Omni inference using vLLM with uniform frame sampling."""
    
    # Load vLLM model
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        # dtype=torch.bfloat16,
        # gpu_memory_utilization=0.95,
        tensor_parallel_size=torch.cuda.device_count(),
        limit_mm_per_prompt={'image': 26, 'video': 0, 'audio': 1},
        max_num_seqs=1,
        seed=1234,
    )
    
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
    )
    
    # Extract frames if video_path is provided
    video_frames = None
    if video_path:
        video_frames = extract_frames_uniformly(video_path, num_frames)
        if shuffle_frames:
            np.random.shuffle(video_frames)
    
    batched_inputs = []
    all_responses = []
    for input_prompt in input_prompts:
        # Build messages
        messages = []
        # if system_prompt:
        #     messages.append({
        #         "role": "system",
        #         "content": [{"type": "text", "text": system_prompt}]
        #     })
        
        user_content = []
        # Add frames as images instead of video
        if video_path and video_frames:
            for frame in video_frames:
                user_content.append({
                    "type": "image",
                    "image": frame
                })
        user_content.append({"type": "text", "text": input_prompt})
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Prepare inputs for vLLM
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        audios, images, videos = process_mm_info(
            messages,
            use_audio_in_video=use_audio
        )
        inputs = {
            'prompt': text,
            'multi_modal_data': {},
            "mm_processor_kwargs": {"use_audio_in_video": use_audio}
        }
        
        if images is not None:
            inputs['multi_modal_data']['image'] = images
        if videos is not None:
            inputs['multi_modal_data']['video'] = videos
        if audios is not None:
            inputs['multi_modal_data']['audio'] = audios
        breakpoint()
        batched_inputs.append(inputs)    # Generate responses
        outputs = llm.generate(batched_inputs, sampling_params=sampling_params)
        responses = [output.outputs[0].text.strip() for output in outputs]
        all_responses.extend(responses)
    breakpoint()
    return all_responses


def qwen_omni_inference_vllm_with_audio(args, input_prompts, video_path=None, audio_path=None, system_prompt=None, shuffle_frames=False, num_frames=30):
    """Qwen3-Omni inference using vLLM with uniform frame sampling and audio extraction."""
    #let's hard code the audio path for now, the video path ends in .mp4, so the audio path ends in .wav
    match = re.search(r'(\d+)\.mp4', video_path)
    if match:
        movie_id = int(match.group(1))  # Returns: 1
    audio_path = f"/mnt/data-poseidon/manos/projects/f3-project/official_repo/MF2/data/audios_processed_4k/{movie_id}.wav"

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at {audio_path}")

    
    # Load vLLM model
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        # dtype=torch.bfloat16,
        # gpu_memory_utilization=0.95,
        tensor_parallel_size=torch.cuda.device_count(),
        limit_mm_per_prompt={'image': num_frames, 'video': 0, 'audio': 1},
        max_num_seqs=1,
        seed=1234,
    )
    
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
    )
    
    # Extract frames if video_path is provided
    video_frames = None
    if video_path:
        video_frames = extract_frames_uniformly(video_path, num_frames)
        if shuffle_frames:
            np.random.shuffle(video_frames)
    
    # Extract or use provided audio
    extracted_audio_path = None
    if video_path:
        if audio_path and os.path.exists(audio_path):
            # Use provided audio path if it exists
            extracted_audio_path = audio_path
        else:
            # Extract audio from video
            raise NotImplementedError("Audio extraction from video is not implemented yet. Use the extract_audio_from_video.py script to extract the audio from the video.")
            extracted_audio_path = extract_audio_from_video(video_path)
    
    batched_inputs = []
    all_responses = []
    for input_prompt in input_prompts:
        # Build messages
        messages = []
        # if system_prompt:
        #     messages.append({
        #         "role": "system",
        #         "content": [{"type": "text", "text": system_prompt}]
        #     })
        
        user_content = []
        # Add frames as images instead of video
        if video_path and video_frames:
            for frame in video_frames:
                user_content.append({
                    "type": "image",
                    "image": frame
                })
        
        # Add audio if available
        if extracted_audio_path:
            user_content.append({
                "type": "audio",
                "audio": extracted_audio_path
            })
        
        user_content.append({"type": "text", "text": input_prompt})
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Prepare inputs for vLLM
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        audios, images, videos = process_mm_info(
            messages,
            use_audio_in_video=True  # We're passing audio separately, not from video
        )
        breakpoint()
        
        inputs = {
            'prompt': text,
            'multi_modal_data': {},
            "mm_processor_kwargs": {"use_audio_in_video": False}
        }
        
        if images is not None:
            inputs['multi_modal_data']['image'] = images
        if videos is not None:
            inputs['multi_modal_data']['video'] = videos
        if audios is not None:
            inputs['multi_modal_data']['audio'] = audios
        
    
    # Generate responses
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        responses = [output.outputs[0].text.strip() for output in outputs]
        all_responses.extend(responses)
    breakpoint()
    return all_responses