import tempfile
import numpy as np
import os
import torch
from decord import VideoReader, cpu
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


MAX_FRAMES = 14  # CogVLM2 uses 24 frames by default


def load_video(video_path, shuffle_frames=False):
    """Load video frames using uniform sampling."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    # Uniform sampling using np.linspace 
    if total_frames <= MAX_FRAMES:
        frame_indices = list(range(total_frames))
    else:
        # Sample evenly spaced frames
        frame_indices = np.linspace(0, total_frames - 1, MAX_FRAMES, dtype=int).tolist()
    # Shuffle frame indices if requested
    if shuffle_frames:
        np.random.shuffle(frame_indices)
    
    # Get frames in the selected order
    video_data = vr.get_batch(frame_indices)  # Returns numpy NDArray
    # Convert to torch tensor and permute from (T, H, W, C) to (C, T, H, W)
    video_data = torch.from_numpy(video_data.asnumpy()).permute(3, 0, 1, 2)
    return video_data

def load_video_v2(video_path, shuffle_frames=False):
    """Load video frames using timestamp-based sampling (one frame per second)."""
    try:
        with open(video_path, 'rb') as f:
            mp4_stream = f.read()
        decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))
    except:
        decord_vr = VideoReader(video_path, ctx=cpu(0))
    
    total_frames = len(decord_vr)
    
    # Timestamp-based: one frame per second
    timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
    timestamps = [i[0] for i in timestamps]
    max_second = round(max(timestamps)) + 1
    frame_id_list = []
    
    for second in range(max_second):
        closest_num = min(timestamps, key=lambda x: abs(x - second))
        index = timestamps.index(closest_num)
        frame_id_list.append(index)
        if len(frame_id_list) >= MAX_FRAMES:
            break
    
    # Shuffle frame indices if requested
    if shuffle_frames:
        np.random.shuffle(frame_id_list)
    
    # Get frames in the selected order
    video_data = decord_vr.get_batch(frame_id_list)  # Returns numpy NDArray
    # Convert to torch tensor and permute from (T, H, W, C) to (C, T, H, W)
    video_data = torch.from_numpy(video_data.asnumpy()).permute(3, 0, 1, 2)
    
    return video_data

# def cogvlm2_inference(args, input_prompts, video_path=None, system_prompt=None, shuffle_frames=False):
#     # Load model with device_map='auto' - let transformers handle device placement
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model,
#         torch_dtype=torch.bfloat16,
#         device_map='auto',
#         low_cpu_mem_usage=True,
#         trust_remote_code=True
#     ).eval()
    
#     tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
#     # Load video if provided
#     video = None
#     if video_path:
#         video = load_video(video_path,  shuffle_frames=shuffle_frames)
    
#     responses = []
#     history = []  # Conversation history
    
#     for input_prompt in tqdm(input_prompts):
#         # Prepare query - prepend system prompt if provided
#         if system_prompt:
#             query = f"{system_prompt}\n\n{input_prompt}"
#         else:
#             query = input_prompt
        
#         # Build conversation input IDs
#         if video_path and video is not None:
#             inputs = model.build_conversation_input_ids(
#                 tokenizer=tokenizer,
#                 query=query,
#                 images=[video],
#                 history=history,
#                 template_version='chat'
#             )
#         else:
#             inputs = model.build_conversation_input_ids(
#                 tokenizer=tokenizer,
#                 query=query,
#                 images=None,
#                 history=history,
#                 template_version='chat'
#             )
        
#         # Prepare inputs for generation - use model.device instead of hardcoded DEVICE
#         if video_path and video is not None:
#             inputs_dict = {
#                 'input_ids': inputs['input_ids'].unsqueeze(0).to(model.device),
#                 'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(model.device),
#                 'attention_mask': inputs['attention_mask'].unsqueeze(0).to(model.device),
#                 'images': [[inputs['images'][0].to(model.device).to(torch.bfloat16)]],
#             }
#         else:
#             inputs_dict = {
#                 'input_ids': inputs['input_ids'].unsqueeze(0).to(model.device),
#                 'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(model.device),
#                 'attention_mask': inputs['attention_mask'].unsqueeze(0).to(model.device),
#             }
        
#         # Generation kwargs
#         gen_kwargs = {
#             "max_new_tokens": 3433,
#             "pad_token_id": 128002,
#             "top_k": 1,
#             "do_sample": True,
#             "top_p": 0.1,
#             "temperature": 0.1,
#         }
        
#         # Generate response
#         with torch.no_grad():
#             outputs = model.generate(**inputs_dict, **gen_kwargs)
#             outputs = outputs[:, inputs_dict['input_ids'].shape[1]:]
#             response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         responses.append(response)
        
#         # Update history for multi-turn conversation
#         history.append((query, response))
    
#     return responses


def cogvlm2_inference(args, input_prompts, video_path=None, system_prompt=None, shuffle_frames=False):
    # Load model with device_map='auto' - let transformers handle device placement
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Get the device from the model's first parameter (works with device_map='auto')
    device = next(model.parameters()).device
    
    # Load video if provided
    video = None
    if video_path:
        video = load_video(video_path, shuffle_frames=shuffle_frames)
        # Move video to the correct device
        video = video.to(device)
    
    responses = []
    history = []  # Conversation history
    
    for input_prompt in tqdm(input_prompts):
        # Prepare query - prepend system prompt if provided
        if system_prompt:
            query = f"{system_prompt}\n\n{input_prompt}"
        else:
            query = input_prompt
        
        # Build conversation input IDs
        if video_path and video is not None:
            inputs = model.build_conversation_input_ids(
                tokenizer=tokenizer,
                query=query,
                images=[video],
                history=history,
                template_version='chat'
            )
        else:
            inputs = model.build_conversation_input_ids(
                tokenizer=tokenizer,
                query=query,
                images=None,
                history=history,
                template_version='chat'
            )
        
        # Prepare inputs for generation - use device from model parameters
        if video_path and video is not None:
            inputs_dict = {
                'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
                'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
                'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
                'images': [[inputs['images'][0].to(device).to(torch.bfloat16)]],
            }
        else:
            inputs_dict = {
                'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
                'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
                'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
            }

        # Generation kwargs
        gen_kwargs = {
            "max_new_tokens": 3433,
            "pad_token_id": 128002,
            "top_k": 1,
            "do_sample": True,
            "top_p": 0.1,
            "temperature": 0.1,
        }
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(**inputs_dict, **gen_kwargs)
            outputs = outputs[:, inputs_dict['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        responses.append(response)
        
        # Update history for multi-turn conversation
        history.append((query, response))
    
    return responses