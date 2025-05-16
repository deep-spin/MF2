from decord import VideoReader, cpu
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import copy
import numpy as np
from tqdm import tqdm
import torch

def llavavideo_inference(args, input_prompts, video_path=None, system_prompt=None):
    MAX_FRAMES=64
    fps=1.0
    
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model, None, "llava_qwen", torch_dtype="bfloat16", device_map="auto")
    model.eval()
    if video_path:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames, video_fps = len(vr), vr.get_avg_fps()
        fps = max(1, round(video_fps / fps))
        frame_idx = list(range(0, total_frames, fps))
        frame_time = [i / video_fps for i in frame_idx]
        
        if len(frame_idx) > MAX_FRAMES:
            frame_idx = np.linspace(0, total_frames - 1, MAX_FRAMES, dtype=int).tolist()
            frame_time = [i / video_fps for i in frame_idx]
        
        video_frames = vr.get_batch(frame_idx).asnumpy()

    responses = []
    
    for input_prompt in tqdm(input_prompts):
        if video_path:
            time_instruction = (
                f"The video lasts for {total_frames / video_fps:.2f} seconds, and {len(video_frames)} frames are uniformly sampled from it. "
                f"These frames are located at {', '.join([f'{t:.2f}s' for t in frame_time])}."
            )
            question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n" + input_prompt
        else:
            question = input_prompt

        conv = copy.deepcopy(conv_templates["qwen_2"])
        if system_prompt:
            conv.system = f"<|im_start|>system\n{system_prompt}"
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
        if video_path:
            video_frames_processed = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].to(torch.bfloat16)
        else:
            video_frames_processed = None
        output = model.generate(
            input_ids,
            images=video_frames_processed,
            modalities=["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=1024,
        )
        responses.append(tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip())

    return responses
