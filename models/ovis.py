import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from moviepy.editor import VideoFileClip
from tqdm import tqdm

MAX_FRAMES=10

def ovis_inference(args, input_prompts, video_path=None, system_prompt=None):

    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                torch_dtype=torch.bfloat16,
                                                multimodal_max_length=32768,
                                                device_map='auto',
                                                trust_remote_code=True)
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()

    responses = []
    for input_prompt in tqdm(input_prompts):
        if video_path:
            with VideoFileClip(args.video_path) as clip:
                total_frames = int(clip.fps * clip.duration)
                if total_frames <= MAX_FRAMES:
                    sampled_indices = range(total_frames)
                else:
                    stride = total_frames / MAX_FRAMES
                    sampled_indices = [min(total_frames - 1, int((stride * i + stride * (i + 1)) / 2)) for i in range(MAX_FRAMES)]
                frames = [clip.get_frame(index / clip.fps) for index in sampled_indices]
                frames = [Image.fromarray(frame, mode='RGB') for frame in frames]
            images = frames
            
            prompt = '\n'.join(['<image>'] * len(images)) + '\n' + input_prompt
        else:
            prompt = input_prompt
            images = None
        prompt, input_ids, pixel_values = model.preprocess_inputs(prompt, images, max_partition=1)
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
        pixel_values = [pixel_values]

        gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True
        )
        output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
        output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
        responses.append(output)
    return responses