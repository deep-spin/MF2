from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from loguru import logger

MAX_FRAMES = 180

def qwenvl_inference(args, input_prompts, video_path=None, system_prompt=None):
    llm = LLM(
        model=args.model,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.9,
        dtype=torch.bfloat16,
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)

    video_inputs = None
    video_kwargs = None
    if video_path:
        dummy_message = [{
            "role": "user",
            "content": [{"type": "video", "video": video_path, "fps": 1.0, "max_frames": MAX_FRAMES}]
        }]
        _, video_inputs, video_kwargs = process_vision_info(dummy_message, return_video_kwargs=True)

    batched_inputs = []
    for input_prompt in input_prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if video_path:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "max_pixels": 23520, "fps": 1.0, "max_frames": MAX_FRAMES},
                    {"type": "text", "text": input_prompt}
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": input_prompt}]
            })

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        mm_data = {}
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data if mm_data else None,
            "mm_processor_kwargs": video_kwargs if mm_data else None,
        }

        batched_inputs.append(llm_inputs)

    try:
        outputs = llm.generate(batched_inputs, sampling_params=sampling_params)
        responses = [output.outputs[0].text.strip() for output in outputs]
    except Exception as e:
        logger.error(f"Batch generation failed: {str(e)}")
        responses = [None] * len(input_prompts)

    return responses