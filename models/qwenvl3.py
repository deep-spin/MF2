from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from tqdm import tqdm

# MAX_FRAMES = 1024
MAX_FRAMES = 180

# def qwenvl3_inference(args, input_prompts, video_path=None, system_prompt=None, shuffle_frames=False):
#     """Inference helper for Qwen/Qwen3-VL-* models."""
#     # model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
#     #     args.model,
#     #     device_map="auto",
#     #     torch_dtype=torch.bfloat16,
#     #     trust_remote_code=True,

#     # ).eval()

#     model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-30B-A3B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     # load_in_8bit=True,
#     device_map="auto",
#     ).eval()

#     processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

#     responses = []
#     for input_prompt in tqdm(input_prompts):
#         messages = []
#         if system_prompt:
#             messages.append({
#                 "role": "system",
#                 "content": [{"type": "text", "text": system_prompt}]
#             })

#         user_content = []
#         if video_path:
#             user_content.append({
#                 "type": "video",
#                 "video": video_path,
#                 "fps": 1.0,
#                 "max_frames": MAX_FRAMES,
#             })
#         user_content.append({"type": "text", "text": input_prompt})

#         messages.append({"role": "user", "content": user_content})

#         inputs = processor.apply_chat_template(
#             messages,
#             tokenize=True,
#             add_generation_prompt=True,
#             return_dict=True,
#             return_tensors="pt",
#         ).to(model.device, dtype=torch.bfloat16)

#         input_len = inputs["input_ids"].shape[-1]
#         with torch.inference_mode():
#             output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
#         output_ids = output[:, input_len:]

#         decoded = processor.batch_decode(
#             output_ids,
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=False,
#         )
#         responses.append(decoded[0].strip())

#     return responses



def qwenvl3_inference(args, input_prompts, video_path=None, system_prompt=None, shuffle_frames=False):
    """Inference helper for Qwen/Qwen3 VL models with optional video input."""
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=False,
        attn_implementation="flash_attention_2",
    ).eval()

    responses = []
    for input_prompt in tqdm(input_prompts):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = []
        if video_path:
            user_content.append({
                "type": "video",
                "video": video_path,
                "fps": 1.0,
                "max_frames": MAX_FRAMES,
            })
        user_content.append({"type": "text", "text": input_prompt})

        messages.append({"role": "user", "content": user_content})

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated)
        ]
        decoded = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        responses.append(decoded[0].strip())

    return responses



def _prepare_vllm_inputs(messages, processor):
    from qwen_vl_utils import process_vision_info
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    return {
        "prompt": prompt,
        "multi_modal_data": mm_data if mm_data else None,
        "mm_processor_kwargs": video_kwargs,
    }


def qwenvl3_inference_vllm(args, input_prompts, video_path=None, system_prompt=None, shuffle_frames=False, max_frames=None):
    if max_frames is None:
        max_frames = MAX_FRAMES
    """Inference helper for Qwen/Qwen3 VL models using vLLM."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise RuntimeError(
            "qwenvl3_inference_vllm requires vllm, but it is not installed in this environment."
        ) from e
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        mm_encoder_tp_mode="data",
        enable_expert_parallel=True,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        top_k=-1,
        stop_token_ids=[],
    )

    batched_inputs = []
    for input_prompt in tqdm(input_prompts):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = []
        if video_path:
            user_content.append({
                "type": "video",
                "video": video_path,
                "fps": 1.0,
                "max_frames": max_frames,
            })
        user_content.append({"type": "text", "text": input_prompt})
        messages.append({"role": "user", "content": user_content})

        batched_inputs.append(_prepare_vllm_inputs(messages, processor))

    outputs = llm.generate(batched_inputs, sampling_params=sampling_params)
    responses = [output.outputs[0].text.strip() if output.outputs else "" for output in outputs]
    return responses