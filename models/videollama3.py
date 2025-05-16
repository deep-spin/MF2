from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm
import torch

def videollama3_inference(args, input_prompts, video_path=None, system_prompt=None):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    responses = []
    for input_prompt in tqdm(input_prompts):
        conversation = []
        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})
        else:
            conversation.append({"role": "system", "content": "You are a helpful assistant."})

        if video_path:
            conversation.append(
                {"role": "user", "content": [
                    {"type": "video", "video": {
                        "video_path": video_path,
                        "fps": 1.0,
                        "max_frames": 180,
                    }},
                    {"type": "text", "text": input_prompt}
                ]}
            )
        else:
            conversation.append({"role": "user", "content": [{"type": "text", "text": input_prompt}]})
        
        inputs = processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        output_ids = model.generate(**inputs, max_new_tokens=1024,do_sample=False,temperature=0.0)
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        responses.append(response)
    return responses