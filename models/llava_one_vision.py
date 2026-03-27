from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from tqdm import tqdm
import torch

# 7B -> 1 A6000 | 72B -> 8 A6000 or 4 A100 80GB

def llavaonevision_inference(args, input_prompts, video_path=None, system_prompt=None,shuffle_frames=False):
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model)

    responses = []

    for input_prompt in tqdm(input_prompts):
        conversation = []
        if system_prompt:
            conversation.append(
                {"role": "system", "content": [
                    {"type": "text", "text": system_prompt}
                ]}
            )
        if video_path:
            conversation.append(
                {"role": "user", "content": [
                {"type": "video", "path": video_path},
                {"type": "text", "text": input_prompt}
                ]}
            )
        else:
            conversation.append(
                {"role": "user", "content": [
                    {"type": "text", "text": input_prompt}
                ]}
            )
        if video_path:
            inputs = processor.apply_chat_template(
            conversation,
            num_frames=32, # training frames!
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
            ).to(model.device, torch.float16)
        else:
            inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt").to(model.device, torch.float16)

        output_ids = model.generate(**inputs, max_new_tokens=1024)
        response = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
        responses.append(response)

    return responses