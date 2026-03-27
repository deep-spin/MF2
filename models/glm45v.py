from transformers import AutoProcessor, Glm4vMoeForConditionalGeneration
import torch
from tqdm import tqdm
from loguru import logger
from openai import OpenAI

def glm45v_inference_after_serve(args, input_prompts, video_path=None, system_prompt=None, shuffle_frames=False):
    base_url="http://localhost:8000/v1"
    client = OpenAI(base_url=base_url, api_key="EMPTY")

    responses = []
    for input_prompt in tqdm(input_prompts, desc="Inference..."):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if video_path:
            video_url=f"file://{video_path}"
            messages.append({
                "role": "user",
                "content": [{"type": "video_url", "video_url": {"url": video_url}},{"type": "text", "text": input_prompt}]})
        else:
            messages.append({
                "role": "user","content": [{"type": "text", "text": input_prompt}]})
        try:
            response = client.chat.completions.create(model="glm-4.5v", messages=messages, max_tokens=4096, temperature=0.9)
        except Exception as e:
            logger.error(f"GLM-4.5V inference failed: {e}")
            response = None
        if response:
            responses.append(response.choices[0].message.content)
        else:
            responses.append(None)
    return responses