from tqdm import tqdm
from loguru import logger
from openai import OpenAI
import time

def inference_vllm_served_model(args, input_prompts, video_path=None, system_prompt=None, shuffle_frames=False,max_frames=None):
    base_url=f"http://{args.host}:{args.port}/v1"
    logger.info(f"Connecting to vLLM server at: {base_url}")
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
            response = client.chat.completions.create(model=args.model, messages=messages, max_tokens=2048, extra_body={"chat_template_kwargs": {"enable_thinking": True}} ) # add this for ernie temperature=0.9,extra_body={"chat_template_kwargs": {"enable_thinking": True}}
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise ValueError(f"Inference failed: {e}")
        if response:
            responses.append(response.choices[0].message.content)
        else:
            responses.append(None)
        time.sleep(0.3)
    return responses