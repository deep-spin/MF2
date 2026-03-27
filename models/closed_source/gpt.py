
import cv2, base64, litellm
from loguru import logger
import os

MAX_FRAMES = 50

def sample_frames_to_b64(video_path, num_frames):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total <= 0 or fps <= 0:
        cap.release()
        logger.error(f"Cannot open/read: {video_path}")
        return frames
    step = max(total // max(num_frames, 1), 1)
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok: break
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok: continue
        frames.append(base64.b64encode(buf).decode("utf-8"))
        if len(frames) >= num_frames: break
    cap.release()
    return frames

def frames_to_image_parts(frames_b64):
    return [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} for b64 in frames_b64]


def build_message(prompt_text, video_path=None, system_prompt=None,  num_frames=None):
    if num_frames is None:
        logger.error("num_frames is required.Should not be None.")
        raise ValueError("num_frames is required.Should not be None.")
    message = []
    if system_prompt:
        message.append({"role": "system", "content": system_prompt})
   
    if video_path:
        image_parts = frames_to_image_parts(sample_frames_to_b64(video_path, num_frames=num_frames))
        message.append({"role": "user", "content": [ *image_parts, {"type": "text", "text": prompt_text}]})
    else:
        image_parts = []
        message.append({"role": "user", "content": [{"type": "text", "text": prompt_text}]})
    return message


def litellm_model_inference(model_provider, api_key, api_base, prompt_text=None, video_path=None, system_prompt=None, shuffle_frames=False, num_frames=None):

    if num_frames is None:
        num_frames = MAX_FRAMES
    
    if shuffle_frames:
        logger.error("Shuffling frames is not implemented yet for closed-source models.")
        raise NotImplementedError("Shuffling frames is not implemented yet for closed-source models.")
    
    if prompt_text is not None and isinstance(prompt_text, list):
        prompt_text = prompt_text[0]
        logger.debug(f"list passed as prompt_text. Using first element as prompt_text.")
    elif prompt_text is not None and isinstance(prompt_text, str):
        logger.debug(f"string passed as prompt_text. Using it as is.")
    else:
        logger.error(f"Invalid prompt_text!")
        raise ValueError("Invalid prompt_text!")

    message = build_message(prompt_text=prompt_text, video_path=video_path, system_prompt=system_prompt, num_frames=num_frames)

    try:
        resp = litellm.completion(
            model=model_provider,
            messages=message,
            temperature=0,
            max_tokens=2056,
            api_key=None if api_key == "None" else api_key,
            base_url=api_base
        )
        if hasattr(resp, "choices") and resp.choices:
            return resp.choices[0].message.get("content", "")
        return getattr(resp, "content", "") or ""
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return ""


def openai_model_inference(model_provider, api_key, api_base, prompt_text, video_path=None, system_prompt=None, num_frames=3600):
    raise NotImplementedError("OpenAI model inference is not implemented yet.")