from pathlib import Path
import os
import base64

import cv2, litellm
from loguru import logger
# litellm._turn_on_debug()

MAX_FRAMES = 120

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
    return [{"type": "image_url", "image_url":{"url": f"data:image/jpeg;base64,{b64}"}} for b64 in frames_b64]



def build_message_frames(prompt_text, video_path=None, system_prompt=None,  num_frames=None):
    if num_frames is None:
        logger.error("num_frames is required.Should not be None.")
        raise ValueError("num_frames is required.Should not be None.")
    message = []
    # if system_prompt:
    #     message.append({"role": "system", "content": system_prompt})
   
    if video_path:
        image_parts = frames_to_image_parts(sample_frames_to_b64(video_path, num_frames))
        message.append({"role": "user", "content": [ {"type": "text", "text": prompt_text}, *image_parts]})
    else:
        image_parts = []
        message.append({"role": "user", "content": [{"type": "text", "text": prompt_text}]})
    
    return message


def inference_w_frames(model_provider, api_key, api_base, prompt_text, video_path=None, system_prompt=None, num_frames=None):
    """
    Runs inference by using only the frames of the video (+ text optionally). No audio is used.
    """
    if num_frames is None:
        num_frames = MAX_FRAMES
    message1 = build_message_frames(prompt_text, video_path, system_prompt, num_frames=num_frames)
    message2 = [{"role": "user", "content": "Hello, how are you?"}]

    # litellm._turn_on_debug()
    # litellm.completion(model=model_provider,messages=message1,api_key=api_key, base_url=api_base)
    
    breakpoint()
    # print(message1)
    # litellm.completion(model=model_provider,messages=message, api_key=api_key, base_url=api_base, reasoning_effort="low", thinking_level="low")
    try:
        resp = litellm.completion(
            model=model_provider,
            messages=message1,
            api_key=api_key,
            base_url=api_base,
            reasoning_effort="low",
            thinking_level="low",
        )
        if hasattr(resp, "choices") and resp.choices:
            return resp.choices[0].message.get("content", "")
        return getattr(resp, "content", "") or ""
    except Exception as e:
        import traceback
        resp = getattr(e, "response", None)
        status = getattr(resp, "status_code", None) if resp else None
        body = None
        try:
            body = resp.text if resp else None
        except Exception:
            pass

        logger.error(f"Inference error: {e}")
        if status is not None:
            logger.error(f"HTTP status: {status}")
        if body:
            logger.error(f"Response body: {body}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None






def build_message_audio(prompt_text, video_path=None, system_prompt=None):

    message = []
    if system_prompt:
        message.append({"role": "system", "content": system_prompt})
   
    if video_path:
        video_bytes = Path(video_path).read_bytes()
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        message.append({"role": "user", "content": [
            {"type": "file", "file": {"file_data": f"data:video/mp4;base64,{video_base64}"}},
            {"type": "text", "text": prompt_text}
            ]})
    else:
        message.append({"role": "user", "content": [{"type": "text", "text": prompt_text}]})
    
    return message
 


def inference_w_audio(model_provider, api_key, api_base, prompt_text, video_path=None, system_prompt=None, num_frames=None):
    """
    Runs inference by using the audio and frames of the video (+ text optionally).
    """
    message = build_message_audio(prompt_text, video_path, system_prompt)
    messages=[{ "content": "Hello, how are you?","role": "user"}]

    
    model_provider1="openrouter/openai/gpt-5"
    model_provider2="openrouter/google/gemini-3.5-flash"
    # litellm._turn_on_debug()
    # litellm.completion(model=model1,messages=messages,api_key=api_key, base_url=api_base)
    
    breakpoint()
    # print(message1)
    # litellm.completion(model=model_provider,messages=message, api_key=api_key, base_url=api_base, reasoning_effort="low", thinking_level="low")
    try:
        resp = litellm.completion(
            model=model_provider,
            messages=message,
            api_key=api_key,
            base_url=api_base,
            reasoning_effort="low",
            thinking_level="low",
        )
        if hasattr(resp, "choices") and resp.choices:
            return resp.choices[0].message.get("content", "")
        return getattr(resp, "content", "") or ""
    except Exception as e:
        import traceback
        resp = getattr(e, "response", None)
        status = getattr(resp, "status_code", None) if resp else None
        body = None
        try:
            body = resp.text if resp else None
        except Exception:
            pass

        logger.error(f"Inference error: {e}")
        if status is not None:
            logger.error(f"HTTP status: {status}")
        if body:
            logger.error(f"Response body: {body}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return ""

