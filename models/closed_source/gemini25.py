
import cv2, base64, litellm
from loguru import logger
import os

# litellm._turn_on_debug()

MAX_FRAMES = 120
FPS = 1.0


def sample_frames_to_b64(video_path, num_frames):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        logger.error(f"Cannot read video: {video_path}")
        return frames

    step = max(total // num_frames, 1)
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        # ok, buf = cv2.imencode(".jpg", frame)
        # if not ok:
        #     continue

        # remove the following code to use the original frame
        # 🔴 Downscale frame
        frame = cv2.resize(frame, (400, 400))  # smaller than 256x256
        # 🔴 Reduce JPEG quality
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        ###############
        if not ok:
            continue
        frames.append(base64.b64encode(buf).decode("utf-8"))
        if len(frames) >= num_frames:
            break
    cap.release()
    return frames

def frames_to_image_parts(frames_b64):
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for b64 in frames_b64
    ]

def build_message_frames_cached(prompt_text, cached_image_parts, system_prompt=None):
    """
    Build a message for inference using pre-encoded video frames (cached).

    Args:
        prompt_text (str): The claim / user prompt.
        cached_image_parts (List[dict]): Precomputed video frames with cache_control.
        system_prompt (str, optional): System-level instruction.
        num_frames (int, optional): Not used here, included for signature compatibility.

    Returns:
        List[dict]: Messages ready for litellm.completion
    """
    if cached_image_parts is None:
        logger.error("cached_image_parts is None. Please provide cached_image_parts.")
        raise ValueError("cached_image_parts is None. Please provide cached_image_parts.")
    messages = []

    # 1️⃣ Add system prompt if provided
    if system_prompt:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        })

    # 2️⃣ Add cached video frames (already includes cache_control)
    if cached_image_parts:
        messages.append({
            "role": "user",
            "content": cached_image_parts
        })

    # 3️⃣ Add claim / prompt text
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": prompt_text}]
    })

    return messages


def build_message_frames(prompt_text, video_path=None, system_prompt=None, num_frames=None):
    if num_frames is None:
        logger.error("num_frames is required.Should not be None.")
        raise ValueError("num_frames is required.Should not be None.")
    messages = []

    # 1️⃣ System prompt
    if system_prompt:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        })

    # 2️⃣ Video frames
    if video_path:
        frames_b64 = sample_frames_to_b64(video_path, num_frames)
        image_parts = frames_to_image_parts(frames_b64)
        messages.append({
            "role": "user",
            "content": image_parts 
        })

    # 3️⃣ Prompt text
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": prompt_text}]
    })

    return messages

def build_message_frames_v2(prompt_text, video_path=None, system_prompt=None, num_frames=None):
    if num_frames is None:
        raise ValueError("num_frames is required")

    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        })

    image_parts = []
    if video_path:
        frames_b64 = sample_frames_to_b64(video_path, num_frames)
        image_parts = frames_to_image_parts(frames_b64)
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}] + image_parts
        })
    else:
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}]
        })
    return messages

def inference_frames(
    model_provider,
    api_key,
    api_base,
    prompt_text,
    video_path=None,
    cached_image_parts=None,
    system_prompt=None,
    shuffle_frames=False,
    num_frames=None,
    max_retries=3,
    retry_delay=1.0
):
    """
    Send video frames + prompt to Gemini 3. No audio is used.
    """
    if num_frames is None:
        logger.info(f"Setting num_frames to {MAX_FRAMES}, as num_frames is not provided.")
        num_frames = MAX_FRAMES
    else:
        logger.info(f"num_frames: {num_frames}")
    
    if not isinstance(prompt_text, str) and prompt_text is not None:
        logger.error(f"prompt_text must be a string. Got {type(prompt_text)}.")
        raise ValueError(f"prompt_text must be a string. Got {type(prompt_text)}.")
    
    if shuffle_frames:
        logger.error("Shuffle frames is not implemented yet.")
        raise NotImplementedError("Shuffle frames is not implemented yet.")

    if cached_image_parts is not None:
        messages = build_message_frames_cached(prompt_text, cached_image_parts, system_prompt)
    else:
        messages = build_message_frames(prompt_text, video_path, system_prompt, num_frames=num_frames)
    
    try:
        resp = litellm.completion(
            model=model_provider,
            messages=messages,
            api_key=api_key,
            base_url=api_base,
        )

        if hasattr(resp, "choices") and resp.choices:
            return resp.choices[0].message.get("content", "")

        return getattr(resp, "content", "") or ""

    except litellm.APIError as e:
        logger.error(f"LiteLLM API error: {e}")
        raise  # re-raise so the caller knows

    except Exception as e:
        logger.error(f"Unexpected error during LiteLLM completion. Exception: {e}")
        raise

    # ========================= Ingore =========================
    # resp = litellm.completion(model=model_provider, messages=messages, api_key=api_key, base_url=api_base)
    # # Extract text content
    # if hasattr(resp, "choices") and resp.choices:
    #     return resp.choices[0].message.get("content", "")
    # return getattr(resp, "content", "") or ""


    # ========================= Ingore =========================

        # except litellm.APIError as e:
        #     # Retry only on server errors (500)
        #     status = getattr(getattr(e, "response", None), "status_code", None)
        #     if status == 500:
        #         logger.error(f"Server error on attempt {attempt}, retrying in {retry_delay}s...")
        #         import time
        #         time.sleep(retry_delay)
        #         continue
        #     else:
        #         logger.error(f"API error: {e}")
        #         return None
        # except Exception as e:
        #     logger.error(f"Unexpected error: {e}")
        #     return None

        # except litellm.APIError as e:
        #     # 🔑 Try to recover response from the exception
        #     resp = getattr(e, "llm_response", None) or getattr(e, "response", None)

        #     if resp and hasattr(resp, "choices") and resp.choices:
        #         logger.warning("Recovered response from APIError.")
        #         return resp.choices[0].message.get("content", "")

        #     status = getattr(getattr(e, "response", None), "status_code", None)
        #     if status == 500:
        #         logger.error(
        #             f"Server error on attempt {attempt}, retrying in {retry_delay}s..."
        #         )
        #         import time
        #         time.sleep(retry_delay)
        #         continue
        #     else:
        #         logger.error(f"API error without recoverable response: {e}")
        #         return None
