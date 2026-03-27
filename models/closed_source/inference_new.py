import base64
import cv2
from loguru import logger
import litellm
from pathlib import Path

MAX_FRAMES = 120
FPS = 1.0
# --- Helpers ---

def sample_frames_to_b64(video_path, num_frames):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        print(f"Cannot read video: {video_path}")
        return frames

    step = max(total // num_frames, 1)
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        ok, buf = cv2.imencode(".jpg", frame)
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


# def build_message_frames(prompt_text, video_path=None, system_prompt=None, num_frames=MAX_FRAMES):
#     """
#     Build message with video frames first, then text
#     """
#     message = []

#     if system_prompt:
#         message.append({
#             "role": "system",
#             "content": [{"type": "text", "text": system_prompt}]
#         })

#     if video_path:
#         frames_b64 = sample_frames_to_b64(video_path, num_frames)
#         image_parts = frames_to_image_parts(frames_b64)
#         message.append({
#             "role": "user",
#             "content": image_parts + [{"type": "text", "text": prompt_text}] 
#         })
#     else:
#         message.append({
#             "role": "user",
#             "content": [{"type": "text", "text": prompt_text}]
#         })

#     return message

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

# --- Main inference function ---

def inference_w_frames(
    model_provider,
    api_key,
    api_base,
    prompt_text,
    video_path=None,
    cached_image_parts=None,
    system_prompt=None,
    shuffle_frames=False,
    num_frames=None,
    reasoning_effort="minimal",
    thinking_level="low",
    max_retries=3,
    retry_delay=1.0
):
    """
    Send video frames + prompt to Gemini 3. No audio is used.
    """
    if not num_frames:
        num_frames = MAX_FRAMES
    
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

    for attempt in range(1, max_retries + 1):
        try:
            resp = litellm.completion(model=model_provider, messages=messages,api_key=api_key, base_url=api_base, reasoning_effort=reasoning_effort, thinking_level=thinking_level)

            # Extract text content
            if hasattr(resp, "choices") and resp.choices:
                return resp.choices[0].message.get("content", "")
            return getattr(resp, "content", "") or ""

        except litellm.APIError as e:
            # Retry only on server errors (500)
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 500:
                logger.error(f"Server error on attempt {attempt}, retrying in {retry_delay}s...")
                import time
                time.sleep(retry_delay)
                continue
            else:
                logger.error(f"API error: {e}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

    logger.error("Max retries reached without success.")
    return None

    

def build_message_video(prompt_text, video_path=None, system_prompt=None, num_frames=MAX_FRAMES):
    """
    Build message with video first, then text
    """

    message = []

    if system_prompt:
        message.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        })

    if video_path:
        video_bytes = Path(video_path).read_bytes()
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        message.append({
            "role": "user",
            "content": [{"type": "file", "file": {"file_data": f"data:video/mp4;base64,{video_base64}"}}, {"type": "text", "text": prompt_text}]
        })
    else:
        message.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}]
        })

    return message


def build_message_video(prompt_text, file_id=None, system_prompt=None, fps=None):
    message = []

    if system_prompt:
        message.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        })

    if file_id:
        message.append({
            "role": "user",
            "content": [{"type": "file", "file_url": f"file://{file_id}"}, {"type": "text", "text": prompt_text}]
        })
    else:
        message.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}]
        })

    return message

def inference_w_video(
    model_provider,
    api_key,
    api_base,
    prompt_text,
    video_path,
    system_prompt=None,
    max_retries=3,
    retry_delay=1.0
):
    """
    Send full video + audio to Gemini 3 via OpenRouter (file-based).
    """

    from openai import OpenAI
    import time

    # --- Create client ---
    try:
        client = OpenAI(api_key=api_key, base_url=api_base)
    except Exception as e:
        print(f"Error creating OpenAI client: {e}")
        return None

    # --- Upload video file ---
    if video_path:
        try:
            with open(video_path, "rb") as f:
                file_obj = client.files.create(
                    file=f,
                    purpose="assistants"
                )
            file_id = file_obj.id
        except Exception as e:
            print(f"Error uploading video: {e}")
            return None
    else:
        file_id = None

    messages = build_message_video(prompt_text, file_id=file_id, system_prompt=system_prompt, fps=FPS)
    breakpoint()
    # --- Retry loop ---
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_provider,
                messages=messages
            )

            return resp.choices[0].message.content

        except Exception as e:
            print(f"Error on attempt {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                return None