from models.closed_source.claude import litellm_model_inference as claude_model_inference_litellm

#gpt
from models.closed_source.gpt import litellm_model_inference as gpt4o_model_inference_litellm

#gemini 2.5 
from models.closed_source.gemini25 import inference_frames as inference_w_frames_gemini25

#gemini3 and gpt-5
from models.closed_source.inference_new import inference_w_frames, inference_w_video

MODELS_MAPPING = {
    'claude-3.7-sonnet': claude_model_inference_litellm,
    'gpt-4o': gpt4o_model_inference_litellm,
    'gemini-2.5-pro': inference_w_frames_gemini25,
    # 'gemini-2.5-pro': inference_w_frames,

    'gpt-5': inference_w_frames,
    'gemini-3-pro-w-frames': inference_w_frames,
    # 'gemini-3-pro-w-video': inference_w_video,  # this is not possible to use for very long videos. Uploading the video using openai-client is not possible.
}

