from models.videollama3 import videollama3_inference
from models.qwenvl import qwenvl_inference
from models.llavavideo import llavavideo_inference
from models.internvl import internvl3_inference
from models.ovis import ovis_inference

MODELS_MAPPING = {
    'DAMO-NLP-SG/VideoLLaMA3-7B': videollama3_inference,
    'Qwen/Qwen2.5-VL-3B-Instruct': qwenvl_inference,
    'Qwen/Qwen2.5-VL-7B-Instruct': qwenvl_inference,
    'Qwen/Qwen2.5-VL-32B-Instruct': qwenvl_inference,       
    'Qwen/Qwen2.5-VL-72B-Instruct': qwenvl_inference,
    'lmms-lab/LLaVA-Video-7B-Qwen2' : llavavideo_inference,
    'lmms-lab/LLaVA-Video-72B-Qwen2': llavavideo_inference,
    'OpenGVLab/InternVL3-1B': internvl3_inference,
    'OpenGVLab/InternVL3-2B': internvl3_inference,
    'OpenGVLab/InternVL3-8B': internvl3_inference,
    'OpenGVLab/InternVL3-9B': internvl3_inference,
    'OpenGVLab/InternVL3-14B': internvl3_inference,
    'OpenGVLab/InternVL3-38B': internvl3_inference,
    'OpenGVLab/InternVL3-78B': internvl3_inference,
    'AIDC-AI/Ovis2-1B': ovis_inference,
    'AIDC-AI/Ovis2-2B': ovis_inference,
    'AIDC-AI/Ovis2-4B': ovis_inference,
    'AIDC-AI/Ovis2-8B': ovis_inference,
    'AIDC-AI/Ovis2-16B': ovis_inference,
    'AIDC-AI/Ovis2-34B': ovis_inference,
}