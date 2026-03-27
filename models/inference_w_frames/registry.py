# from models.videollama3 import videollama3_inference
# from models.qwenvl import qwenvl_inference
# from models.gemma3 import gemma3_inference
# from models.qwenvl3 import qwenvl3_inference_vllm
# from models.llavavideo import llavavideo_inference
# from models.internvl import internvl3_inference
# from models.ovis import ovis_inference
# from models.kangaroo import kangaroo_inference
# from models.cogvlm2 import cogvlm2_inference
# from models.llava_one_vision import llavaonevision_inference
# from models.gemma3 import gemma3_inference
# from models.qwen3_omni_vllm import  qwen_omni_inference_vllm
# from models.qwen3_omni_vllm import  qwen_omni_inference_vllm_with_audio
# from models.qwen3_omni_transformers import  qwen_omni_inference_transformers


# from models.aria import aria_inference
from models.mplugowl import mplug_owl3_inference
from models.mimovl import mimovl_inference
from models.minicpm import minicpm_inference
from models.vllm_served_model import inference_vllm_served_model

from models.internvl3_global import internvl3_inference
from models.gemma3_global import gemma3_inference
from models.qwenvl import qwenvl_inference
from models.qwenvl3 import qwenvl3_inference_vllm

MODELS_MAPPING = {
    # 'rhymes-ai/Aria': aria_inference, #done!
    'XiaomiMiMo/MiMo-VL-7B-RL': mimovl_inference, #running on poseidon with MAX_FRAMES=180 frames
    'openbmb/MiniCPM-V-4_5': minicpm_inference, #running with MAX_NUM_FRAMES=64 and MAX_NUM_PACKING=3 for MiniCPM-V-4_5
    'mPLUG/mPLUG-Owl3-7B-240728': mplug_owl3_inference, # done - produces empty outputs
    'ernie-4.5-vl': inference_vllm_served_model,
    'glm-4.5v-fp8': inference_vllm_served_model, #To run  on dionysus with 2 GPUs / hades with 2 GPUs
    'glm-4.5v': inference_vllm_served_model,

    'Qwen/Qwen2.5-VL-72B-Instruct': qwenvl_inference,
    'Qwen/Qwen3-VL-30B-A3B-Instruct': qwenvl3_inference_vllm,
    'OpenGVLab/InternVL3-78B': internvl3_inference,
    'OpenGVLab/InternVL3_5-38B-Instruct': internvl3_inference,
    'google/gemma-3-27b-it': gemma3_inference,
    # 'KangarooGroup/kangaroo': kangaroo_inference, # done - not following the prompt


    # 'Qwen/Qwen2.5-VL-7B-Instruct': qwenvl_inference,
    # 'OpenGVLab/InternVL3-8B': internvl3_inference,
    # 'OpenGVLab/InternVL3_5-8B': internvl3_inference,
    # 'OpenGVLab/InternVL3_5-38B': internvl3_inference,
    # 'OpenGVLab/InternVL3_5-8B-Instruct': internvl3_inference,
    # 'Qwen/Qwen2.5-Omni-7B': qwen_omni_inference_transformers,
    # 'Qwen/Qwen3-Omni-30B-A3B-Instruct': qwen_omni_inference_vllm,
    # '/mnt/scratch-artemis/manos/qwen3_omni-30b-it': qwen_omni_inference_vllm,
    # '/mnt/scratch-artemis/manos/qwen3-omni-30b-it-4bit': qwen_omni_inference_vllm,
    # '/mnt/scratch-artemis/manos/qwen3-omni-30b-it-4bit': qwen_omni_inference_vllm_with_audio,

    # 'DAMO-NLP-SG/VideoLLaMA3-7B': videollama3_inference,
    # 'Qwen/Qwen2.5-VL-3B-Instruct': qwenvl_inference,
    # 'Qwen/Qwen2.5-VL-7B-Instruct': qwenvl_inference,
    # 'Qwen/Qwen2.5-VL-32B-Instruct': qwenvl_inference,       
    # 'Qwen/Qwen2.5-VL-72B-Instruct': qwenvl_inference,
    # 'lmms-lab/LLaVA-Video-72B-Qwen2' : llavavideo_inference,
    # 'lmms-lab/LLaVA-Video-72B-Qwen2': llavavideo_inference,
    # 'OpenGVLab/InternVL3-1B': internvl3_inference,
    # 'OpenGVLab/InternVL3-2B': internvl3_inference,
    # 'OpenGVLab/InternVL3-8B': internvl3_inference,
    # 'OpenGVLab/InternVL3-9B': internvl3_inference,
    # 'OpenGVLab/InternVL3-14B': internvl3_inference,
    # 'OpenGVLab/InternVL3-38B': internvl3_inference,
    # 'OpenGVLab/InternVL3-78B': internvl3_inference,
    # 'OpenGVLab/InternVL3_5-8B': internvl3_inference,
    # 'OpenGVLab/InternVL3_5-30B': internvl3_inference,
    # 'Qwen/Qwen3-VL-8B-Instruct': qwenvl_inference,
    # 'KangarooGroup/kangaroo': kangaroo_inference,
    # 'AIDC-AI/Ovis2-1B': ovis_inference,
    # 'AIDC-AI/Ovis2-2B': ovis_inference,
    # 'AIDC-AI/Ovis2-4B': ovis_inference,
    # 'AIDC-AI/Ovis2-8B': ovis_inference,
    # 'AIDC-AI/Ovis2-16B': ovis_inference,
    # 'AIDC-AI/Ovis2-34B': ovis_inference,
    
    # 'zai-org/cogvlm2-video-llama3-chat': cogvlm2_inference,
    # 'lmms-lab/LLaVA-One-Vision-1.5-8B-Instruct': llavaonevision_inference,
}