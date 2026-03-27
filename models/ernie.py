from decord import VideoReader, cpu
from tqdm import tqdm
import torch
# from transformers import AutoProcessor, AutoModelForCausalLM
from loguru import logger

MAX_FRAMES = 64

# def ernie_vl_inference(args, input_prompts, video_path=None, system_prompt=None, shuffle_frames=False):
#     model_path = 'baidu/ERNIE-4.5-VL-28B-A3B-Thinking'
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         device_map="auto",
#         dtype=torch.bfloat16,
#         trust_remote_code=True
#     )
#     processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
#     model.add_image_preprocess(processor)

#     responses = []
#     for input_prompt in tqdm(input_prompts, desc="Inference..."):
#         messages = []
#         if system_prompt:
#             messages.append({"role": "system", "content": system_prompt})
#         if video_path:
#             messages.append({
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": input_prompt},
#                     {"type": "video_url", "video_url": {"url": video_path}},
#                 ]
#             })
#         else:
#             messages.append({
#                 "role": "user",
#                 "content": [{"type": "text", "text": input_prompt}]
#             })


#         text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         image_inputs, video_inputs = processor.process_vision_info(messages)
#         breakpoint()
#         inputs = processor(
#             text=[text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",)
#         device = next(model.parameters()).device
#         inputs = inputs.to(device)

#         try:
#             generated_ids = model.generate(**inputs,max_new_tokens=1024,use_cache=False)
#             output_text = processor.decode(generated_ids[0][len(inputs['input_ids'][0]):])
#         except Exception as e:
#             logger.error(f"Generation failed: {str(e)}")
#             output_text = None
#         breakpoint()
#         responses.append(output_text)
#     return responses

