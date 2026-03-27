from transformers import  Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
# from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import soundfile as sf
from qwen_omni_utils import process_mm_info
from tqdm import tqdm
import torch



def qwen_omni_inference_transformers(args, input_prompts, video_path=None, system_prompt=None, shuffle_frames=False, use_audio=False):
    """Inference helper for Qwen2.5-Omni models with optional video and audio support."""
    # Load model with flash_attention_2 as recommended
    breakpoint()
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model,
        # device_map="auto",
        # torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).eval()
    breakpoint()
    model.disable_talker()
    breakpoint()    
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model)
    
    responses = []
    
    for input_prompt in tqdm(input_prompts):
        # Build conversation
        conversation = []
        
        # Add system prompt if provided
        if system_prompt:
            conversation.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        # Build user content
        user_content = []
        if video_path:
            user_content.append({
                "type": "video",
                "video": video_path
            })
        user_content.append({"type": "text", "text": input_prompt})
        
        conversation.append({
            "role": "user",
            "content": user_content
        })
        
        # Preparation for inference
        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Process multimodal info (audio, images, videos)
        audios, images, videos = process_mm_info(
            conversation,
            use_audio_in_video=use_audio
        )
        
        # Prepare inputs
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio
        )
        inputs = inputs.to(model.device).to(model.dtype)
        
        # Generate output (text and audio)
        text_ids, audio = model.generate(
            **inputs,
            use_audio_in_video=use_audio
        )
        
        # Decode text response
        decoded_text = processor.batch_decode(
            text_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        # For now, we only return text responses
        # Audio output is available in the 'audio' variable if needed
        responses.append(decoded_text[0] if decoded_text else "")
    
    return responses



