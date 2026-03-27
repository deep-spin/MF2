from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch


def kangaroo_inference(args, input_prompts, video_path=None, system_prompt=None, shuffle_frames=False):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to("cuda")
    
    # Set up terminators for EOS tokens
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    
    responses = []
    for input_prompt in tqdm(input_prompts):
        # Prepare query - prepend system prompt if provided
        if system_prompt:
            query = f"{system_prompt}\n\n{input_prompt}"
        else:
            query = input_prompt
        
        # Call model.chat() - each prompt is independent (no shared history between prompts)
        out, history = model.chat(
            video_path=video_path,
            query=query,
            tokenizer=tokenizer,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        
        responses.append(out)
    
    return responses