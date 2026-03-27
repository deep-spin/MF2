import torch
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoConfig, AutoModel, AutoTokenizer


from tqdm import tqdm
import numpy as np

MAX_NUM_FRAMES = 16  # default max frames per video

def encode_video(video_path, max_frames=MAX_NUM_FRAMES, shuffle_frames=False):
    """
    Extract frames from video and return as list of PIL Images.
    """
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)
    frame_idx = [i for i in range(0, len(vr), sample_fps)]

    if len(frame_idx) > max_frames:
        frame_idx = uniform_sample(frame_idx, max_frames)

    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]

    if shuffle_frames:
        np.random.shuffle(frames)
    return frames


def mplug_owl3_inference(
    args,
    input_prompts,
    video_path=None,
    system_prompt=None,
    shuffle_frames=False
):
    """
    Inference function for mPLUG-Owl3 (video-text model) following GEMMA3 pattern.

    Args:
        args: Object with args.model (model path or ID)
        input_prompts: List of user prompts (text)
        video_paths: Optional list of video paths
        system_prompt: Optional system prompt text
        shuffle_frames: Whether to shuffle video frames
    Returns:
        List of generated text responses
    """
    # Load model, tokenizer, and processor


    # model = AutoModel.from_pretrained(
    #     args.model,
    #     attn_implementation='flash_attention_2',
    #     torch_dtype=torch.bfloat16,
    #     trust_remote_code=True
    # ).eval().cuda()

    model = AutoModel.from_pretrained(args.model, attn_implementation='sdpa', torch_dtype=torch.half, trust_remote_code=True)
    model.eval().cuda()

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    processor = model.init_processor(tokenizer)

    # Load video frames if provided
    video_frames_list = None
    if video_path:
        video_frames_list = encode_video(video_path, max_frames=MAX_NUM_FRAMES, shuffle_frames=shuffle_frames)

    
    responses = []
    for input_prompt in tqdm(input_prompts):
        # Construct messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_message = {"role": "user", "content": ""}
        if video_frames_list:
            # Multi-video support: include all frames
            user_message["content"] = "<|image|>"*len(video_frames_list) + input_prompt
        else:
            user_message["content"] = input_prompt
        messages.append(user_message)

        # Prepare inputs for the model
        if video_frames_list:
            inputs = processor(messages, images=video_frames_list, videos=None)
        else:
            inputs = processor(messages, images=None, videos=None)

        inputs.to(device)
        inputs.update({
            'tokenizer': tokenizer,
            'max_new_tokens': 1024,
            'decode_text': True
        })

        # Generate response
        with torch.inference_mode():
            output = model.generate(**inputs)
        responses.append(output[0])

    return responses