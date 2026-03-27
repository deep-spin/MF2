import os
from openai import OpenAI




def qwen_omni_model_inference(client,model_provider, prompt_text, video_path=None, system_prompt=None,stream=True):
    message = []
    if system_prompt:
        message.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    else:
        message.append({"role": "system", "content": [{"type": "text", "text": "You are Qwen-Omni, a smart voice assistant created by Alibaba Qwen."}]})
   
    if video_path:
        message.append({"role": "user", "content": [{"type": "video", "video": video_path},{"type": "text", "text": prompt_text}]})
    else:
        message.append({"role": "user", "content": [{"type": "text", "text": prompt_text}]})


    completion = client.chat.completions.create(
        model=model_provider,
        
        messages=message,
        # Set the modality for the output data. The following modalities are supported: ["text","audio"]、["text"]
        modalities=["text"],
        # The stream parameter must be set to True. Otherwise, an error is reported
        stream=stream,
        stream_options={"include_usage": True},
    )

    for chunk in completion:
        if chunk.choices:
            print(chunk.choices[0].delta)
        else:
            print(chunk.usage)