
## System prompts
default_system_prompt = """You are a helpful AI assistant. Your task is to carefully analyze the provided content and determine whether statements made about it are true or false based on the available information."""
video_synopsis_template_system_prompt = """You are a helpful AI assistant. Your task is to carefully analyze the provided video content and the corresponding synopsis, and determine whether statements made about it are true or false based on the available information."""


## User prompts
video_synopsis_template_explanation_user_prompt = """You are provided with a movie, the corresponding synopsis, and a statement. Your task is to carefully watch the movie, read the synopsis, and then determine whether the statement is true or false.
Answer TRUE if the statement is true in its entirety based on the movie and the synopsis.
Answer FALSE if any part of the statement is false based on the movie and the synopsis.

Synopsis: {synopsis}
Statement: {claim}
Based on the movie and the synopsis, is the above statement TRUE or FALSE?
First provide an explanation of your decision-making process in at most one paragraph, and then provide your final answer. Use the following format:
<explanation>YOUR EXPLANATION</explanation>
<answer>YOUR ANSWER</answer>"""

video_synopsis_template_explanation_free_user_prompt = """You are provided with a movie, the corresponding synopsis, and a statement. Your task is to carefully watch the movie, read the synopsis, and then determine whether the statement is true or false.
Answer TRUE if the statement is true in its entirety based on the movie and the synopsis.
Answer FALSE if any part of the statement is false based on the movie and the synopsis.

Synopsis: {synopsis}
Statement: {claim}
Based on the movie and the synopsis, is the above statement TRUE or FALSE?
First provide an explanation of your decision-making process in at most one paragraph, and then provide your final answer."""

video_synopsis_template_direct_user_prompt = """You are provided with a movie, the corresponding synopsis, and a statement. Your task is to carefully watch the movie, read the synopsis, and then determine whether the statement is true or false.
Answer TRUE if the statement is true in its entirety based on the movie and the synopsis.
Answer FALSE if any part of the statement is false based on the movie and the synopsis.

Synopsis: {synopsis}
Statement: {claim}
Based on the movie and the synopsis, is the above statement TRUE or FALSE?
Provide only your final answer. Use the following format:
<answer>YOUR ANSWER</answer>"""

video_synopsis_template_direct_free_user_prompt = """You are provided with a movie, the corresponding synopsis, and a statement. Your task is to carefully watch the movie, read the synopsis, and then determine whether the statement is true or false.
Answer TRUE if the statement is true in its entirety based on the movie and the synopsis.
Answer FALSE if any part of the statement is false based on the movie and the synopsis.

Synopsis: {synopsis}
Statement: {claim}
Based on the movie and the synopsis, is the above statement TRUE or FALSE?
Provide only your final answer."""




USER_PROMPTS_TEMPLATE_DICT = {
    "explanation":  video_synopsis_template_explanation_user_prompt,
    "explanation_free":  video_synopsis_template_explanation_free_user_prompt,
    "direct": video_synopsis_template_direct_user_prompt,
    "direct_free": video_synopsis_template_direct_free_user_prompt,
    # add more templates here if needed, e.g. model specific templates, or custom templates
    #"llavaonevision": video_template_llavaonevision_user_prompt    
}


SYSTEM_PROMPTS_TEMPLATE_DICT = {
    "default": default_system_prompt,
    "video_synopsis": video_synopsis_template_system_prompt,
    # add more templates here if needed, e.g. model specific templates, or custom templates
    #"llavaonevision": video_template_llavaonevision_system_prompt
}

 
