
## System prompts
default_system_prompt = """You are a helpful AI assistant. Your task is to carefully analyze the provided content and determine whether statements made about it are true or false based on any information."""
statement_only_template_system_prompt = """You are a helpful AI assistant. Your task is to carefully analyze the provided statement, and determine whether it is true or false based on any information."""

## User prompts
statement_only_template_explanation_user_prompt = """You are provided with a movie title, and a statement. Your task is to carefully read the statement, and then determine whether it is true or false based on any information regarding the movie.

Movie Title: {movie_title}
Statement: {claim}
Based on any information regarding the movie, is the above statement TRUE or FALSE?
First provide an explanation of your decision-making process in at most one paragraph, and then provide your final answer. Use the following format:
<explanation>YOUR EXPLANATION</explanation>
<answer>YOUR ANSWER</answer>"""

statement_only_template_explanation_free_user_prompt = """You are provided with a movie title, and a statement. Your task is to carefully read the statement, and then determine whether it is true or false based on any information regarding the movie.

Movie Title: {movie_title}
Statement: {claim}
Based on any information regarding the movie, is the above statement TRUE or FALSE?
First provide an explanation of your decision-making process in at most one paragraph, and then provide your final answer."""

statement_only_template_direct_user_prompt = """You are provided with a movie title, and a statement. Your task is to carefully read the statement, and then determine whether it is true or false based on any information regarding the movie.

Movie Title: {movie_title}
Statement: {claim}
Based on any information regarding the movie, is the above statement TRUE or FALSE?
Provide only your final answer. Use the following format:
<answer>YOUR ANSWER</answer>"""

statement_only_template_direct_free_user_prompt = """You are provided with a movie title, and a statement. Your task is to carefully read the statement, and then determine whether it is true or false based on any information regarding the movie.

Movie Title: {movie_title}
Statement: {claim}
Based on any information regarding the movie, is the above statement TRUE or FALSE?
Provide only your final answer."""


USER_PROMPTS_TEMPLATE_DICT = {
    "explanation":  statement_only_template_explanation_user_prompt,
    "explanation_free":  statement_only_template_explanation_free_user_prompt,
    "direct": statement_only_template_direct_user_prompt,
    "direct_free": statement_only_template_direct_free_user_prompt,
    # add more templates here if needed, e.g. model specific templates, or custom templates
    #"llavaonevision": video_template_llavaonevision_user_prompt    
}


SYSTEM_PROMPTS_TEMPLATE_DICT = {
    "default": default_system_prompt,
    "statement_only": statement_only_template_system_prompt,
    # add more templates here if needed, e.g. model specific templates, or custom templates
    #"llavaonevision": video_template_llavaonevision_system_prompt
}