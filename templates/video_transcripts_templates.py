## System prompts
default_system_prompt = """You are a helpful AI assistant. Your task is to carefully analyze the provided content and determine whether statements made about it are true or false based on the available information."""
video_transcripts_template_system_prompt = """You are a helpful AI assistant. Your task is to carefully analyze the provided video content and the corresponding transcript, and determine whether statements made about it are true or false based on the available information."""


## User prompts
video_transcripts_template_explanation_user_prompt = """You are provided with a movie, the corresponding transcript, and a statement. Your task is to carefully watch the movie, read the transcript, and then determine whether the statement is true or false.
Answer TRUE if the statement is true in its entirety based on the movie and the transcript.
Answer FALSE if any part of the statement is false based on the movie and the transcript.

Transcript: {transcripts}
Statement: {claim}
Based on the movie and the transcript, is the above statement TRUE or FALSE?
First provide an explanation of your decision-making process in at most one paragraph, and then provide your final answer. Use the following format:
<explanation>YOUR EXPLANATION</explanation>
<answer>YOUR ANSWER</answer>"""

video_transcripts_template_explanation_free_user_prompt = """You are provided with a movie, the corresponding transcript, and a statement. Your task is to carefully watch the movie, read the transcript, and then determine whether the statement is true or false.
Answer TRUE if the statement is true in its entirety based on the movie and the transcript.
Answer FALSE if any part of the statement is false based on the movie and the transcript.

Transcript: {transcripts}
Statement: {claim}
Based on the movie and the transcript, is the above statement TRUE or FALSE?
First provide an explanation of your decision-making process in at most one paragraph, and then provide your final answer."""

video_transcripts_template_direct_user_prompt = """You are provided with a movie, the corresponding transcript, and a statement. Your task is to carefully watch the movie, read the transcript, and then determine whether the statement is true or false.
Answer TRUE if the statement is true in its entirety based on the movie and the transcript.
Answer FALSE if any part of the statement is false based on the movie and the transcript.

Transcript: {transcripts}
Statement: {claim}
Based on the movie and the transcript, is the above statement TRUE or FALSE?
Provide only your final answer. Use the following format:
<answer>YOUR ANSWER</answer>"""

video_transcripts_template_direct_free_user_prompt = """You are provided with a movie, the corresponding transcript, and a statement. Your task is to carefully watch the movie, read the transcript, and then determine whether the statement is true or false.
Answer TRUE if the statement is true in its entirety based on the movie and the transcript.
Answer FALSE if any part of the statement is false based on the movie and the transcript.

Transcript: {transcripts}
Statement: {claim}
Based on the movie and the transcript, is the above statement TRUE or FALSE?
Provide only your final answer."""

video_transcripts_template_direct_user_prompt_confidence = """
You are provided with a movie, the corresponding transcript, and a statement. 
Your task is to carefully watch the movie, read the transcript, and then determine whether the statement is true or false.

Answer TRUE if the statement is true in its entirety based on the movie and the transcript.
Answer FALSE if any part of the statement is false based on the movie and the transcript.

Transcript: {transcripts}
Statement: {claim}

Based on the movie and the transcript, is the above statement TRUE or FALSE?

Provide your final answer and your confidence score (0–100), using the following XML-style tags:
<answer>FINAL_ANSWER_HERE</answer>
<confidence>CONFIDENCE_SCORE_HERE</confidence>
"""

video_transcripts_template_direct_multiple_choice = """
You are provided with a movie, its transcript, and two candidate statements. 
Your task is to carefully watch the movie, read the transcript, and determine 
which one of the two statements is TRUE.

Exactly one of the statements is true, and the other is false.

Transcript:
{transcripts}

Statements:
A. {claim_a}
B. {claim_b}

Question:
Based on the movie and the transcript, which statement (A or B) is TRUE?
Provide only the letter of the true statement (A or B) as follows:
Answer: <answer>LETTER_OF_THE_TRUE_STATEMENT_HERE</answer>
"""

USER_PROMPTS_TEMPLATE_DICT = {
    "explanation":  video_transcripts_template_explanation_user_prompt,
    "explanation_free":  video_transcripts_template_explanation_free_user_prompt,
    "direct": video_transcripts_template_direct_user_prompt,
    "direct_free": video_transcripts_template_direct_free_user_prompt,
    "direct_multiple_choice": video_transcripts_template_direct_multiple_choice,
    "direct_confidence": video_transcripts_template_direct_user_prompt_confidence,
    # add more templates here if needed, e.g. model specific templates, or custom templates
    #"llavaonevision": video_template_llavaonevision_user_prompt    
}


SYSTEM_PROMPTS_TEMPLATE_DICT = {
    "default": default_system_prompt,
    "video_transcripts": video_transcripts_template_system_prompt,
    # add more templates here if needed, e.g. model specific templates, or custom templates
    #"llavaonevision": video_template_llavaonevision_system_prompt
}

 
