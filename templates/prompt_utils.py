MOVIES_DICT = {
    1: {"title": "The last chance", "year": 1945},
    2: {"title": "They Made Me a Criminal", "year": 1939},
    3: {"title": "Tokyo After Dark", "year": 1959},
    4: {"title": "The Sadist", "year": 1963},
    5: {"title": "Suddenly", "year": 1954},
    6: {"title": "Sabotage (Hitchcock)", "year": 1936},
    7: {"title": "Murder By Contract", "year": 1958},
    8: {"title": "Pushover", "year": 1954},
    9: {"title": "Go for Broke", "year": 1951},
    10: {"title": "Meet John Doe", "year": 1941},
    11: {"title": "Scarlet Street", "year": 1945},
    12: {"title": "Little Lord Fauntleroy", "year": 1936},
    13: {"title": "Deadline - U.S.A.", "year": 1952},
    14: {"title": "My Favorite Brunette", "year": 1947},
    15: {"title": "Woman in the Moon", "year": 1929},
    16: {"title": "Lonely Wives", "year": 1931},
    17: {"title": "Nothing Sacred", "year": 1937},
    18: {"title": "Fingerman", "year": 1955},
    19: {"title": "Borderline", "year": 1950},
    20: {"title": "Babes in Toyland", "year": 1934},
    21: {"title": "The Man From Utah", "year": 1934},
    22: {"title": "The Man With The Golden Arm", "year": 1955},
    23: {"title": "A Star Is Born", "year": 1937},
    24: {"title": "Africa Screams", "year": 1949},
    25: {"title": "Dementia 13", "year": 1963},
    26: {"title": "Fear and Desire", "year": 1952},
    27: {"title": "The Little Princess", "year": 1939},
    28: {"title": "Father's Little Dividend", "year": 1951},
    29: {"title": "Kansas City Confidential", "year": 1952},
    30: {"title": "Of Human Bondage", "year": 1934},
    31: {"title": "Half Shot at Sunrise", "year": 1930},
    32: {"title": "Bowery at Midnight", "year": 1942},
    33: {"title": "The Emperor Jones", "year": 1933},
    34: {"title": "The Deadly Companions", "year": 1961},
    35: {"title": "The Red House", "year": 1947},
    36: {"title": "Trapped", "year": 1949},
    37: {"title": "City of Fear", "year": 1959},
    38: {"title": "Kid Monk Baroni", "year": 1952},
    39: {"title": "Tight Spot", "year": 1955},
    40: {"title": "Captain Kidd", "year": 1945},
    41: {"title": "Algiers", "year": 1938},
    42: {"title": "The Front Page", "year": 1931},
    43: {"title": "The Hitch-Hiker", "year": 1953},
    44: {"title": "Obsession", "year": 1949},
    45: {"title": "Thunderbolt", "year": 1929},
    46: {"title": "Cyrano de Bergerac", "year": 1950},
    47: {"title": "Scandal Sheet", "year": 1952},
    48: {"title": "Ladies in Retirement", "year": 1941},
    49: {"title": "Detour", "year": 1945},
    50: {"title": "The Crooked Way", "year": 1949},
    51: {"title": "A Bucket of Blood", "year": 1959},
    52: {"title": "Love Affair", "year": 1939},
    53: {"title": "The Jackie Robinson Story", "year": 1950},
    54: {"title": "The Last Time I Saw Paris", "year": 1954}
}


def build_prompts(claims, user_prompt_template_name, system_prompt_template_name=None, video_path=None, transcripts=None, synopsis=None, movie_id=None):
    if video_path is not None and transcripts is None and synopsis is None:
        from templates.video_only_templates import USER_PROMPTS_TEMPLATE_DICT,SYSTEM_PROMPTS_TEMPLATE_DICT
        user_prompt = USER_PROMPTS_TEMPLATE_DICT[user_prompt_template_name]
        user_prompts = [user_prompt.format(claim=claim) for claim in claims]
        if system_prompt_template_name is not None:
            system_prompt = SYSTEM_PROMPTS_TEMPLATE_DICT[system_prompt_template_name]
        else:
            system_prompt = None
    elif video_path is not None and transcripts is not None and synopsis is None:
        from templates.video_transcripts_templates import USER_PROMPTS_TEMPLATE_DICT,SYSTEM_PROMPTS_TEMPLATE_DICT
        user_prompt = USER_PROMPTS_TEMPLATE_DICT[user_prompt_template_name]
        user_prompts = [user_prompt.format(claim=claim, transcripts=transcripts) for claim in claims]
        if system_prompt_template_name is not None:
            system_prompt = SYSTEM_PROMPTS_TEMPLATE_DICT[system_prompt_template_name]
        else:
            system_prompt = None
    elif video_path is not None  and synopsis is not None and transcripts is None:
        from templates.video_synopsis_templates import USER_PROMPTS_TEMPLATE_DICT,SYSTEM_PROMPTS_TEMPLATE_DICT
        user_prompt = USER_PROMPTS_TEMPLATE_DICT[user_prompt_template_name]
        user_prompts = [user_prompt.format(claim=claim, synopsis=synopsis) for claim in claims]
        if system_prompt_template_name is not None:
            system_prompt = SYSTEM_PROMPTS_TEMPLATE_DICT[system_prompt_template_name]
        else:
            system_prompt = None
    elif video_path is None and transcripts is not None and synopsis is not None:
        # load templates for both transcripts and synopsis
        from templates.video_transcripts_synopsis_templates import USER_PROMPTS_TEMPLATE_DICT,SYSTEM_PROMPTS_TEMPLATE_DICT
        user_prompt = USER_PROMPTS_TEMPLATE_DICT[user_prompt_template_name]
        user_prompts = [user_prompt.format(claim=claim, transcripts=transcripts, synopsis=synopsis) for claim in claims]
        if system_prompt_template_name is not None:
            system_prompt = SYSTEM_PROMPTS_TEMPLATE_DICT[system_prompt_template_name]
        else:
            system_prompt = None
    elif video_path is None and transcripts is not None and synopsis is None:
        # load templates for transcripts only
        from templates.transcripts_only import USER_PROMPTS_TEMPLATE_DICT,SYSTEM_PROMPTS_TEMPLATE_DICT
        user_prompt = USER_PROMPTS_TEMPLATE_DICT[user_prompt_template_name]
        user_prompts = [user_prompt.format(claim=claim, transcripts=transcripts) for claim in claims]
        if system_prompt_template_name is not None:
            system_prompt = SYSTEM_PROMPTS_TEMPLATE_DICT[system_prompt_template_name]
        else:
            system_prompt = None
    elif video_path is None and synopsis is not None and transcripts is None:
        # load templates for synopsis only  
        from templates.synopsis_only import USER_PROMPTS_TEMPLATE_DICT,SYSTEM_PROMPTS_TEMPLATE_DICT
        user_prompt = USER_PROMPTS_TEMPLATE_DICT[user_prompt_template_name]
        user_prompts = [user_prompt.format(claim=claim, synopsis=synopsis) for claim in claims]
        if system_prompt_template_name is not None:
            system_prompt = SYSTEM_PROMPTS_TEMPLATE_DICT[system_prompt_template_name]
        else:
            system_prompt = None
    elif video_path is None and synopsis is  None and transcripts is None:
        # load templates for statement only 
        from templates.statement_only import USER_PROMPTS_TEMPLATE_DICT,SYSTEM_PROMPTS_TEMPLATE_DICT
        user_prompt = USER_PROMPTS_TEMPLATE_DICT[user_prompt_template_name]
        movie_title = MOVIES_DICT[movie_id]["title"]+" ("+str(MOVIES_DICT[movie_id]["year"])+")"
        user_prompts = [user_prompt.format(claim=claim, movie_title=movie_title) for claim in claims]
        if system_prompt_template_name is not None:
            system_prompt = SYSTEM_PROMPTS_TEMPLATE_DICT[system_prompt_template_name]
        else:
            system_prompt = None
        
    else:
        raise NotImplementedError("Not implemented yet.")
    return user_prompts, system_prompt


