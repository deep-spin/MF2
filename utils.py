from loguru import logger

def extract_srt_text(srt_path):
    lines = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, numbers, and timestamps
            if not line or line.isdigit() or '-->' in line:
                continue
            lines.append(line)
    return ' '.join(lines)


def validate_paths_for_mode(modality, video_path, transcripts_path, synopsis_path):
    logger.info(f"Validating paths for mode {modality}. Non used paths will be set to None.")
    if modality == "video_only":
        if not video_path:
            raise ValueError("video_path is required for video_only mode")
        transcripts_path = None
        synopsis_path = None
        logger.info(f"Selected mode: {modality}. No transcripts or synopsis are used.")
    elif modality == "transcripts_only":
        if not transcripts_path:
            raise ValueError("transcripts_path is required for transcripts_only mode")
        video_path = None
        synopsis_path = None
        logger.info(f"Selected mode: {modality}. Transcripts are only used.")
    elif modality == "synopsis_only":
        if not synopsis_path:
            raise ValueError("synopsis_path is required for synopsis_only mode")
        video_path = None
        transcripts_path = None
        logger.info(f"Selected mode: {modality}. Synopsis is only used.")
    elif modality == "video_and_transcripts":
        if not video_path:
            raise ValueError("video_path is required for video_and_transcripts mode")
        if not transcripts_path:
            raise ValueError("transcripts_path is required for video_and_transcripts mode")
        synopsis_path = None
        logger.info(f"Selected mode: {modality}. Video and transcripts are used.")
    elif modality == "video_and_synopsis":
        if not video_path:
            raise ValueError("video_path is required for video_and_synopsis mode")
        if not synopsis_path:
            raise ValueError("synopsis_path is required for video_and_synopsis mode")
        transcripts_path = None
        logger.info(f"Selected mode: {modality}. Video and synopsis are used.")
    elif modality == "video_transcripts_and_synopsis":
        if not video_path:
            raise ValueError("video_path is required for video_transcripts_and_synopsis mode")
        if not synopsis_path:
            raise ValueError("synopsis_path is required for video_transcripts_and_synopsis mode")
        if not transcripts_path:
            raise ValueError("transcripts_path is required for video_transcripts_and_synopsis mode")
        logger.info(f"Selected mode: {modality}. Video, transcripts and synopsis are used.")
    elif modality == "statement_only":
        video_path = None
        transcripts_path = None
        synopsis_path = None
        logger.info(f"Selected mode: {modality}. Statement is only used.")
    else:
        raise NotImplementedError(f"Inference mode {modality} not implemented. Please construct the corresponding prompt.")
    return video_path, transcripts_path, synopsis_path

def load_transcripts(transcripts_path):
    if transcripts_path:
        logger.info(f"Loading transcripts...")
        transcripts = extract_srt_text(transcripts_path)
        print("read transcripts!")
    else:
        transcripts = None
    return transcripts

def load_synopsis(synopsis_path):
    if synopsis_path:
        logger.info(f"Loading synopsis...")
        with open(synopsis_path, 'r', encoding='utf-8') as f:
            synopsis = f.read()
    else:
        synopsis = None
    return synopsis