def format_duration(seconds: float) -> str:
    if seconds < 3600:
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_speaker(speaker: str) -> str:
    try:
        num = int(speaker.split("_")[1])
        if num < 26:
            return f"[{chr(ord('A') + num)}]"
        return f"[{num + 1}]"
    except (IndexError, ValueError):
        return f"[{speaker}]"
