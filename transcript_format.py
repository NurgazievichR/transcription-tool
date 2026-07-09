"""Numbered utterance format for transcripts."""

from utils import format_timestamp


def speaker_to_letter(speaker: str) -> str:
    if speaker.startswith("SPEAKER_"):
        try:
            num = int(speaker.split("_")[1])
            return chr(ord("A") + num)
        except (IndexError, ValueError):
            pass
    if speaker.startswith("Speaker "):
        return speaker.replace("Speaker ", "").strip()
    return speaker


def letter_to_speaker(letter: str) -> str:
    letter = letter.strip().upper()
    if len(letter) == 1 and letter.isalpha():
        num = ord(letter) - ord("A")
        return f"SPEAKER_{num:02d}"
    return letter


def structure_segments(segments: list[dict]) -> list[dict]:
    """Add utterance ids and display labels to raw segments."""
    structured = []
    for idx, seg in enumerate(segments, start=1):
        speaker_letter = speaker_to_letter(seg["speaker"])
        structured.append({
            "id": idx,
            "start": seg["start"],
            "end": seg["end"],
            "speaker": seg["speaker"],
            "speaker_label": f"Speaker {speaker_letter}",
            "text": seg["text"],
        })
    return structured


def renumber_utterances(utterances: list[dict]) -> list[dict]:
    for idx, utt in enumerate(utterances, start=1):
        utt["id"] = idx
    return utterances


def format_utterance_line(utt: dict) -> str:
    start = format_timestamp(utt["start"])
    end = format_timestamp(utt["end"])
    return (
        f'[{utt["id"]}] ({utt["speaker_label"]}, {start}–{end}): '
        f'"{utt["text"]}"'
    )


def format_numbered_transcript(utterances: list[dict]) -> str:
    return "\n".join(format_utterance_line(u) for u in utterances)
