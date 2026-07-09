"""AssemblyAI transcription with speaker diarization."""

import logging
import time
from pathlib import Path

import requests

from config import validate_assemblyai_config, load_config
from utils import format_duration

logger = logging.getLogger(__name__)

ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com"
POLL_INTERVAL = 5
POLL_TIMEOUT = 4 * 3600
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2


def _api_headers(config: dict) -> dict:
    return {"authorization": config["api_key"]}


def _api_request(method: str, url: str, headers: dict,
                 json_body: dict = None, data=None,
                 description: str = "") -> requests.Response:
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.request(
                method, url, headers=headers, json=json_body,
                data=data, timeout=300,
            )
            if resp.status_code >= 500:
                last_error = f"Server error {resp.status_code}: {resp.text}"
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_BASE ** (attempt + 1))
                    continue
            return resp
        except requests.RequestException as e:
            last_error = str(e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE ** (attempt + 1))

    raise RuntimeError(f"{description} failed after {MAX_RETRIES} attempts: {last_error}")


def upload_audio(audio_path: str, config: dict) -> str:
    headers = _api_headers(config)
    url = f"{ASSEMBLYAI_BASE_URL}/v2/upload"

    with open(audio_path, "rb") as f:
        resp = _api_request("POST", url, headers, data=f, description="Upload audio")

    if resp.status_code != 200:
        raise RuntimeError(f"Upload failed: {resp.status_code} {resp.text}")

    return resp.json()["upload_url"]


def _locale_to_assemblyai_code(locale: str) -> str:
    full_code = locale.lower().replace("-", "_")
    supported_regional = {
        "en_us", "en_gb", "en_au", "en_in", "es_es", "fr_fr", "fr_ca",
        "pt_br", "pt_pt", "zh_cn", "zh_tw", "de_de", "it_it", "nl_nl",
        "pl_pl", "ru_ru", "ja_jp", "ko_kr", "tr_tr", "vi_vn", "uk_ua",
    }
    if full_code in supported_regional:
        return full_code
    return full_code.split("_")[0]


def submit_transcription(audio_url: str, config: dict,
                         language_code: str | None = "en",
                         speakers_expected: int | None = None) -> str:
    headers = {**_api_headers(config), "Content-Type": "application/json"}
    url = f"{ASSEMBLYAI_BASE_URL}/v2/transcript"

    body = {
        "audio_url": audio_url,
        "speech_models": ["universal-3-pro", "universal-2"],
        "speaker_labels": True,
        "language_detection": True,
    }

    if language_code:
        body.pop("language_detection", None)
        body["language_code"] = language_code

    if speakers_expected is not None:
        body["speakers_expected"] = speakers_expected

    resp = _api_request("POST", url, headers, json_body=body, description="Submit transcription")
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Submit failed: {resp.status_code} {resp.text}")

    return resp.json()["id"]


def poll_transcription(transcript_id: str, config: dict, on_progress=None) -> dict:
    headers = _api_headers(config)
    url = f"{ASSEMBLYAI_BASE_URL}/v2/transcript/{transcript_id}"
    start_time = time.time()
    last_status = ""

    while True:
        elapsed = time.time() - start_time
        if elapsed > POLL_TIMEOUT:
            raise RuntimeError(f"Transcription timed out after {POLL_TIMEOUT // 3600} hours")

        resp = _api_request("GET", url, headers, description="Poll status")
        if resp.status_code != 200:
            raise RuntimeError(f"Poll failed: {resp.status_code} {resp.text}")

        result = resp.json()
        status = result.get("status", "unknown")

        if status != last_status and on_progress:
            elapsed_min = int(elapsed // 60)
            elapsed_sec = int(elapsed % 60)
            on_progress(f"Status: **{status}** ({elapsed_min}m {elapsed_sec}s)")
            last_status = status

        if status == "completed":
            return result
        if status == "error":
            raise RuntimeError(f"AssemblyAI error: {result.get('error', 'Unknown error')}")

        time.sleep(POLL_INTERVAL)


def fetch_transcript(transcript_id: str, config: dict) -> dict:
    """Fetch a completed transcript by ID (includes utterances + per-word speakers)."""
    headers = _api_headers(config)
    url = f"{ASSEMBLYAI_BASE_URL}/v2/transcript/{transcript_id}"
    resp = _api_request("GET", url, headers, description="Fetch transcript")
    if resp.status_code != 200:
        raise RuntimeError(f"Fetch failed: {resp.status_code} {resp.text}")
    result = resp.json()
    if result.get("status") != "completed":
        raise RuntimeError(f"Transcript not completed: {result.get('status')}")
    return result


def _map_speakers(result: dict) -> dict[str, str]:
    speaker_map: dict[str, str] = {}
    for utterance in result.get("utterances") or []:
        label = utterance.get("speaker", "A")
        if label not in speaker_map:
            speaker_map[label] = f"SPEAKER_{len(speaker_map):02d}"
    return speaker_map


def extract_words(result: dict) -> list[dict]:
    """Flatten per-word speaker labels from AssemblyAI utterances."""
    words: list[dict] = []
    for utterance in result.get("utterances") or []:
        fallback_speaker = utterance.get("speaker", "A")
        for word in utterance.get("words") or []:
            text = (word.get("text") or "").strip()
            if not text:
                continue
            words.append({
                "text": text,
                "start": word.get("start", 0),
                "end": word.get("end", 0),
                "speaker": word.get("speaker") or fallback_speaker,
                "confidence": word.get("confidence", 0.0),
            })
    words.sort(key=lambda w: (w["start"], w["end"]))
    return words


def segments_from_utterances(result: dict) -> list[dict]:
    segments = []
    speaker_map = _map_speakers(result)

    for utterance in result.get("utterances") or []:
        text = utterance.get("text", "").strip()
        if not text:
            continue

        speaker_label = utterance.get("speaker", "A")
        segments.append({
            "start": round(utterance.get("start", 0) / 1000.0, 3),
            "end": round(utterance.get("end", 0) / 1000.0, 3),
            "text": text,
            "speaker": speaker_map[speaker_label],
        })

    return segments


def segments_from_words(
    result: dict,
    *,
    pause_ms: int = 700,
    max_segment_s: float = 45.0,
) -> list[dict]:
    """
    Rebuild segments from per-word speaker labels.

    Splits when the speaker changes, on long pauses, or when a segment exceeds max_segment_s.
    """
    words = extract_words(result)
    if not words:
        return segments_from_utterances(result)

    speaker_map = _map_speakers(result)
    segments: list[dict] = []
    chunk_words: list[dict] = []
    chunk_speaker: str | None = None
    chunk_start_ms: int | None = None
    last_end_ms: int | None = None

    def flush():
        nonlocal chunk_words, chunk_speaker, chunk_start_ms, last_end_ms
        if not chunk_words or chunk_speaker is None:
            chunk_words = []
            chunk_speaker = None
            chunk_start_ms = None
            last_end_ms = None
            return
        text = " ".join(w["text"] for w in chunk_words).strip()
        if text:
            segments.append({
                "start": round(chunk_start_ms / 1000.0, 3),
                "end": round(chunk_words[-1]["end"] / 1000.0, 3),
                "text": text,
                "speaker": speaker_map.get(chunk_speaker, f"SPEAKER_{chunk_speaker}"),
            })
        chunk_words = []
        chunk_speaker = None
        chunk_start_ms = None
        last_end_ms = None

    for word in words:
        speaker = word["speaker"]
        start_ms = int(word["start"])
        end_ms = int(word["end"])
        pause = (start_ms - last_end_ms) if last_end_ms is not None else 0
        duration_s = (end_ms - (chunk_start_ms or start_ms)) / 1000.0

        new_turn = (
            chunk_speaker is not None
            and (
                speaker != chunk_speaker
                or pause >= pause_ms
                or duration_s >= max_segment_s
            )
        )
        if new_turn:
            flush()

        if chunk_speaker is None:
            chunk_speaker = speaker
            chunk_start_ms = start_ms
        chunk_words.append(word)
        last_end_ms = end_ms

    flush()
    return segments


def convert_results_to_segments(
    result: dict,
    *,
    segmentation: str = "utterances",
    pause_ms: int = 700,
) -> list[dict]:
    if segmentation == "utterances":
        return segments_from_utterances(result)
    return segments_from_words(result, pause_ms=pause_ms)


def package_transcription_result(
    api_result: dict,
    *,
    duration: float,
    filename: str,
    speakers_expected: int | None,
    segmentation: str = "utterances",
    pause_ms: int = 700,
) -> dict:
    segments = convert_results_to_segments(api_result, segmentation=segmentation, pause_ms=pause_ms)
    if not segments:
        raise RuntimeError("No speech detected in the file")

    segments_utterance = segments_from_utterances(api_result)
    words = extract_words(api_result)
    num_speakers = len({seg["speaker"] for seg in segments})
    detected = api_result.get("language_code", "en")
    language = detected.split("_")[0] if detected else "en"

    return {
        "segments": segments,
        "segments_utterance": segments_utterance,
        "duration": duration,
        "language": language,
        "num_speakers": num_speakers,
        "filename": filename,
        "transcript_id": api_result.get("id"),
        "speakers_expected": speakers_expected,
        "segmentation": segmentation,
        "word_count": len(words),
        "utterance_count": len(segments_utterance),
    }


def get_audio_duration_ffprobe(audio_path: str) -> float:
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return 0.0


def transcribe_assemblyai(audio_path: str, config: dict,
                          locale: str = "en-US",
                          speakers_expected: int | None = None,
                          on_progress=None,
                          segmentation: str = "utterances",
                          pause_ms: int = 700) -> dict:
    audio_path = Path(audio_path)
    language_code = None if locale == "auto" else _locale_to_assemblyai_code(locale)

    if on_progress is None:
        on_progress = lambda _msg: None

    on_progress("**[1/4] Preparing file...**")
    duration = get_audio_duration_ffprobe(str(audio_path))
    if duration:
        on_progress(f"Duration: {format_duration(duration)}")

    on_progress("**[2/4] Uploading to AssemblyAI...**")
    audio_url = upload_audio(str(audio_path), config)

    on_progress("**[3/4] Submitting job...**")
    if speakers_expected is not None:
        on_progress(f"Speaker hint: **{speakers_expected}** (manual)")
    else:
        on_progress("Speaker count: **auto** (AssemblyAI decides)")
    transcript_id = submit_transcription(
        audio_url, config, language_code, speakers_expected,
    )
    on_progress(f"Job ID: `{transcript_id}`")

    on_progress("**[4/4] Waiting for result...**")
    api_result = poll_transcription(transcript_id, config, on_progress)

    return package_transcription_result(
        api_result,
        duration=duration,
        filename=audio_path.name,
        speakers_expected=speakers_expected,
        segmentation=segmentation,
        pause_ms=pause_ms,
    )


def load_assemblyai_config() -> dict:
    app_config = load_config()
    is_valid, missing = validate_assemblyai_config(app_config)
    if not is_valid:
        raise ValueError(
            f"Missing AssemblyAI credentials: {', '.join(missing)}. "
            "Add ASSEMBLYAI_API_KEY to .env"
        )
    return app_config["assemblyai"]
