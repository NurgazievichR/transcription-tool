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


def convert_results_to_segments(result: dict) -> list[dict]:
    segments = []
    speaker_map = {}

    for utterance in result.get("utterances") or []:
        text = utterance.get("text", "").strip()
        if not text:
            continue

        speaker_label = utterance.get("speaker", "A")
        if speaker_label not in speaker_map:
            speaker_map[speaker_label] = f"SPEAKER_{len(speaker_map):02d}"

        segments.append({
            "start": round(utterance.get("start", 0) / 1000.0, 3),
            "end": round(utterance.get("end", 0) / 1000.0, 3),
            "text": text,
            "speaker": speaker_map[speaker_label],
        })

    return segments


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
                          on_progress=None) -> dict:
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
    result = poll_transcription(transcript_id, config, on_progress)

    segments = convert_results_to_segments(result)
    if not segments:
        raise RuntimeError("No speech detected in the file")

    num_speakers = len({seg["speaker"] for seg in segments})
    detected = result.get("language_code", language_code or "en")
    language = detected.split("_")[0] if detected else "en"

    return {
        "segments": segments,
        "duration": duration,
        "language": language,
        "num_speakers": num_speakers,
        "filename": audio_path.name,
        "transcript_id": transcript_id,
        "speakers_expected": speakers_expected,
    }


def load_assemblyai_config() -> dict:
    app_config = load_config()
    is_valid, missing = validate_assemblyai_config(app_config)
    if not is_valid:
        raise ValueError(
            f"Missing AssemblyAI credentials: {', '.join(missing)}. "
            "Add ASSEMBLYAI_API_KEY to .env"
        )
    return app_config["assemblyai"]
