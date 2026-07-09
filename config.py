import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus",
    ".webm", ".mp4", ".aac", ".mov", ".mkv", ".avi",
}

UPLOAD_TYPES = ["mp3", "wav", "m4a", "flac", "ogg", "opus", "mp4", "mov", "webm", "aac", "mkv", "avi"]


def load_config() -> dict:
    return {
        "assemblyai": {
            "api_key": os.environ.get("ASSEMBLYAI_API_KEY", ""),
        },
        "azure_speech": {
            "speech_key": os.environ.get("AZURE_SPEECH_KEY", ""),
            "speech_region": os.environ.get("AZURE_SPEECH_REGION", ""),
            "storage_connection_string": os.environ.get("AZURE_STORAGE_CONNECTION_STRING", ""),
        },
        "plunet": {
            "base_url": os.environ.get("PLUNET_BASE_URL", ""),
            "username": os.environ.get("PLUNET_USERNAME", ""),
            "password": os.environ.get("PLUNET_PASSWORD", ""),
        },
        "azure_openai": {
            "api_key": os.environ.get("AZURE_OPENAI_API_KEY", ""),
            "endpoint": os.environ.get("ENDPOINT_URL", ""),
            "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            "deployment": os.environ.get("DEPLOYMENT_NAME", "gpt-4o"),
        },
    }


def validate_azure_openai_config(config: dict | None = None) -> tuple[bool, list[str]]:
    if config is None:
        config = load_config()
    section = config.get("azure_openai", {})
    missing = [k for k, v in section.items() if not v]
    return len(missing) == 0, missing


def validate_assemblyai_config(config: dict | None = None) -> tuple[bool, list[str]]:
    if config is None:
        config = load_config()
    section = config.get("assemblyai", {})
    missing = [k for k, v in section.items() if not v]
    return len(missing) == 0, missing
