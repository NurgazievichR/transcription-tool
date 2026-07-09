"""LLM post-processing: speaker correction and split of merged utterances."""

import json
import re
from difflib import SequenceMatcher

from openai import AzureOpenAI

from config import load_config, validate_azure_openai_config
from transcript_format import format_numbered_transcript, letter_to_speaker, renumber_utterances, speaker_to_letter


SYSTEM_PROMPT = """You review a diarized transcript that may have errors.

Two tasks:
1) SPEAKER CORRECTION — flip speaker label on whole utterances that look wrong.
2) SPLIT — when one utterance clearly contains multiple speaker turns merged together,
   split it into parts. Each part gets its own speaker label.

Rules:
- Do NOT rewrite, translate, or paraphrase text.
- For splits: copy text verbatim from the original utterance into parts.
- Parts must concatenate to the full original utterance (same words, only split).
- Use dialogue cues: Q&A, names, self-reference, style shifts, turn-taking.
- Speakers are labeled Speaker A, Speaker B, Speaker C, etc.
- Return JSON only, no markdown.

Output schema:
{
  "corrections": [
    {"utterance_id": 12, "corrected_speaker": "Speaker B", "confidence": 0.91, "reason": "..."}
  ],
  "splits": [
    {
      "utterance_id": 7,
      "confidence": 0.88,
      "reason": "question then answer from different speakers",
      "parts": [
        {"speaker": "Speaker A", "text": "exact substring from utterance"},
        {"speaker": "Speaker B", "text": "exact substring from utterance"}
      ]
    }
  ]
}

Include only utterances you would change. confidence is 0.0 to 1.0.
Prefer splits for long utterances with obvious turn changes inside."""


def load_azure_openai_config() -> dict:
    app_config = load_config()
    ok, missing = validate_azure_openai_config(app_config)
    if not ok:
        raise ValueError(
            f"Missing Azure OpenAI credentials: {', '.join(missing)}. "
            "Add AZURE_OPENAI_API_KEY, ENDPOINT_URL, DEPLOYMENT_NAME to .env"
        )
    return app_config["azure_openai"]


def _build_client(config: dict) -> AzureOpenAI:
    return AzureOpenAI(
        api_key=config["api_key"],
        api_version=config["api_version"],
        azure_endpoint=config["endpoint"].rstrip("/"),
    )


def _normalize_label(label: str) -> str:
    if label.startswith("Speaker "):
        return label
    return f"Speaker {speaker_to_letter(label)}"


def _texts_match(original: str, parts: list[dict]) -> bool:
    joined = " ".join(p.get("text", "").strip() for p in parts)
    a = re.sub(r"\s+", " ", original.strip().lower())
    b = re.sub(r"\s+", " ", joined.strip().lower())
    if a == b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.95


def postprocess_with_llm(
    utterances: list[dict],
    *,
    expected_speakers: int | None = None,
    config: dict | None = None,
) -> dict:
    """Send numbered transcript to GPT-4o; get speaker corrections and split suggestions."""
    if config is None:
        config = load_azure_openai_config()

    numbered = format_numbered_transcript(utterances)
    speaker_hint = (
        f"Expected speaker count: {expected_speakers}."
        if expected_speakers
        else "Infer the likely speaker count from context."
    )

    user_prompt = (
        f"{speaker_hint}\n\n"
        "Review this transcript. Return corrections and/or splits JSON.\n\n"
        f"{numbered}"
    )

    client = _build_client(config)
    response = client.chat.completions.create(
        model=config["deployment"],
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    if isinstance(parsed, list):
        parsed = {"corrections": parsed, "splits": []}

    corrections = []
    for item in parsed.get("corrections", []):
        corrections.append({
            "utterance_id": int(item["utterance_id"]),
            "corrected_speaker": _normalize_label(item["corrected_speaker"]),
            "confidence": float(item.get("confidence", 0.0)),
            "reason": item.get("reason", ""),
        })

    splits = []
    for item in parsed.get("splits", []):
        parts = []
        for part in item.get("parts", []):
            parts.append({
                "speaker": _normalize_label(part["speaker"]),
                "text": part.get("text", "").strip(),
            })
        splits.append({
            "utterance_id": int(item["utterance_id"]),
            "confidence": float(item.get("confidence", 0.0)),
            "reason": item.get("reason", ""),
            "parts": parts,
        })

    return {
        "corrections": corrections,
        "splits": splits,
        "raw_response": content,
    }


def correct_speakers_with_llm(
    utterances: list[dict],
    *,
    expected_speakers: int | None = None,
    config: dict | None = None,
) -> dict:
    """Backward-compatible wrapper (corrections only in return shape)."""
    result = postprocess_with_llm(
        utterances, expected_speakers=expected_speakers, config=config,
    )
    return {
        "corrections": result["corrections"],
        "raw_response": result["raw_response"],
    }


def apply_corrections(
    utterances: list[dict],
    corrections: list[dict],
    *,
    confidence_threshold: float = 0.75,
) -> tuple[list[dict], list[dict]]:
    """
    Apply LLM corrections above threshold.

    Returns (corrected_utterances, applied_flips_log)
    """
    by_id = {u["id"]: dict(u) for u in utterances}
    flips = []

    for corr in corrections:
        uid = corr["utterance_id"]
        if uid not in by_id:
            continue
        if corr["confidence"] < confidence_threshold:
            continue

        old = by_id[uid]["speaker_label"]
        new = corr["corrected_speaker"]
        if old == new:
            continue

        by_id[uid]["speaker_label"] = new
        by_id[uid]["speaker"] = letter_to_speaker(speaker_to_letter(new))
        flips.append({
            "utterance_id": uid,
            "from": old,
            "to": new,
            "confidence": corr["confidence"],
            "reason": corr.get("reason", ""),
        })

    corrected = [by_id[u["id"]] for u in utterances]
    return corrected, flips


def apply_splits(
    utterances: list[dict],
    splits: list[dict],
    *,
    confidence_threshold: float = 0.75,
) -> tuple[list[dict], list[dict]]:
    """
    Split merged utterances into shorter parts with per-part speaker labels.

    Timestamps are estimated proportionally by text length within the original span.
    """
    split_by_id = {
        s["utterance_id"]: s
        for s in splits
        if s.get("confidence", 0) >= confidence_threshold and s.get("parts")
    }
    logs: list[dict] = []
    output: list[dict] = []

    for utt in utterances:
        uid = utt["id"]
        spec = split_by_id.get(uid)
        if not spec or len(spec["parts"]) < 2:
            output.append(dict(utt))
            continue

        if not _texts_match(utt["text"], spec["parts"]):
            output.append(dict(utt))
            logs.append({
                "utterance_id": uid,
                "status": "skipped",
                "reason": "split text did not match original utterance",
                "confidence": spec["confidence"],
            })
            continue

        start, end = utt["start"], utt["end"]
        duration = max(end - start, 0.001)
        total_len = sum(max(len(p["text"]), 1) for p in spec["parts"])
        cursor = start
        new_parts = []

        for i, part in enumerate(spec["parts"]):
            share = max(len(part["text"]), 1) / total_len
            part_end = end if i == len(spec["parts"]) - 1 else cursor + duration * share
            label = part["speaker"]
            new_parts.append({
                "start": round(cursor, 3),
                "end": round(part_end, 3),
                "text": part["text"],
                "speaker": letter_to_speaker(speaker_to_letter(label)),
                "speaker_label": label,
            })
            cursor = part_end

        output.extend(new_parts)
        logs.append({
            "utterance_id": uid,
            "status": "applied",
            "from_utterances": 1,
            "to_utterances": len(new_parts),
            "confidence": spec["confidence"],
            "reason": spec.get("reason", ""),
            "speakers": [p["speaker_label"] for p in new_parts],
        })

    return renumber_utterances(output), logs


def postprocess_transcript(
    utterances: list[dict],
    llm_result: dict,
    *,
    confidence_threshold: float = 0.75,
    enable_splits: bool = True,
    enable_corrections: bool = True,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Apply splits first, then speaker label corrections. Returns (utterances, split_logs, flip_logs)."""
    current = utterances
    split_logs: list[dict] = []
    flip_logs: list[dict] = []

    if enable_splits and llm_result.get("splits"):
        current, split_logs = apply_splits(
            current, llm_result["splits"], confidence_threshold=confidence_threshold,
        )

    if enable_corrections and llm_result.get("corrections"):
        current, flip_logs = apply_corrections(
            current, llm_result["corrections"], confidence_threshold=confidence_threshold,
        )

    return current, split_logs, flip_logs
