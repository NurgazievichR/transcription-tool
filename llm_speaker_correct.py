"""LLM post-processing: speaker correction and split of merged utterances."""

import json
import re
from difflib import SequenceMatcher

from openai import AzureOpenAI

from config import load_config, validate_azure_openai_config
from transcript_format import format_numbered_transcript, letter_to_speaker, renumber_utterances, speaker_to_letter


SYSTEM_PROMPT = """You review a diarized transcript that may have errors.

Two tasks (in priority order):
1) SPLIT — when one utterance clearly contains multiple speaker turns merged together,
   split it into parts. Each part gets its own speaker label.
2) SPEAKER CORRECTION — flip speaker label on a whole utterance ONLY when you are very sure.

Rules:
- Do NOT rewrite, translate, or paraphrase text.
- For splits: copy text verbatim from the original utterance into parts.
- Parts must concatenate to the full original utterance (same words, only split).
- Prefer SPLIT over CORRECTION for long utterances (>20 seconds or >120 characters).
- Only correct a whole utterance when pronouns, names, or turn-taking clearly prove the label is wrong.
- "don Jesús" / "Don Jesús" in text usually means the speaker is NOT Jesús (they talk about him).
- "me dijo su mamá" / "dígale a Don Jesús" — track who reports vs who is mentioned; do not flip on weak guesses.
- Short backchannel lines ("sí", "claro", "exacto", "órale") belong to the listener, not the talker.
- If unsure, return nothing for that utterance. False changes are worse than leaving the label.
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
Use confidence >= 0.90 only for whole-utterance corrections. Splits can be 0.80+."""


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


def _utterance_duration(utt: dict) -> float:
    return max(float(utt.get("end", 0)) - float(utt.get("start", 0)), 0.0)


# Third-person references to a named party — flipping TO that party is usually wrong.
_NAMED_PARTY_RE = re.compile(r"\b(don|doña)\s+[a-záéíóúñ]+", re.IGNORECASE)
_SHORT_BACKCHANNEL_RE = re.compile(
    r"^(sí|si|claro|exacto|órale|ora(le)?|ajá|bueno|listo|ok|okey|mm[- ]?hm|jeje+)\b",
    re.IGNORECASE,
)


def _reject_correction_reason(utt: dict, corr: dict, utterances: list[dict]) -> str | None:
    """Return rejection reason, or None if correction may be applied."""
    uid = utt["id"]
    conf = corr.get("confidence", 0.0)
    text = utt.get("text", "")
    old = utt["speaker_label"]
    new = corr["corrected_speaker"]
    duration = _utterance_duration(utt)

    if old == new:
        return "same label"

    # Long blocks should be split, not flipped.
    if duration > 25 or len(text) > 200:
        if conf < 0.95:
            return f"long utterance ({duration:.0f}s) — split instead of flip"

    by_id = {u["id"]: u for u in utterances}
    idx = next((i for i, u in enumerate(utterances) if u["id"] == uid), None)
    if idx is not None:
        prev_label = utterances[idx - 1]["speaker_label"] if idx > 0 else None
        next_label = utterances[idx + 1]["speaker_label"] if idx < len(utterances) - 1 else None
        if prev_label == next_label == old and conf < 0.93:
            return "sandwiched same speaker — likely merge, not flip"

    # Very short backchannels rarely need a flip unless very confident.
    if len(text.split()) <= 4 and _SHORT_BACKCHANNEL_RE.match(text.strip()) and conf < 0.90:
        return "short backchannel — skip flip"

    # Direct report to "usted" — usually the reporter is the current speaker.
    if re.search(r"\bme dijo\b", text, re.IGNORECASE) and re.search(r"\busted\b", text, re.IGNORECASE):
        if conf < 0.94:
            return "reporting speech to usted — likely correct speaker"

    # Talking about Don/Doña by name — usually not that person speaking.
    if _NAMED_PARTY_RE.search(text) and conf < 0.94:
        return "third-person honorific reference — likely correct speaker"

    return None


def filter_corrections(
    utterances: list[dict],
    corrections: list[dict],
    *,
    confidence_threshold: float,
    skip_utterance_ids: set[int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Keep only corrections that pass confidence + safety checks."""
    skip = skip_utterance_ids or set()
    by_id = {u["id"]: u for u in utterances}
    kept: list[dict] = []
    rejected: list[dict] = []

    for corr in corrections:
        uid = corr["utterance_id"]
        if uid in skip:
            rejected.append({**corr, "status": "rejected", "reject_reason": "utterance was split"})
            continue
        if uid not in by_id:
            rejected.append({**corr, "status": "rejected", "reject_reason": "unknown utterance id"})
            continue
        if corr.get("confidence", 0) < confidence_threshold:
            rejected.append({
                **corr,
                "status": "rejected",
                "reject_reason": f"below threshold ({confidence_threshold:.0%})",
            })
            continue

        reason = _reject_correction_reason(by_id[uid], corr, utterances)
        if reason:
            rejected.append({**corr, "status": "rejected", "reject_reason": reason})
            continue

        kept.append(corr)

    return kept, rejected


def filter_splits(
    utterances: list[dict],
    splits: list[dict],
    *,
    confidence_threshold: float,
    min_duration_s: float = 12.0,
) -> tuple[list[dict], list[dict]]:
    """Keep splits that pass confidence, text match, and sanity checks."""
    by_id = {u["id"]: u for u in utterances}
    kept: list[dict] = []
    rejected: list[dict] = []

    for spec in splits:
        uid = spec.get("utterance_id")
        if uid not in by_id:
            rejected.append({**spec, "status": "rejected", "reject_reason": "unknown utterance id"})
            continue
        if spec.get("confidence", 0) < confidence_threshold:
            rejected.append({
                **spec,
                "status": "rejected",
                "reject_reason": f"below threshold ({confidence_threshold:.0%})",
            })
            continue
        parts = spec.get("parts") or []
        if len(parts) < 2:
            rejected.append({**spec, "status": "rejected", "reject_reason": "need 2+ parts"})
            continue

        speakers = {p.get("speaker") for p in parts}
        if len(speakers) < 2:
            rejected.append({**spec, "status": "rejected", "reject_reason": "all parts same speaker"})
            continue

        utt = by_id[uid]
        if _utterance_duration(utt) < min_duration_s:
            rejected.append({
                **spec,
                "status": "rejected",
                "reject_reason": f"utterance shorter than {min_duration_s:.0f}s",
            })
            continue

        if not _texts_match(utt["text"], parts):
            rejected.append({**spec, "status": "rejected", "reject_reason": "split text mismatch"})
            continue

        kept.append(spec)

    return kept, rejected


def utterances_for_gpt(
    utterances: list[dict],
    *,
    min_duration_s: float = 12.0,
    min_chars: int = 120,
) -> list[dict]:
    """Only send long / likely-merged blocks to GPT."""
    return [
        u for u in utterances
        if _utterance_duration(u) >= min_duration_s or len(u.get("text", "")) >= min_chars
    ]


def postprocess_with_llm(
    utterances: list[dict],
    *,
    expected_speakers: int | None = None,
    config: dict | None = None,
    only_long: bool = True,
) -> dict:
    """Send numbered transcript to GPT-4o; get speaker corrections and split suggestions."""
    target = utterances_for_gpt(utterances) if only_long else utterances
    if not target:
        return {"corrections": [], "splits": [], "raw_response": "{}"}
    if config is None:
        config = load_azure_openai_config()

    numbered = format_numbered_transcript(target)
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
    confidence_threshold: float = 0.88,
    skip_utterance_ids: set[int] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Apply LLM corrections that pass safety filters.

    Returns (corrected_utterances, applied_flips_log, rejected_log)
    """
    filtered, rejected = filter_corrections(
        utterances,
        corrections,
        confidence_threshold=confidence_threshold,
        skip_utterance_ids=skip_utterance_ids,
    )

    by_id = {u["id"]: dict(u) for u in utterances}
    flips = []

    for corr in filtered:
        uid = corr["utterance_id"]
        old = by_id[uid]["speaker_label"]
        new = corr["corrected_speaker"]

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
    return corrected, flips, rejected


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
    correction_confidence_threshold: float | None = None,
    enable_splits: bool = True,
    enable_corrections: bool = True,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Apply conservative label fixes, then splits on merged blocks.

    Returns (utterances, split_logs, flip_logs, rejected_corrections)
    """
    split_threshold = confidence_threshold
    flip_threshold = correction_confidence_threshold or max(confidence_threshold, 0.88)

    pending_split_ids = {
        s["utterance_id"]
        for s in llm_result.get("splits", [])
        if s.get("confidence", 0) >= split_threshold and len(s.get("parts", [])) >= 2
    }

    current = utterances
    flip_logs: list[dict] = []
    rejected: list[dict] = []

    if enable_corrections and llm_result.get("corrections"):
        current, flip_logs, rejected = apply_corrections(
            current,
            llm_result["corrections"],
            confidence_threshold=flip_threshold,
            skip_utterance_ids=pending_split_ids,
        )

    split_logs: list[dict] = []
    if enable_splits and llm_result.get("splits"):
        filtered_splits, split_rejected = filter_splits(
            current,
            llm_result["splits"],
            confidence_threshold=split_threshold,
        )
        rejected.extend(split_rejected)
        current, split_logs = apply_splits(
            current, filtered_splits, confidence_threshold=0.0,
        )

    return current, split_logs, flip_logs, rejected
