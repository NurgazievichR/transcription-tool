#!/usr/bin/env python3
"""
Evaluate diarization: raw AssemblyAI vs GPT-corrected vs reference docx.

Uses cached assemblyai_result.json if present (pass --retranscribe to refresh).
"""

import argparse
import json
import re
import sys
from pathlib import Path

from assemblyai_transcribe import load_assemblyai_config, transcribe_assemblyai
from diarization_eval import evaluate_variant, format_comparison_table
from llm_speaker_correct import apply_corrections, postprocess_transcript, postprocess_with_llm
from reference_docx import load_reference_docx
from transcript_format import format_numbered_transcript, structure_segments

ROOT = Path(__file__).parent
AUDIO = ROOT / "downloads/O-33603/Recordings-0000000248.wav"
DOCX = ROOT / "downloads/O-33603/Recordings-0000000248_SPA - EDT_EN 6.11.26 - SPA-ENG.docx"
OUT_DIR = ROOT / "results/O-33603"
AAI_CACHE = OUT_DIR / "assemblyai_result.json"


def _load_or_transcribe(retranscribe: bool, on_progress) -> dict:
    if AAI_CACHE.exists() and not retranscribe:
        print(f"Using cached AssemblyAI result: {AAI_CACHE}")
        return json.loads(AAI_CACHE.read_text(encoding="utf-8"))

    config = load_assemblyai_config()
    result = transcribe_assemblyai(str(AUDIO), config, locale="auto", on_progress=on_progress)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AAI_CACHE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate raw vs GPT-corrected diarization")
    parser.add_argument("--retranscribe", action="store_true", help="Re-run AssemblyAI instead of cache")
    parser.add_argument("--confidence", type=float, default=0.80, help="Split threshold")
    parser.add_argument("--flip-confidence", type=float, default=0.88, help="Label-fix threshold")
    parser.add_argument("--expected-speakers", type=int, default=2, help="Hint for LLM")
    args = parser.parse_args()

    if not AUDIO.exists() or not DOCX.exists():
        print("Missing audio or reference docx", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def progress(msg: str):
        print(f"  {re.sub(r'[*`]', '', msg).strip()}")

    print("=" * 70)
    print("O-33603 diarization eval: raw AssemblyAI vs GPT-corrected")
    print("=" * 70)

    ref = load_reference_docx(DOCX)
    ref_speakers = sorted({s["speaker"] for s in ref if s["speaker"] != "N/A"})
    print(f"Reference: {len(ref)} turns, speakers: {ref_speakers}")

    aai = _load_or_transcribe(args.retranscribe, progress)
    raw_utterances = structure_segments(aai["segments"])
    print(f"AssemblyAI raw: {len(raw_utterances)} utterances, {aai['num_speakers']} speakers")

    (OUT_DIR / "raw_structured.txt").write_text(
        format_numbered_transcript(raw_utterances), encoding="utf-8",
    )

    print("\nRunning GPT-4o post-processing (Azure OpenAI)...")
    llm_result = postprocess_with_llm(
        raw_utterances,
        expected_speakers=args.expected_speakers,
    )
    corrected_utterances, split_logs, flips, rejected_flips = postprocess_transcript(
        raw_utterances,
        llm_result,
        confidence_threshold=args.confidence,
        correction_confidence_threshold=args.flip_confidence,
        enable_splits=True,
        enable_corrections=True,
    )
    applied_splits = sum(1 for s in split_logs if s.get("status") == "applied")
    print(
        f"LLM splits proposed {len(llm_result.get('splits', []))}, applied {applied_splits}; "
        f"label fixes proposed {len(llm_result.get('corrections', []))}, "
        f"applied {len(flips)}, rejected {len(rejected_flips)}"
    )

    (OUT_DIR / "llm_corrections.json").write_text(
        json.dumps(llm_result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (OUT_DIR / "gpt_corrected_structured.txt").write_text(
        format_numbered_transcript(corrected_utterances), encoding="utf-8",
    )
    (OUT_DIR / "speaker_flips.json").write_text(
        json.dumps(flips, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (OUT_DIR / "speaker_flips_rejected.json").write_text(
        json.dumps(rejected_flips, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    raw_metrics = evaluate_variant(ref, raw_utterances)
    gpt_metrics = evaluate_variant(ref, corrected_utterances)
    raw_metrics["ref_turns"] = len(ref)
    gpt_metrics["ref_turns"] = len(ref)

    table = format_comparison_table([
        ("raw AssemblyAI", raw_metrics),
        ("GPT-corrected", gpt_metrics),
    ])

    report = "\n".join([
        "=" * 70,
        "COMPARISON (vs reference Spanish source text + speaker names)",
        "=" * 70,
        "",
        table,
        "",
        f"GPT flips applied (split ≥ {args.confidence:.0%}, flip ≥ {args.flip_confidence:.0%}): {len(flips)}",
        f"GPT flips rejected: {len(rejected_flips)}",
        "",
        "Sample flips:",
    ])
    for flip in flips[:10]:
        report += (
            f"\n  #{flip['utterance_id']}: {flip['from']} -> {flip['to']} "
            f"({flip['confidence']:.0%}) — {flip.get('reason', '')[:80]}"
        )

    print("\n" + report)
    (OUT_DIR / "diarization_comparison.txt").write_text(report, encoding="utf-8")

    print(f"\nSaved to {OUT_DIR}/")
    print("  raw_structured.txt")
    print("  gpt_corrected_structured.txt")
    print("  llm_corrections.json")
    print("  speaker_flips.json")
    print("  diarization_comparison.txt")


if __name__ == "__main__":
    main()
