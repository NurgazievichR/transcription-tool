#!/usr/bin/env python3
"""Transcribe O-33603 audio via AssemblyAI and compare with reference docx."""

import json
import re
import sys
import zipfile
from difflib import SequenceMatcher
from pathlib import Path
from xml.etree import ElementTree as ET

from assemblyai_transcribe import load_assemblyai_config, transcribe_assemblyai

ROOT = Path(__file__).parent
AUDIO = ROOT / "downloads/O-33603/Recordings-0000000248.wav"
DOCX = ROOT / "downloads/O-33603/Recordings-0000000248_SPA - EDT_EN 6.11.26 - SPA-ENG.docx"
OUT_DIR = ROOT / "results/O-33603"


def parse_ts(s: str) -> float | None:
    m = re.match(r"^(\d+):(\d{2}):(\d{1,2}(?:\.\d+)?)$", s)
    if not m:
        return None
    h, m1, s1 = m.groups()
    return int(h) * 3600 + int(m1) * 60 + float(s1)


def load_reference_docx(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paras = []
    for p in root.findall(".//w:p", ns):
        texts = [t.text or "" for t in p.findall(".//w:t", ns)]
        line = "".join(texts).strip()
        if line:
            paras.append(line)

    start = 0
    for i, p in enumerate(paras):
        if p == "N/A" or parse_ts(p):
            start = i
            break

    segments = []
    i = start
    while i < len(paras):
        speaker = paras[i]
        if i + 1 >= len(paras):
            break
        ts = parse_ts(paras[i + 1])
        if ts is None:
            i += 1
            continue
        src = paras[i + 2] if i + 2 < len(paras) else ""
        eng = paras[i + 3] if i + 3 < len(paras) else ""
        segments.append({
            "speaker": speaker,
            "start": ts,
            "source": src,
            "english": eng,
        })
        i += 4
    return segments


def normalize(text: str, keep_brackets: bool = False) -> str:
    text = text.lower()
    if not keep_brackets:
        text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    return normalize(text).split()


def word_similarity(a: str, b: str) -> float:
    wa, wb = words(a), words(b)
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return SequenceMatcher(None, wa, wb).ratio()


def align_by_time(ref: list[dict], hyp: list[dict], window: float = 8.0) -> list[tuple[dict, dict | None, float]]:
    """Match reference segments to nearest AssemblyAI utterance by start time."""
    used = set()
    pairs = []
    for r in ref:
        best_i = None
        best_dt = window
        for i, h in enumerate(hyp):
            if i in used:
                continue
            dt = abs(h["start"] - r["start"])
            if dt < best_dt:
                best_dt = dt
                best_i = i
        if best_i is not None:
            used.add(best_i)
            sim = word_similarity(r["english"], hyp[best_i]["text"])
            pairs.append((r, hyp[best_i], sim))
        else:
            pairs.append((r, None, 0.0))
    return pairs


def main():
    if not AUDIO.exists():
        print(f"Missing audio: {AUDIO}", file=sys.stderr)
        sys.exit(1)
    if not DOCX.exists():
        print(f"Missing reference: {DOCX}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("O-33603 comparison: AssemblyAI vs reference docx")
    print("=" * 70)
    print(f"Audio:     {AUDIO}")
    print(f"Reference: {DOCX.name}")
    print()

    ref = load_reference_docx(DOCX)
    print(f"Reference: {len(ref)} turns, speakers: {sorted(set(s['speaker'] for s in ref))}")

    config = load_assemblyai_config()

    def progress(msg: str):
        clean = re.sub(r"\*+", "", msg).strip()
        print(f"  {clean}")

    print("\nRunning AssemblyAI (language auto-detect, speaker labels)...")
    result = transcribe_assemblyai(str(AUDIO), config, locale="auto", on_progress=progress)
    hyp = result["segments"]
    print(f"\nAssemblyAI: {len(hyp)} utterances, {result['num_speakers']} speakers, lang={result['language']}")

    (OUT_DIR / "assemblyai_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ref_text_en = " ".join(s["english"] for s in ref)
    hyp_text = " ".join(s["text"] for s in hyp)
    overall_sim = word_similarity(ref_text_en, hyp_text)

    pairs = align_by_time(ref, hyp)
    matched = [p for p in pairs if p[1] is not None]
    sims = [p[2] for p in matched]
    avg_aligned = sum(sims) / len(sims) if sims else 0.0

    # Speaker mapping (reference names vs SPEAKER_XX)
    speaker_stats: dict[str, list[float]] = {}
    for r, h, sim in pairs:
        if h:
            speaker_stats.setdefault(h["speaker"], []).append(sim)

    report_lines = [
        "=" * 70,
        "SUMMARY",
        "=" * 70,
        f"Overall text similarity (English ref vs AssemblyAI): {overall_sim:.1%}",
        f"Time-aligned turn similarity (avg):               {avg_aligned:.1%}",
        f"Reference turns: {len(ref)}  |  AssemblyAI utterances: {len(hyp)}",
        f"Aligned pairs within 8s: {len(matched)} / {len(ref)}",
        f"AssemblyAI speakers: {result['num_speakers']}  |  Reference speakers: {len(set(s['speaker'] for s in ref))}",
        "",
        "Per AssemblyAI speaker (avg similarity to matched English ref):",
    ]
    for sp, vals in sorted(speaker_stats.items()):
        report_lines.append(f"  {sp}: {sum(vals)/len(vals):.1%} ({len(vals)} turns)")

    report_lines += ["", "SAMPLE COMPARISONS (first 15 aligned turns)", "-" * 70]
    shown = 0
    for r, h, sim in pairs:
        if not h:
            continue
        report_lines.append(
            f"\n[{r['start']:.1f}s] REF {r['speaker']} | AAI {h['speaker']} | sim {sim:.0%}\n"
            f"  REF: {r['english'][:120]}\n"
            f"  AAI: {h['text'][:120]}"
        )
        shown += 1
        if shown >= 15:
            break

    report_lines += ["", "WORST MATCHES (bottom 10)", "-" * 70]
    worst = sorted([p for p in pairs if p[1]], key=lambda x: x[2])[:10]
    for r, h, sim in worst:
        report_lines.append(
            f"\n[{r['start']:.1f}s] sim {sim:.0%} | REF {r['speaker']} / AAI {h['speaker']}\n"
            f"  REF: {r['english'][:120]}\n"
            f"  AAI: {h['text'][:120]}"
        )

    report = "\n".join(report_lines)
    print("\n" + report)
    (OUT_DIR / "comparison_report.txt").write_text(report, encoding="utf-8")
    print(f"\nSaved: {OUT_DIR / 'assemblyai_result.json'}")
    print(f"Saved: {OUT_DIR / 'comparison_report.txt'}")


if __name__ == "__main__":
    main()
