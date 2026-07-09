"""Evaluation metrics: raw AssemblyAI vs GPT-corrected vs reference."""

import re
from difflib import SequenceMatcher


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_similarity(a: str, b: str) -> float:
    wa, wb = normalize(a).split(), normalize(b).split()
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return SequenceMatcher(None, wa, wb).ratio()


def align_by_time(ref: list[dict], hyp: list[dict], window: float = 8.0) -> list[tuple[dict, dict | None]]:
    used = set()
    pairs = []
    for r in ref:
        best_i, best_dt = None, window
        for i, h in enumerate(hyp):
            if i in used:
                continue
            dt = abs(h["start"] - r["start"])
            if dt < best_dt:
                best_dt = dt
                best_i = i
        if best_i is not None:
            used.add(best_i)
            pairs.append((r, hyp[best_i]))
        else:
            pairs.append((r, None))
    return pairs


def _majority_speaker_map(pairs: list[tuple[dict, dict | None]], ref_key: str, hyp_key: str) -> dict[str, str]:
    votes: dict[str, dict[str, int]] = {}
    for ref, hyp in pairs:
        if not hyp:
            continue
        rs, hs = ref[ref_key], hyp[hyp_key]
        votes.setdefault(hs, {})
        votes[hs][rs] = votes[hs].get(rs, 0) + 1
    return {hs: max(rv, key=rv.get) for hs, rv in votes.items() if rv}


def speaker_accuracy(
    ref: list[dict],
    hyp: list[dict],
    *,
    ref_speaker_key: str = "speaker",
    hyp_speaker_key: str = "speaker_label",
) -> tuple[float, int, int]:
    pairs = align_by_time(ref, hyp)
    mapping = _majority_speaker_map(pairs, ref_speaker_key, hyp_speaker_key)
    correct = total = 0
    for r, h in pairs:
        if not h:
            continue
        mapped = mapping.get(h[hyp_speaker_key])
        if mapped is None:
            continue
        total += 1
        if mapped == r[ref_speaker_key]:
            correct += 1
    rate = correct / total if total else 0.0
    return rate, correct, total


def evaluate_variant(
    ref: list[dict],
    hyp: list[dict],
    *,
    text_key: str = "text",
    ref_text_key: str = "source",
    hyp_speaker_key: str = "speaker_label",
) -> dict:
    hyp_text = " ".join(h[text_key] for h in hyp)
    ref_text = " ".join(r[ref_text_key] for r in ref)
    pairs = align_by_time(ref, hyp)
    matched = [(r, h) for r, h in pairs if h]
    aligned_sims = [
        word_similarity(r[ref_text_key], h[text_key]) for r, h in matched
    ]
    spk_acc, spk_ok, spk_total = speaker_accuracy(
        ref, hyp, hyp_speaker_key=hyp_speaker_key,
    )

    return {
        "utterances": len(hyp),
        "speakers": len({h[hyp_speaker_key] for h in hyp}),
        "overall_text_similarity": word_similarity(ref_text, hyp_text),
        "aligned_turn_similarity": sum(aligned_sims) / len(aligned_sims) if aligned_sims else 0.0,
        "aligned_pairs": len(matched),
        "speaker_accuracy": spk_acc,
        "speaker_correct": spk_ok,
        "speaker_total": spk_total,
    }


def format_comparison_table(rows: list[tuple[str, dict]]) -> str:
    headers = [
        "Variant",
        "Utterances",
        "Speakers",
        "Text sim",
        "Turn sim",
        "Speaker acc",
        "Aligned",
    ]
    lines = [
        " | ".join(headers),
        " | ".join(["---"] * len(headers)),
    ]
    for name, m in rows:
        lines.append(
            " | ".join([
                name,
                str(m["utterances"]),
                str(m["speakers"]),
                f"{m['overall_text_similarity']:.1%}",
                f"{m['aligned_turn_similarity']:.1%}",
                f"{m['speaker_accuracy']:.1%} ({m['speaker_correct']}/{m['speaker_total']})",
                f"{m['aligned_pairs']}/{m.get('ref_turns', '?')}",
            ])
        )
    return "\n".join(lines)
