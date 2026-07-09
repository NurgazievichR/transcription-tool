"""
Streamlit app: upload audio/video → transcribe with AssemblyAI + optional GPT speaker correction.
"""

import json
import tempfile
from pathlib import Path

import streamlit as st

from assemblyai_transcribe import load_assemblyai_config, transcribe_assemblyai
from config import UPLOAD_TYPES, validate_assemblyai_config, validate_azure_openai_config, load_config
from llm_speaker_correct import (
    load_azure_openai_config,
    postprocess_transcript,
    postprocess_with_llm,
)
from transcript_format import format_numbered_transcript, structure_segments
from utils import format_duration

st.set_page_config(page_title="Transcription", page_icon="🎙️", layout="wide")

st.title("🎙️ Transcription")
st.caption("AssemblyAI diarization with optional GPT speaker-label correction")

app_config = load_config()
assemblyai_ok, assemblyai_missing = validate_assemblyai_config(app_config)
azure_ok, azure_missing = validate_azure_openai_config(app_config)

with st.sidebar:
    st.header("Settings")
    st.markdown(
        f"**AssemblyAI:** {'✅' if assemblyai_ok else '❌ ' + ', '.join(assemblyai_missing)}"
    )
    st.markdown(
        f"**Azure OpenAI (GPT):** {'✅' if azure_ok else '❌ ' + ', '.join(azure_missing)}"
    )

    locale_options = [
        "Auto-detect",
        "en-US — English (US)",
        "en-GB — English (UK)",
        "es-MX — Spanish (Mexico)",
        "es-ES — Spanish (Spain)",
        "fr-FR — French",
        "de-DE — German",
        "pt-BR — Portuguese",
        "ru-RU — Russian",
    ]
    locale_selection = st.selectbox("Source language", locale_options, index=0)
    locale = "auto" if locale_selection == "Auto-detect" else locale_selection.split(" — ")[0]

    st.divider()
    st.subheader("Speakers")
    auto_speakers = st.checkbox(
        "Auto speaker count",
        value=True,
        help="AssemblyAI detects the number of speakers automatically.",
    )
    manual_speakers = None
    if not auto_speakers:
        manual_speakers = st.number_input(
            "Expected speakers",
            min_value=1,
            max_value=10,
            value=2,
            help="Sent to AssemblyAI as speakers_expected and used as a hint for GPT.",
        )

    st.divider()
    enable_gpt = st.checkbox(
        "GPT post-processing",
        value=azure_ok,
        disabled=not azure_ok,
        help="GPT may flip speaker labels and split merged utterances. Text is not rewritten.",
    )
    enable_gpt_splits = st.checkbox(
        "Split merged utterances (GPT)",
        value=True,
        disabled=not enable_gpt,
        help="If one block contains multiple speaker turns, GPT can split it into shorter parts.",
    )
    enable_gpt_corrections = st.checkbox(
        "Fix speaker labels (GPT)",
        value=True,
        disabled=not enable_gpt,
        help="Flip Speaker A/B where dialogue context suggests the label is wrong.",
    )
    gpt_confidence = st.slider(
        "GPT confidence threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.75,
        step=0.05,
        disabled=not enable_gpt,
    )

    st.divider()
    st.markdown("**Supported formats**")
    st.markdown("Audio: MP3, WAV, M4A, FLAC, OGG …")
    st.markdown("Video: MP4, MOV, WEBM, MKV, AVI …")

if not assemblyai_ok:
    st.error("Add `ASSEMBLYAI_API_KEY` to the `.env` file in this folder.")
    st.stop()

uploaded = st.file_uploader(
    "Choose a file",
    type=UPLOAD_TYPES,
    help="AssemblyAI accepts audio and video — audio is extracted automatically",
)

if uploaded is None:
    st.info("Upload an audio or video file to transcribe.")
    st.stop()

col1, col2 = st.columns([1, 3])
with col1:
    st.metric("File", uploaded.name)
    size_mb = len(uploaded.getvalue()) / (1024 * 1024)
    st.metric("Size", f"{size_mb:.1f} MB")

run = st.button("Transcribe", type="primary", disabled=not uploaded)


def _utterances_to_export(utterances: list[dict], meta: dict, variant: str) -> dict:
    return {
        "variant": variant,
        "file": meta["filename"],
        "duration_seconds": meta["duration"],
        "language": meta["language"],
        "speakers_count": len({u["speaker_label"] for u in utterances}),
        "transcript_id": meta.get("transcript_id"),
        "speakers_mode": meta.get("speakers_mode"),
        "utterances": utterances,
    }


if run:
    suffix = Path(uploaded.name).suffix or ".mp3"
    status = st.status("Processing...", expanded=True)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / f"input{suffix}"
            input_path.write_bytes(uploaded.getvalue())

            status.write(f"File saved: `{uploaded.name}`")
            config = load_assemblyai_config()
            speakers_expected = None if auto_speakers else int(manual_speakers)

            with st.spinner("Transcription may take several minutes..."):
                result = transcribe_assemblyai(
                    str(input_path),
                    config,
                    locale=locale,
                    speakers_expected=speakers_expected,
                    on_progress=lambda msg: status.write(msg),
                )

            raw_utterances = structure_segments(result["segments"])
            corrected_utterances = raw_utterances
            flips: list[dict] = []
            split_logs: list[dict] = []

            if enable_gpt and azure_ok:
                status.write("**[GPT] Reviewing transcript (splits + speaker labels)...**")
                expected_for_gpt = result["num_speakers"] if auto_speakers else int(manual_speakers)
                llm_result = postprocess_with_llm(
                    raw_utterances,
                    expected_speakers=expected_for_gpt,
                    config=load_azure_openai_config(),
                )
                corrected_utterances, split_logs, flips = postprocess_transcript(
                    raw_utterances,
                    llm_result,
                    confidence_threshold=gpt_confidence,
                    enable_splits=enable_gpt_splits,
                    enable_corrections=enable_gpt_corrections,
                )
                applied_splits = sum(1 for s in split_logs if s.get("status") == "applied")
                status.write(
                    f"GPT splits: proposed **{len(llm_result.get('splits', []))}**, "
                    f"applied **{applied_splits}** | "
                    f"label fixes: proposed **{len(llm_result.get('corrections', []))}**, "
                    f"applied **{len(flips)}** (threshold {gpt_confidence:.0%})"
                )

        status.update(label="Complete", state="complete", expanded=False)

        meta = {
            **result,
            "speakers_mode": "auto" if auto_speakers else f"manual ({manual_speakers})",
        }

        cols = st.columns(5 if enable_gpt and azure_ok else 4)
        cols[0].metric("Utterances (raw)", len(raw_utterances))
        i = 1
        if enable_gpt and azure_ok:
            cols[i].metric("Utterances (GPT)", len(corrected_utterances))
            i += 1
        cols[i].metric("Speakers", result["num_speakers"])
        cols[i + 1].metric("Duration", format_duration(result["duration"]))
        cols[i + 2].metric("Speakers mode", "Auto" if auto_speakers else str(manual_speakers))

        view_options = ["AssemblyAI (raw)"]
        if enable_gpt and azure_ok:
            view_options.append("GPT corrected")

        view = st.radio("Transcript view", view_options, horizontal=True)
        active = corrected_utterances if view == "GPT corrected" else raw_utterances

        st.subheader(view)
        numbered = format_numbered_transcript(active)
        st.text_area(
            "Numbered transcript",
            numbered,
            height=420,
            label_visibility="collapsed",
        )

        if view == "GPT corrected":
            applied_splits = [s for s in split_logs if s.get("status") == "applied"]
            if applied_splits:
                with st.expander(f"Splits applied ({len(applied_splits)})"):
                    for item in applied_splits:
                        st.markdown(
                            f"- **#{item['utterance_id']}** → **{item['to_utterances']}** parts "
                            f"({item['confidence']:.0%}) — {item.get('reason', '')}"
                        )
            if flips:
                with st.expander(f"Speaker label changes ({len(flips)})"):
                    for flip in flips:
                        st.markdown(
                            f"- **#{flip['utterance_id']}**: {flip['from']} → {flip['to']} "
                            f"({flip['confidence']:.0%}) — {flip.get('reason', '')}"
                        )

        stem = Path(uploaded.name).stem
        export = _utterances_to_export(active, meta, variant=view)
        json_bytes = json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8")
        txt_bytes = numbered.encode("utf-8")

        dl1, dl2 = st.columns(2)
        suffix_tag = "gpt" if view == "GPT corrected" else "raw"
        with dl1:
            st.download_button(
                "Download JSON",
                data=json_bytes,
                file_name=f"{stem}_{suffix_tag}.json",
                mime="application/json",
            )
        with dl2:
            st.download_button(
                "Download TXT",
                data=txt_bytes,
                file_name=f"{stem}_{suffix_tag}.txt",
                mime="text/plain",
            )

    except Exception as e:
        status.update(label="Error", state="error", expanded=True)
        st.error(str(e))
