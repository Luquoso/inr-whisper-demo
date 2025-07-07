import streamlit as st
import whisper
import tempfile
import os
import torchaudio
import torch

st.title("🎤 Video/Audio Transcriber (Whisper-based)")

st.markdown("Upload a video or audio file below, and we'll transcribe it using OpenAI's Whisper model (runs locally).")

# Upload file
uploaded_file = st.file_uploader("Upload video or audio file", type=["mp4", "mp3", "wav", "m4a", "mov"])

if uploaded_file:
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    st.success("File uploaded. Transcribing...")

    # Load whisper model
    model = whisper.load_model("base")

    # Load audio using torchaudio (bypasses FFmpeg)
    waveform, sample_rate = torchaudio.load(tmp_file_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    audio = waveform.squeeze().numpy()

    # Prepare audio for Whisper
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Transcribe
    options = whisper.DecodingOptions()
    decode_result = whisper.decode(model, mel, options)
    result = {"text": decode_result.text}

    # Show transcript
    st.subheader("📝 Transcript")
    st.text_area("Transcript", result["text"], height=300)

    # Download option
    st.download_button(
        label="Download Transcript as .txt",
        data=result["text"],
        file_name="transcript.txt",
        mime="text/plain"
    )

    # Clean up temp file
    os.remove(tmp_file_path)
