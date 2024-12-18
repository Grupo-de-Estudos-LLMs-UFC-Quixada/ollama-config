import ollama
import streamlit as st
import os 
import shutil
import glob
import whisper
import tempfile
import cv2
import ollama
import numpy as np
from pydub import AudioSegment

model = whisper.load_model("medium")

def save_uploaded_video(uploaded_file):
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    audio_path = temp_video_path.replace(".mp4", ".wav")
    video = AudioSegment.from_file(temp_video_path)
    video.export(audio_path, format="wav")

    return audio_path

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path, verbose=True)
    return result["text"]
    
st.title("Video to Text and Q&A with LLM")

uploaded_file = st.file_uploader("Upload your video file", type=["mp4"])

if uploaded_file is not None:
    audio_file_path = save_uploaded_video(uploaded_file)
    transcribed_text = transcribe_audio(audio_file_path)

    st.subheader("Transcribed Text:")
    st.write(transcribed_text)

    prompt = st.text_area("Ask a question based on the transcribed text:")
    button = st.button("Get Answer")
    if button:
        if prompt:
            combined_prompt = f"Based on the following content: {transcribed_text}, answer the following question: {prompt}"
            response = ollama.generate(model="llama3.2:latest", prompt=combined_prompt)
            st.markdown(response['response'])