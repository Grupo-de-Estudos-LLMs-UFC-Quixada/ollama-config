import streamlit as st
import ollama

st.title("Ollama!")

prompt = st.text_area(label = "Write your prompt.")
button = st.button("OK")

if button:
    if prompt:
        response = ollama.generate(prompt=prompt, model='llama3.2')
        st.markdown(response['response'])