import streamlit as st

st.title("Ollama!")

prompt = st.text_area(label = "Write your prompt.")
button = st.button("OK")

if button:
    if prompt:
        st.markdown(prompt)