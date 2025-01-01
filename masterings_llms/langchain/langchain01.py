import streamlit as st
from langchain_ollama import OllamaLLM

st.title("Local LLM with Langchain!")

# Input for the prompt
prompt = st.text_area("Write your prompt.")
button = st.button("Okay")

if button:
    if prompt:
        # Initialize the local LLM
        llm = OllamaLLM(model="llama3.2:latest")

        # Generate a response using the local LLM
        response = llm.invoke(prompt)

        # Display the response
        st.markdown(response)