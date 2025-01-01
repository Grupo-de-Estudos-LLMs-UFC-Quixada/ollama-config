import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

st.title("Local LLM with Langchain!")

# Input for the prompt
prompt = st.text_area("Write your prompt.")
button = st.button("Okay")

if button:
    if prompt:
        # Initialize the local LLM
        llm = OllamaLLM(model="llama3.2:latest")

        # Define the prompt template
        template = """You are a helpful assistante. You have been asked the following question:
        Question: {question}
        Please provide a detailed and informative answer as a list with short items."""
        prompt_template = PromptTemplate(input_variables=["question"], template=template)

        # Create an LLMChain with the prompt template
        chain = prompt_template | llm

        # Generate a response using the local LLM
        response = chain.invoke(input={"question": prompt})
    
        # Display the response
        st.markdown(response)