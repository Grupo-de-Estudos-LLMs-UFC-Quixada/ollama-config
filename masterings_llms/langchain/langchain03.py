import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

st.title("Local LLM with Langchain!")

# Input for the prompt
user_name = st.text_input("Your Name:")
topic = st.text_input("Topic:")
question = st.text_input("Question:")
instructions = st.text_area("Instructions:")
button = st.button("Okay")

if button:
    if user_name and topic and question and instructions:
        # Initialize the local LLM
        llm = OllamaLLM(model="llama3.2:latest")

        # Define the prompt template
        template = """You are a helpful assistante. {name} has asked a question about {topic}. 

        Question: {question}
        
        Additional Instructions: {instructions}
        
        Please provide a detailed and thoughtful response.
        """
        prompt_template = PromptTemplate(input_variables=["name", "topic", "question", "instructions"], template=template)

        # Create an LLMChain with the prompt template
        chain = prompt_template | llm

        # Generate a response using the local LLM
        response = chain.invoke(input={
            "name": user_name,
            "topic": topic,
            "question": question,
            "instructions": instructions})
    
        # Display the response
        st.markdown(response)