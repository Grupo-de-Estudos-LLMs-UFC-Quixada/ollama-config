import streamlit as st
import spacy
import json
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime

# Load the spaCy model
# spacy.cli.download("en_core_web_md")
nlp = spacy.load("en_core_web_md")

st.title("Chat with Your Diary")

# Get the current date
today = datetime.now().date()

# Create a date selector
selected_date = st.date_input("Select a date:", value=today)
with st.form(key='my_form', clear_on_submit=True):
    # Input fields for the form
    note = st.text_area(label="Write your diary here.", height=300)
    # Submit button
    submit_button = st.form_submit_button(label='Save')

if submit_button:
    if note:
        selected_date = str(selected_date)
        with open("diary.json", "r") as json_file:
            data = json.load(json_file)
            if selected_date in data.keys():
                if note not in data[selected_date]:
                    data[selected_date] += f"\n\n{note}"
                    with open("diary.json", "w") as json_file:
                        json.dump(data, json_file)
            else:
                data[selected_date] = note
                with open("diary.json", "w") as json_file:
                    json.dump(data, json_file)
    else:
        st.error('Please enter a note.')

question = st.text_input(label="Enter your question:")
button = st.button('ASK')

if button:
    if question:
        # Initialize the local LLM
        llm = OllamaLLM(model="llama3.2:latest")

        # Define the prompt template
        template = """You are a helpful assistant. Please answer the following question based on the below diary:
        Question: {question}
        Diary: {text}        
        """
        prompt_template = PromptTemplate(input_variables=["question", "text"], template=template)

        with open("diary.json", "r") as json_file:
            data = json.load(json_file)

        similarities = []
        # Summarize each chunk
        for date, diary in data.items():
            similarity_score = nlp(question).similarity(nlp(diary))
            similarities.append((similarity_score,f"Date: {date}\n Diary: {diary}"))
        
        ordered_chunks = sorted(similarities, key=lambda x: x[0], reverse=True)[:3]

        text=""
        for score, sentence in ordered_chunks:
            st.markdown(f"**Similarity Score:** {score:.4f} - Sentence: {sentence}")
            text += f"{sentence}\n\n"
        chain = prompt_template | llm
        answer = chain.invoke({"question": question, "text": text})

        st.subheader("Answer:")
        st.markdown(answer)
