import streamlit as st
import spacy
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the spaCy model
# spacy.cli.download("en_core_web_md")
nlp = spacy.load("en_core_web_md")

with st.form(key='my_form', clear_on_submit=True):
    # Input fields for the form
    note = st.text_area(label="Paste your note here.", height=300)
    # Submit button
    submit_button = st.form_submit_button(label='Save')

if submit_button:
    if note:
        with open("note.txt", 'r') as file:
            content = file.read()
            if note not in content:
                content += f"\n\n{note}"
                with open("note.txt", 'w') as file:
                    file.write(content)

question = st.text_input(label="Enter your question:")
button = st.button('ASK')

if button:
    if question:
        # Initialize the local LLM
        llm = OllamaLLM(model="llama3.2:latest")

        # Define the prompt template
        template = """You are a helpful assistant. Please answer the following question based on the below text:
        Question: {question}
        Text: {text}        
        """
        prompt_template = PromptTemplate(input_variables=["question", "text"], template=template)

        with open("note.txt", 'r') as file:
            content = file.read()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        chunks = text_splitter.create_documents([content])

        # Initialize an empty list to store the summaries
        similarities = []
        # Summarize each chunk
        for chunk in chunks:
            similarity_score = nlp(question).similarity(nlp(chunk.page_content))
            similarities.append((similarity_score,chunk.page_content))

        ordered_chunks = sorted(similarities, key=lambda x: x[0], reverse=True)[:3]

        text=""
        for score, sentence in ordered_chunks:
            st.markdown(f"**Similarity Score:** {score:.4f} - Sentence: {sentence}")
            text += f"{sentence}\n\n"
        chain = prompt_template | llm
        answer = chain.invoke({"question": question, "text": text})
        
        st.subheader("Answer:")
        st.markdown(answer)
