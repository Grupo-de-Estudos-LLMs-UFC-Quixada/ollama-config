import streamlit as st
import spacy
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the spaCy model
# spacy.cli.download("en_core_web_md")
nlp = spacy.load("en_core_web_md")

st.title("Text Summarizer with Chunking and Similarity")

# Input for the long text
long_text = st.text_area("Paste your long text here:", height=300)
question = st.text_input("Enter your question:")
button = st.button("Answer")
st.markdown(f"{len(long_text)}")

if button:
    if long_text:
        # Initialize the local LLM
        llm = OllamaLLM(model="llama3.2:latest")

        # Define the prompt template
        template = """You are a helpful assistant. Please answer the following question based on the below text:
        Question: {question}
        Text: {text}        
        """
        prompt_template = PromptTemplate(input_variables=["question", "text"], template=template)

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.create_documents([long_text])

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