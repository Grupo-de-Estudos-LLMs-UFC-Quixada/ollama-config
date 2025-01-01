import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.title("Text Summarizer with Chunking.")

# Input for the prompt
long_text = st.text_area(label="Paste your long text here.", height=300)
button = st.button("Summarize.")

if button:
    if long_text:
        # Initialize the local LLM
        llm = OllamaLLM(model="llama3.2:latest")

        # Define the prompt template
        template = """You are a helpful assistant. Please summarize the following text:
        Text: {text}
        Provide a concise summary."""
        prompt_template = PromptTemplate(input_variables=["text"], template=template)

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.create_documents([long_text])
        # st.markdown(chunks)

        # Initialize an empty list to store the summaries
        summaries = []
        # Summarize each chunk
        for chunk in chunks:
            chain = prompt_template | llm
            # st.markdown(chunk.page_content)
            summary = chain.invoke({"text": chunk.page_content})
            summaries.append(summary)

        # Combine all summaries into one   
        final_summary = "\n\n".join(summaries)

        # st.markdown(final_summary)
        template = """Combine the following summaries into one concise summary:
        Summaries: {final_summary}
        """
        prompt_template = PromptTemplate(input_variables=["final_summary"], template=template)    
        chain = prompt_template | llm
        combined_final_summary = chain.invoke({"final_summary": final_summary})

        # Display the final summary
        st.subheader("Final Summary")
        st.markdown(combined_final_summary)

