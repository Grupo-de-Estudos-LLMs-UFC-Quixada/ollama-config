import ollama
import streamlit as st
import os
import fitz

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(uploaded_file)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def save_uploaded_file(uploaded_file):
    save_path = os.getcwd()
    file_path = os.path.join(save_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return st.success(f"Saved file: {uploaded_file.name} to {save_path}")
 

st.title("Chat with PDF!")
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")
print(uploaded_file)

if uploaded_file is not None:
    save_uploaded_file(uploaded_file)
    pdf_text = extract_text_from_pdf(uploaded_file)

    st.subheader("PDF Content:")
    st.markdown(pdf_text)

    prompt = st.text_area(label="Ask a question based on the PDF content.")
    button = st.button("Ok")

    if button:
        if prompt:
            combined_prompt = f"Based on the following content; {pdf_text}\n\n Question: {prompt}"
            response = ollama.generate(model="llama3.2:latest", prompt=combined_prompt)
            st.markdown(response["response"])

    