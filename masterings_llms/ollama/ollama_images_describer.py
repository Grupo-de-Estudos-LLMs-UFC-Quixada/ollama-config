import ollama
import streamlit as st
import os

def save_uploaded_file(uploaded_file):
    # To get the current directory, you can use os.getcwd()
    save_path = os.getcwd()
    # Create the full path for the file
    full_path = os.path.join(save_path, uploaded_file.name)
    # Save the file to the current directory
    with open(full_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return st.success(f"Saved file: '{uploaded_file.name}' saved to '{save_path}'.")

st.title("Image Describer!")

uploaded_files = st.file_uploader("Choose an image", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

print(uploaded_files)
if len(uploaded_files) != 0:
    for uploaded_file in uploaded_files:
        save_uploaded_file(uploaded_file)
        print(uploaded_file.name)
        print(type(uploaded_file))
        st.image(uploaded_file, caption="Uploaded Image.", use_container_width=True)
        response = ollama.chat(
            model="llava",
            messages = [
                { 
                    'role':'user',
                    'content': 'Describe the following images separately.',
                    'images':  [uploaded_file.name]
                }
            ]
        )

        st.markdown(response['message']['content'])
        print(response['message']['content'])

   

   

    
    