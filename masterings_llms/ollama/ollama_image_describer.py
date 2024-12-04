import ollama
import shutil
import streamlit as st

st.title("Image Describer!")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

print(uploaded_file)

if uploaded_file is not None:
    print(uploaded_file.name)
    print(type(uploaded_file))
    st.image(uploaded_file, caption="Uploaded Image.", use_container_width=True)
    
    with open(uploaded_file.name, "wb") as f: 
        shutil.copyfileobj(uploaded_file, f)

    response = ollama.chat(
        model="llava",
        messages = [
            { 
                'role':'user',
                'content': 'Describe the image.',
                'images':  [uploaded_file.name]
            }
        ]
    )

    st.markdown(response['message']['content'])
    print(response['message']['content'])
    