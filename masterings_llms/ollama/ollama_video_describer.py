import ollama
import cv2
import shutil
import streamlit as st
import glob

def video_to_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return
    
    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the interval in frames (2 seconds)
    frame_interval = int(fps * 2)
    
    # Initialize frame counter
    frame_count = 0
    saved_frame_count = 0

    while True:
        # Read the next frame from the video
        ret, frame = cap.read()

        # If the frame was read successfully, break the loop
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Construct the file name
            filename = f"frame_{saved_frame_count}.jpg"
            # Save the frame as an image file
            cv2.imwrite(filename, frame)
            print(f"Saved {filename} ")
            saved_frame_count += 1
        
        frame_count += 1

st.title("Video Describer!")

uploaded_video = st.file_uploader("Choose an video", type=["mp4"])

print(uploaded_video)
if uploaded_video is not None:
    print(uploaded_video.name)
    print(type(uploaded_video))

    with open(uploaded_video.name, "wb") as f:
        shutil.copyfileobj(uploaded_video, f)   

    video_to_frames(uploaded_video.name)

    full_text = ""

    frames = glob.glob(f"frame_*.jpg")
    for frame in frames:
        st.image(frame)
        response = ollama.chat(
            model = 'llama3.2-vision',
            messages = [
                {
                    'role': 'user',
                    'content': 'Describe the image with one sentence',
                    'images': [frame]
                }
            ]
        )
        st.markdown(response['message']['content'])
        full_text += response['message']['content']

    response = ollama.chat(
        model="llama3.2",
        messages = [
        {
            'role': 'user',
            'content': 'Please, summarize the following content:' + full_text,
        }],
    )

    st.markdown(response['message']['content'])
