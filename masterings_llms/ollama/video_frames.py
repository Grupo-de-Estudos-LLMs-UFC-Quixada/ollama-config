import cv2

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
    
    print(frame_count)
    # Release the video capture object
    cap.release()

    print(f"Total frames saved: {saved_frame_count}")
    
video_path = "/Users/jmhal/Downloads/cachorros.mp4"
video_to_frames(video_path)