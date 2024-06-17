import streamlit as st
import tempfile
import os
from func import detect_emotion_func


# Placeholder for your emotion detection function
def detect_emotion(video_path):
    final_emotion=detect_emotion_func(video_path)
    return final_emotion

# Streamlit app
def main():
    st.title("Emotion Detection from Video")

    # File uploader
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

    if uploaded_file is not None:
        # Save the uploaded video file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name

        st.video(temp_video_path)

        # Process button
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Detect emotion from the video
                emotion = detect_emotion(temp_video_path)

            # Display the detected emotion
            st.success(f"Detected Emotion: {emotion}")

        # Clean up the temporary file after processing
        os.remove(temp_video_path)

if __name__ == "__main__":
    main()
