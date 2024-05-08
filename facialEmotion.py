import streamlit as st
import cv2
import numpy as np
import pickle
import keras
import os
from time import sleep
from keras.models import load_model


filename = 'emotion_model.h5'
# Load emotion detection model
def load_emotion_model(filename):
    try:
        return load_model(filename)
    except Exception as e:
        st.error(f"Error loading model: {e}")

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Function to detect faces and predict emotions using images
def detect_emotions_image(frame, emotion_model):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Detect faces in the grayscale frame
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in num_faces:
        # Extract the region of interest (face)
        roi_frame = gray_frame[y:y + h, x: x+w]

        # Resize the ROI to match the input size of the model
        cropped_image = cv2.resize(roi_frame, (48, 48))

        # Expand dimensions to match the model's input shape
        cropped_image = np.expand_dims(cropped_image, axis=-1)
        cropped_image = np.expand_dims(cropped_image, axis=0)

        # Perform prediction using the model on the cropped image
        emotion_prediction = emotion_model.predict(cropped_image)
        maxindex = int(np.argmax(emotion_prediction))

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Add text with the predicted emotion label
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 2, cv2.LINE_AA)

    return frame

# Function to detect faces and predict emotions for videos
def detect_emotions_video(video_path, emotion_model, max_frames=5):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()

        if not ret:
            break

        # Detect emotions for each frame
        frame_with_emotions = detect_emotions_image(frame, emotion_model)

        # Display frame with emotions
        st.image(frame_with_emotions, channels="BGR", use_column_width=True)

        frame_count += 1

    cap.release()


# Streamlit app
def main():
    # Set page config and styling
    st.set_page_config(page_title="Emotion Detection App", page_icon=":smiley:", layout="wide", initial_sidebar_state="collapsed")
    
    # Load emotion detection model
    emotion_model = load_emotion_model('emotion_model.h5')

    # Add title and introduction
    st.title('🎭 Emotion Detection App')
    st.write('Welcome to the Emotion Detection Application! This app allows you to upload images and videos to detect emotions. '
            'Simply select the file type you want to analyze and let the magic begin! 😊')

    st.write('The available emotions that can be detected are: angry, disgusted, fearful, happy, neutral, sad, and surprised. '
            'Upload your file and explore the emotions within!')

    st.write("Let's dive in and explore the fascinating world of emotions together! 🚀")

    # Add file type selection radio button
    file_type = st.radio("Select the type of file you want to analyze:", ("Image", "Video"))

    if file_type == "Image":
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Process image
            image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
            frame_with_emotions = detect_emotions_image(frame, emotion_model)
            st.image(frame_with_emotions, channels="BGR", use_column_width=True)

    elif file_type == "Video":
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

        if uploaded_file is not None:
            # Save video file
            video_path = "temp_video.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())
            detect_emotions_video(video_path, emotion_model)

if __name__ == '__main__':
    main()




#https://www.educative.io/answers/how-to-capture-a-single-photo-with-webcam-using-opencv-in-python