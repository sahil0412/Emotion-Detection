import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename

# Pull the model from DVC
os.system('dvc pull')

# Define the static and upload directories
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

model_name = "Final_Resnet50_Best_model.h5"
# Load the best trained model
model = tf.keras.models.load_model(f'best_model/{model_name}', compile=False)

# Initialize the face classifier
face_classifier = cv2.CascadeClassifier('best_model/haarcascade_frontalface_default.xml')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_and_predict_emotion_video(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    emotions = []
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        if model_name == "Final_Resnet50_Best_model.h5":
            face_resized = cv2.resize(frame, (224, 224))
            face_normalized = face_resized.astype("float") / 255.0
            face_array = img_to_array(face_normalized)
            face_array = np.expand_dims(face_array, axis=0)
        elif model_name == "CNNFromScratchModel.h5":
            face_resized = cv2.resize(face, (48, 48))
            face_array = img_to_array(face_resized)
            face_array = np.expand_dims(face_array, axis=0)
        elif model_name in ["CNNFromScratchRescaledlModel.h5", "CNNFromScratchAugmentedModel.h5"]:
            face_resized = cv2.resize(face, (48, 48))
            face_normalized = face_resized.astype("float") / 255.0
            face_array = img_to_array(face_normalized)
            face_array = np.expand_dims(face_array, axis=0)
        elif model_name == "Resnet50V2Baisc.h5":
            face_resized = cv2.resize(frame, (224, 224))
            face_array = img_to_array(face_resized)
            face_array = np.expand_dims(face_array, axis=0)
            face_array = tf.keras.applications.resnet50.preprocess_input(face_array)
        elif model_name == "Resnet50V2PretrainedAug48.h5":
            face_resized = cv2.resize(frame, (48, 48))
            face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_normalized = face.astype("float") / 255.0
            face_array = img_to_array(face_normalized)
            face_array = np.expand_dims(face_array, axis=0)
        elif model_name == "Resnet50V2PretrainedAug224.h5":
            face_resized = cv2.resize(frame, (224, 224))
            face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_normalized = face.astype("float") / 255.0
            face_array = img_to_array(face_normalized)
            face_array = np.expand_dims(face_array, axis=0)
        elif model_name == "Final_Resnet50_Best_model.h5":
            face_resized = cv2.resize(frame, (224, 224))
            face = face_resized.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

        prediction = model.predict(face_array)
        emotion = emotion_labels[np.argmax(prediction)]
        emotions.append((x, y, w, h, emotion))
    return emotions

def detect_and_predict_emotion_image(frame):
    
    face_resized = cv2.resize(frame, (224, 224))
    face_normalized = face_resized.astype("float") / 255.0
    face_array = img_to_array(face_normalized)
    face_array = np.expand_dims(face_array, axis=0)

    prediction = model.predict(face_array)
    emotion = emotion_labels[np.argmax(prediction)]
    return emotion

def main():
    st.title("Emotion Detection")

    st.sidebar.title("Upload Image or Start Webcam")

    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Image Upload", "Live Webcam"])

    if app_mode == "Image Upload":
        st.subheader("Upload an image to detect emotions")
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "gif"])
        if uploaded_file is not None:
            file_path = os.path.join(UPLOAD_FOLDER, secure_filename(uploaded_file.name))
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image = cv2.imread(file_path)
            emotion = detect_and_predict_emotion_image(image)
            # for (x, y, w, h, emotion) in emotions:
            #     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #     cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            st.image(image, channels="BGR")
            st.write(f"Predicted Emotion is: {emotion}")

    elif app_mode == "Live Webcam":
        st.subheader("Live webcam feed")
        run = st.checkbox('Run')
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)

        while run:
            _, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            emotions = detect_and_predict_emotion_video(frame)
            for (x, y, w, h, emotion) in emotions:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            FRAME_WINDOW.image(frame)
        else:
            st.write('Stopped')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    main()