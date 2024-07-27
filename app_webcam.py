from flask import Flask, render_template, request, url_for
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from flask import Response, stream_with_context

app = Flask(__name__)

# Define the static and upload directories
STATIC_FOLDER = 'static'
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_name = "Final_Resnet50_Best_model.h5"
# Load the best trained model
model = tf.keras.models.load_model(f'models/{model_name}', compile=False)

# Initialize the face classifier
face_classifier = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed', methods=['POST'])
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
# sahiltest1 (480, 640, 3)
# sahiltest1 (480, 640)
def gen_frames():
    camera = cv2.VideoCapture(0)  # Use 0 for the default camera
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
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
                    face_resized = cv2.resize(frame, (224, 224)) # Load and resize the image
                    face_array = img_to_array(face_resized)  # Convert image to array
                    face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension
                    face_array = tf.keras.applications.resnet50.preprocess_input(face_array)
                elif model_name == "Resnet50V2PretrainedAug48.h5":
                    face_resized = cv2.resize(frame, (48, 48)) # Load and resize the image
                    face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                    face_normalized = face.astype("float") / 255.0
                    face_array = img_to_array(face_normalized)  # Convert image to array
                    face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension
                elif model_name == "Resnet50V2PretrainedAug224.h5":
                    face_resized = cv2.resize(frame, (224, 224)) # Load and resize the image
                    face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                    face_normalized = face.astype("float") / 255.0
                    face_array = img_to_array(face_normalized)  # Convert image to array
                    face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension
                elif model_name == "Final_Resnet50_Best_model.h5":
                    face_resized = cv2.resize(frame, (224, 224))
                    face = face_resized.astype("float") / 255.0
                    face = img_to_array(face)
                    face = np.expand_dims(face, axis=0)

                # Make prediction using your deep learning model
                prediction = model.predict(face_array)
                emotion = emotion_labels[np.argmax(prediction)]
                # Draw a rectangle around the face and put the emotion label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
            # Convert the frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame as a multipart response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                       
# Prediction page route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Read and preprocess the image
            file_img = cv2.imread(filepath)
            face = cv2.resize(file_img, (224, 224))
            # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            prediction = model.predict(face)
            emotion = emotion_labels[np.argmax(prediction)]
            return render_template('result.html', prediction=emotion, image_file=filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=True)
