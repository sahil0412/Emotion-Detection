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

# Load the deep learning model
model = tf.keras.models.load_model('models/Final_Resnet50_Best_model.h5', compile=False)

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

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    camera = cv2.VideoCapture(0)  # Use 0 for the default camera
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Preprocess the frame
            face = cv2.resize(frame, (224, 224))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            # Make prediction using your deep learning model
            prediction = model.predict(face)
            emotion = emotion_labels[np.argmax(prediction)]

            # Overlay prediction on the frame
            cv2.putText(frame, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            # Make prediction using your deep learning model
            prediction = model.predict(face)
            emotion = emotion_labels[np.argmax(prediction)]

            # Pass prediction and image file path to result template
            return render_template('result.html', prediction=emotion, image_file=filename)

if __name__ == '__main__':
    # Ensure the upload directory exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=True)
