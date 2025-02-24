import os
import cv2
import numpy as np
import pickle
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

#check if the model files exist before loading
MODEL_FILE = "face_model.yml"
NAMES_FILE = "person_names.pkl"

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Error: Model file '{MODEL_FILE}' not found.")

if not os.path.exists(NAMES_FILE):
    raise FileNotFoundError(f"Error: Names file '{NAMES_FILE}' not found.")

#load the trained face recognition model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(MODEL_FILE)

#load all person names
with open(NAMES_FILE, "rb") as f:
    person_names = pickle.load(f)

if not isinstance(person_names, dict):
    raise ValueError("Error: 'person_names.pkl' must contain a dictionary of names.")

#load Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#base confidence threshold (will be adjusted dynamically)
BASE_CONFIDENCE_THRESHOLD = 95 #95 seems best

#initialize the webcam lazily
cap = None

def initialize_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Error: Could not open webcam.")

def preprocess_face(face):
    """applies preprocessing: resizing, histogram equalization."""
    face = cv2.resize(face, (100, 100))  # Normalize size
    face = cv2.equalizeHist(face)  # Improve contrast
    return face

def adaptive_threshold(confidence):
    """
    adjusts confidence threshold dynamically:
    - Lower thresholds when confidence is consistently high.
    - Higher thresholds when confidence is spread out (to reduce false positives).
    """
    if confidence < 30:
        return 35  # More strict when confidence is very high
    elif confidence < 50:
        return 55  # slightly lenient
    else:
        return BASE_CONFIDENCE_THRESHOLD  #efault case

def generate_frames():
    """Generates frames from the webcam with face detection and recognition overlay."""
    initialize_camera()
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = preprocess_face(gray[y:y+h, x:x+w])  # Apply preprocessing
            try:
                label, confidence = face_recognizer.predict(face)
                threshold = adaptive_threshold(confidence)  # Dynamically set threshold
                
                if confidence > threshold:
                    name = "Unknown"
                else:
                    name = person_names.get(label, "Unknown")

            except Exception as e:
                print(f"Recognition error: {e}")
                name, confidence = "Unknown", 0

            # Draw rectangle around face and put label text
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({round(confidence, 2)})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Route that provides the video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognize_face')
def recognize_face():
    """Route that captures a single frame and returns recognized faces in JSON format."""
    initialize_camera()
    
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Could not capture frame"}), 500

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    recognized_faces = []
    for (x, y, w, h) in faces:
        face = preprocess_face(gray[y:y+h, x:x+w])  # Apply preprocessing
        
        try:
            label, confidence = face_recognizer.predict(face)
            threshold = adaptive_threshold(confidence)

            if confidence > threshold:
                name = "Unknown"
            else:
                name = person_names.get(label, "Unknown")

        except Exception as e:
            print(f"Recognition error: {e}")
            name, confidence = "Unknown", 0

        recognized_faces.append({"name": name, "confidence": round(confidence, 2)})

    return jsonify({"recognized_faces": recognized_faces})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """shuts down the server and releases the webcam."""
    global cap
    if cap and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    return jsonify({"message": "Server shutting down"})

if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        if cap and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
