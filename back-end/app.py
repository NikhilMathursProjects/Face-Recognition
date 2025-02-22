import os
import cv2
import numpy as np
import pickle
from flask import Flask, Response, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained face recognition model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_model.yml")

# Load person names
with open("person_names.pkl", "rb") as f:
    person_names = pickle.load(f)

# Load OpenCV's face detector (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open the webcam
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]

                try:
                    label, confidence = face_recognizer.predict(face)
                    name = person_names.get(label, "Unknown")
                except Exception:
                    name, confidence = "Unknown", 0

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({round(confidence, 2)})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognize_face')
def recognize_face():
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Could not capture frame"}), 500

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    recognized_faces = []
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]

        try:
            label, confidence = face_recognizer.predict(face)
            name = person_names.get(label, "Unknown")
        except Exception:
            name, confidence = "Unknown", 0

        recognized_faces.append({"name": name, "confidence": round(confidence, 2)})

    return jsonify({"recognized_faces": recognized_faces})

if __name__ == "__main__":
    app.run(debug=True)
