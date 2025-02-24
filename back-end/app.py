import os
import cv2
import numpy as np
import pickle
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_cors import cross_origin
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Check if the model files exist before loading
DATASET_DIR = "face_dataset"
MODEL_FILE = "face_model.yml"   
NAMES_FILE = "person_names.pkl"

if os.path.exists(MODEL_FILE):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(MODEL_FILE)
else:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.path.exists(NAMES_FILE):
    with open(NAMES_FILE, "rb") as f:
        person_names = pickle.load(f)
else:
    person_names = {}

# Load Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

BASE_CONFIDENCE_THRESHOLD = 95
cap = None

def initialize_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Error: Could not open webcam.")

def preprocess_face(face):
    face = cv2.resize(face, (100, 100))
    face = cv2.equalizeHist(face)
    return face

def adaptive_threshold(confidence):
    if confidence < 30:
        return 35
    elif confidence < 50:
        return 55
    else:
        return BASE_CONFIDENCE_THRESHOLD

def generate_frames():
    initialize_camera()
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = preprocess_face(gray[y:y+h, x:x+w])
            try:
                label, confidence = face_recognizer.predict(face)
                threshold = adaptive_threshold(confidence)
                
                if confidence > threshold:
                    name = "Unknown"
                else:
                    name = person_names.get(label, "Unknown")
            except Exception as e:
                name, confidence = "Unknown", 0

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({round(confidence, 2)})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        _, buffer = cv2.imencode('.png', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognize_face')
def recognize_face():
    initialize_camera()
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Could not capture frame"}), 500

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    recognized_faces = []
    for (x, y, w, h) in faces:
        face = preprocess_face(gray[y:y+h, x:x+w])
        
        try:
            label, confidence = face_recognizer.predict(face)
            threshold = adaptive_threshold(confidence)

            if confidence > threshold:
                name = "Unknown"
            else:
                name = person_names.get(label, "Unknown")
        except Exception as e:
            name, confidence = "Unknown", 0

        recognized_faces.append({"name": name, "confidence": round(confidence, 2)})

    return jsonify({"recognized_faces": recognized_faces})

@app.route('/upload_face', methods=['POST'])
def upload_face():
    if 'file' not in request.files or 'name' not in request.form:
        return jsonify({"error": "Missing file or name"}), 400

    file = request.files['file']
    name = request.form['name'].strip()

    if not name:
        return jsonify({"error": "Invalid name"}), 400

    # Create a folder for the person if it doesn't exist
    person_folder = os.path.join(DATASET_DIR, secure_filename(name))
    os.makedirs(person_folder, exist_ok=True)

    # Save the file with a unique name
    filename = secure_filename(file.filename)
    image_path = os.path.join(person_folder, filename)

    try:
        # Save the file directly
        file.save(image_path)

        # Verify the file was saved correctly
        if os.path.getsize(image_path) == 0:
            os.remove(image_path)
            return jsonify({"error": "File was not saved correctly."}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

    return jsonify({"message": f"Image saved for {name}", "path": image_path})


def train_model():
    faces, labels = [], []
    label_map = person_names.copy()
    next_label = max(label_map.values(), default=-1) + 1

    for person_name in os.listdir(DATASET_DIR):
        person_folder = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue

        if person_name not in label_map:
            label_map[person_name] = next_label
            next_label += 1

        for image_file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, image_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            face = preprocess_face(img)
            faces.append(face)
            labels.append(label_map[person_name])

    if len(faces) == 0:
        return {"error": "No valid faces found in dataset."}

    face_recognizer.train(faces, np.array(labels))
    face_recognizer.write(MODEL_FILE)

    with open(NAMES_FILE, "wb") as f:
        pickle.dump(label_map, f)

    return {"message": "Model trained successfully!"}

@app.route('/train_model', methods=['POST'])
# @cross_origin()  # Enable CORS for this route
def train_model_endpoint():
    print("Train model endpoint called")
    result = train_model()
    return jsonify(result)

if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        if cap and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
