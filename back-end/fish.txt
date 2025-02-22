import cv2
import os
import numpy as np
import pickle


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def get_images_and_labels(dataset_path):
    face_samples = []
    labels = []
    person_names = {}

    label_id = 0  #id for each person
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        
        if not os.path.isdir(person_path):
            continue

        #assign a unique id to each person    like index
        person_names[label_id] = person_name
    

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            #detect a  face in the image
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face_samples.append(face)
                labels.append(label_id)

        label_id += 1

    return face_samples, labels, person_names


#load all my data(imgs)
dataset_path = "back-end/Face_dataset"
faces, labels, person_names = get_images_and_labels(dataset_path)

#trains the recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

#save the trained model and names
face_recognizer.save("back-end\\face_model.yml")
with open("back-end\person_names.pkl", "wb") as f:
    pickle.dump(person_names, f)

print("Model training complete! 🎉")


#load the trained recognizer with labels
face_recognizer.read("back-end\\face_model.yml")
with open("back-end/person_names.pkl", "rb") as f:
    person_names = pickle.load(f)


cap = cv2.VideoCapture(0)#opens webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Error: Could not read frame.")
        break
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect a face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]

        #recognize the face
        label, confidence = face_recognizer.predict(face)

        #getting the name
        name = person_names.get(label, "Unknown")

        #draws a rectangle with the name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({round(confidence, 2)})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
