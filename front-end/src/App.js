import React, { useState, useEffect, useRef } from "react";
import axios from "axios";

function App() {
    const [recognizedFaces, setRecognizedFaces] = useState([]);
    const [allowedStatus, setAllowedStatus] = useState(false);
    const [capturedImage, setCapturedImage] = useState(null);
    const [personName, setPersonName] = useState("");
    const [isTraining, setIsTraining] = useState(false);
    const videoRef = useRef(null);

    useEffect(() => {
        if (videoRef.current) {
            videoRef.current.src = "/video_feed";  // Use a relative path
        }

        const interval = setInterval(() => {
            axios.get("http://127.0.0.1:5000/recognize_face")
                .then(response => {
                    const faces = response.data.recognized_faces;
                    setRecognizedFaces(faces);

                    // Check if at least one recognized face is NOT "Unknown"
                    const isAllowed = faces.some(face => face.name !== "Unknown");
                    setAllowedStatus(isAllowed);
                })
                .catch(error => console.log(error));
        }, 2000);

        return () => clearInterval(interval);
    }, []);

    const captureImage = () => {
        const imgElement = videoRef.current;
        if (!imgElement) {
            console.error("Image element is not available.");
            return;
        }

        // Create a canvas element
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");

        // Set canvas dimensions to match the image
        canvas.width = imgElement.width;
        canvas.height = imgElement.height;

        // Draw the image onto the canvas
        ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);

        // Check if the canvas has content
        try {
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            if (imageData.data.every(pixel => pixel === 0)) {
                console.error("Canvas is empty. Image feed may not be working.");
                return;
            }
        } catch (error) {
            console.error("Failed to get image data from canvas:", error);
            return;
        }

        // Try to convert the canvas to a Blob
        if (canvas.toBlob) {
            canvas.toBlob(
                (blob) => {
                    if (blob) {
                        // Create a File object from the Blob
                        const file = new File([blob], `${personName}.jpg`, { type: "image/jpeg" });
                        setCapturedImage(file);
                        console.log("Image captured and converted to File object:", file);
                    } else {
                        console.error("Failed to capture image as Blob.");
                    }
                },
                "image/jpeg", // MIME type
                0.95 // Quality (0.0 to 1.0)
            );
        } else {
            // Fallback: Use toDataURL if toBlob is not supported
            const dataURL = canvas.toDataURL("image/jpeg", 0.95);
            console.log("Image captured as data URL:", dataURL);

            // Convert data URL to Blob
            fetch(dataURL)
                .then((res) => res.blob())
                .then((blob) => {
                    const file = new File([blob], `${personName}.jpg`, { type: "image/jpeg" });
                    setCapturedImage(file);
                    console.log("Image converted to File object:", file);
                })
                .catch((error) => {
                    console.error("Failed to convert data URL to Blob:", error);
                });
        }
    };

    const uploadImage = async () => {
        if (!capturedImage || !personName.trim()) {
            alert("Please capture an image and enter a name first!");
            return;
        }

        const confirmUpload = window.confirm("Are you sure you want to upload this image?");
        if (!confirmUpload) return;

        setIsTraining(true);

        try {
            // Create FormData and append the file
            const formData = new FormData();
            formData.append("file", capturedImage);
            formData.append("name", personName);

            console.log("FormData content:", formData);

            // Upload the captured image
            await axios.post("/upload_face", formData, { //http://127.0.0.1:5000/upload_face
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            });
            alert("Image uploaded successfully! Training model...");

            // Train the model after uploading
            await axios.post("/train_model");
            alert("Model training completed! Video feed will update shortly.");
        } catch (error) {
            console.error("Error uploading image:", error);
            alert("Failed to upload image.");
        } finally {
            setIsTraining(false);
        }
    };

    return (
        <div>
            <h1>Face Recognition</h1>
            <img ref={videoRef} src="http://127.0.0.1:5000/video_feed" autoPlay playsInline width="640" height="480" />
            
            <h2>Recognized Faces:</h2>
            <ul>
                {recognizedFaces.map((face, index) => (
                    <li key={index}>{face.name} ({face.confidence}%)</li>
                ))}
            </ul>

            <h3>Allowed Status:</h3>
            {allowedStatus ? <h4>ALLOWED</h4> : <h4>NOT ALLOWED</h4>}

            <hr />
            <h2>Capture and Upload Image</h2>
            <input
                type="text"
                placeholder="Enter name"
                value={personName}
                onChange={(e) => setPersonName(e.target.value)}
            />
            <button onClick={captureImage}>Capture Image</button>

            {capturedImage && (
                <div>
                    <h3>Preview:</h3>
                    <img src={URL.createObjectURL(capturedImage)} alt="Captured" width="200" />
                    <button onClick={uploadImage} disabled={isTraining}>
                        {isTraining ? "Training..." : "Upload and Train"}
                    </button>
                </div>
            )}
        </div>
    );
}

export default App;