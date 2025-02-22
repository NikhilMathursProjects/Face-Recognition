import React, { useState, useEffect } from "react";
import axios from "axios";

function App() {
    const [recognizedFaces, setRecognizedFaces] = useState([]);

    useEffect(() => {
        const interval = setInterval(() => {
            axios.get("http://127.0.0.1:5000/recognize_face")
                .then(response => setRecognizedFaces(response.data.recognized_faces))
                .catch(error => console.log(error));
        }, 2000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div>
            <h1>Face Recognition</h1>
            <img src="http://127.0.0.1:5000/video_feed" alt="Video Stream" />
            <h2>Recognized Faces:</h2>
            <ul>
                {recognizedFaces.map((face, index) => (
                    <li key={index}>{face.name} ({face.confidence}%)</li>
                ))}
            </ul>
        </div>
    );
}

export default App;
