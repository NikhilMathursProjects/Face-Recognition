import React, { useState, useEffect } from "react";
import axios from "axios";

function App() {
    const [recognizedFaces, setRecognizedFaces] = useState([]);
    const [allowedStatus, setAllowedStatus] = useState(false);
    useEffect(() => {
        const interval = setInterval(() => {
            axios.get("http://127.0.0.1:5000/recognize_face")
                .then(response => 
                    {const faces = response.data.recognized_faces;
                    setRecognizedFaces(faces);

                    //Check if there one face thats not unknown
                    const isAllowed = faces.some(face => face.name !== "Unknown");
                    setAllowedStatus(isAllowed);}   
            )
                .catch(error => console.log(error));
        }, 2000);
        
        return () => clearInterval(interval);
    }, []);
    
    // face=recognizedFaces.map(face);
    // recognizedFaces.forEach((faces) => {
    //     console.log(faces);
    //     if(faces!=='Unknown'){
    //         setAllowedStatus(true);
    //     }else{
    //         setAllowedStatus(false);
    //     }
    // });
    // recognizedFaces
    // if(recognizedFaces!=="Unknown"){
    //     setAllowedStatus(true);
    // }else{
    //     setAllowedStatus(false);
    // }
    


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
            <h3>Allowed Status:</h3>
            {allowedStatus?(<h4>ALLOWED</h4>):(<h4>NOT ALLOWED</h4>)}
        </div>
    );
}

export default App;
