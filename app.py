from flask import Flask, render_template, Response, jsonify
import os
import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance
import threading
import playsound  

app = Flask(__name__)

detector = dlib.get_frontal_face_detector()
predictor_path = os.path.join(os.getcwd(), "models", "shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(predictor_path)

beep_sound = os.path.join(os.getcwd(), "static", "700-hz-beeps-86815.mp3")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 15  
counter = 0
drowsy = False
alert_triggered = False  

cap = cv2.VideoCapture(0)

def play_alert():
    global alert_triggered
    if not alert_triggered: 
        alert_triggered = True
        playsound.playsound(beep_sound)

def generate_frames():
    global counter, drowsy, alert_triggered
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.resize(frame, (720, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[42:48]
            rightEye = shape[36:42]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

        
            cv2.polylines(frame, [leftEye], True, (0, 255, 0), 2)
            cv2.polylines(frame, [rightEye], True, (0, 255, 0), 2)

            if ear < EYE_AR_THRESH:
                counter += 1
                if counter >= EYE_AR_CONSEC_FRAMES:
                    drowsy = True
                    cv2.putText(frame, "DROWSINESS ALERT!", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                   
                    threading.Thread(target=play_alert).start()
            else:
                counter = 0
                drowsy = False
                alert_triggered = False  

            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/drowsiness_status')
def drowsiness_status():
    return jsonify({'drowsy': drowsy})

if __name__ == '__main__':
    app.run(debug=True)
