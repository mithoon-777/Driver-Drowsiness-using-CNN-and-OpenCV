# Driver Drowsiness Detection System

## Project Overview
This project detects driver drowsiness in real-time using **OpenCV, Dlib, and Flask**. It tracks eye movements, calculates **Eye Aspect Ratio (EAR)**, and triggers a **beep sound alert** when drowsiness is detected.

---

## Project Structure
```
driver_drowsiness/
│── models/
│   ├── shape_predictor_68_face_landmarks.dat  
│── static/
│   ├── 700-hz-beeps-86815.mp3  
│   ├── style.css  
│── templates/
│   ├── index.html  
│── app.py  
│── requirements.txt  
│── README.md  
```

### File Explanations
- **`app.py`** – Main Flask backend for video processing and drowsiness detection.  
- **`detector.py`** – Handles face and eye landmark detection (if applicable).  
- **`models/shape_predictor_68_face_landmarks.dat`** – Dlib model for facial landmark detection.  
- **`static/700-hz-beeps-86815.mp3`** – Alert sound when drowsiness is detected.  
- **`static/style.css`** – CSS file for styling the front-end UI.  
- **`templates/index.html`** – Front-end page displaying the live video feed.  
- **`requirements.txt`** – List of required Python packages.  

---

## Installation and Setup
### 1. Install Dependencies
Ensure you have Python installed, then install the required packages:
```sh
pip install -r requirements.txt
```

### 2. Store Files Correctly
- Place **`shape_predictor_68_face_landmarks.dat`** inside the `models/` folder.  
- Place **`700-hz-beeps-86815.mp3`** inside the `static/` folder.  
- Place **`index.html`** inside the `templates/` folder.  

### 3. Run the Flask Application
```sh
python app.py
```
The application will start at:
```
http://127.0.0.1:5000/
```

---

## How It Works
1. The webcam captures real-time video.  
2. The system detects the face and eyes using **Dlib’s shape predictor**.  
3. The **EAR (Eye Aspect Ratio)** is calculated to determine drowsiness.  
4. If EAR falls below the threshold for a certain period:
   - **An alert message appears on the screen**  
   - **A beep sound plays**  
5. If the driver opens their eyes, the alert and beep stop immediately.  

---

## Output Preview
- **Normal State:**  
  ```
  Status: Awake 😊
  ```
- **Drowsy State:**  
  ```
  Status: Drowsy! 🚨
  ```
  - A red alert message appears on the screen.  
  - The beep sound plays as a warning.  

---

## Troubleshooting
1. **Dlib model not loading?**  
   - Ensure **`shape_predictor_68_face_landmarks.dat`** is inside the `models/` folder.  

2. **Beep sound not working?**  
   - Ensure **`700-hz-beeps-86815.mp3`** is inside the `static/` folder.  

3. **Webcam not detected?**  
   - Check your system permissions and allow camera access.  

---

## License
This project is open-source and free to use for research and learning purposes.
