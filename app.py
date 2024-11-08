from flask import Flask, render_template, Response, jsonify
import cv2
from fer import FER
import numpy as np
import mediapipe as mp
from math import sqrt

# Initializing Flask app
app = Flask(__name__)

# Global variables
camera_running = False
capture = None
current_engagement_status = "Not Engaged"
current_dominant_emotion = "None"
distraction_status = "Not Distracted"
posture_status = "Unknown"

# Initializing emotion detector
emotion_detector = FER()

# Loading YOLO for object detection (distraction)
yolo_weights = "model/yolov4.weights"
yolo_cfg = "model/yolov4.cfg"
coco_names = "model/coco.names"
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)

# Loading COCO class names
with open(coco_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initializing MediaPipe pose for posture detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Posture thresholds
HEAD_ALIGNMENT_THRESHOLD = 0.1
SHOULDER_TILT_THRESHOLD = 0.05
MIN_HEAD_SHOULDER_DISTANCE_THRESHOLD = 0.15

def euclidean_distance(point1, point2):
    return sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def classify_posture(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]

    shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
    head_offset_x = abs(nose.x - shoulder_mid_x)
    shoulder_tilt = abs(left_shoulder.y - right_shoulder.y)
    left_head_distance = euclidean_distance(nose, left_shoulder)
    right_head_distance = euclidean_distance(nose, right_shoulder)

    if head_offset_x > HEAD_ALIGNMENT_THRESHOLD:
        return "Bad Posture (Head Forward)"
    elif shoulder_tilt > SHOULDER_TILT_THRESHOLD:
        return "Bad Posture (Shoulder Tilt)"
    elif left_head_distance < MIN_HEAD_SHOULDER_DISTANCE_THRESHOLD or right_head_distance < MIN_HEAD_SHOULDER_DISTANCE_THRESHOLD:
        return "Bad Posture (Too Close)"
    else:
        return "Good Posture"

# Starting camera function with all detections
def start_camera():
    global capture, camera_running, current_engagement_status, current_dominant_emotion, distraction_status, posture_status
    capture = cv2.VideoCapture(0)
    camera_running = True

    while camera_running:
        ret, frame = capture.read()
        if not ret:
            break

        # Emotion detection
        emotions = emotion_detector.detect_emotions(frame)
        current_engagement_status = "Not Engaged"
        current_dominant_emotion = "None"

        if emotions:
            current_engagement_status = "Engaged"
            for emotion in emotions:
                bbox = emotion['box']
                dominant_emotion = max(emotion['emotions'], key=emotion['emotions'].get)
                dominant_score = emotion['emotions'][dominant_emotion] * 100
                current_dominant_emotion = f"{dominant_emotion.capitalize()} ({dominant_score:.1f}%)"
                
                cv2.rectangle(frame, (bbox[0], bbox[1]), 
                              (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                              (217, 24, 242), 2)
                cv2.putText(frame, current_dominant_emotion, (bbox[0], bbox[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (217, 24, 242), 1)

        # Distraction detection
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        distraction_detected = False
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] in ["cell phone", "laptop", "book"]:
                    distraction_detected = True
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, f"{classes[class_id]} ({confidence * 100:.1f}%)", 
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        distraction_status = "Distracted" if distraction_detected else "Not Distracted"

        # Posture detection with key points
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            posture_status = classify_posture(landmarks)

            # Draw key points
            key_points = [
                mp_pose.PoseLandmark.NOSE,
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_EYE,
                mp_pose.PoseLandmark.RIGHT_EYE,
                mp_pose.PoseLandmark.LEFT_EAR,
                mp_pose.PoseLandmark.RIGHT_EAR
            ]
            
            for point in key_points:
                landmark = landmarks[point]
                cv2.circle(frame, (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])), 5, (0, 255, 0), -1)

            # Draw line between shoulders
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            cv2.line(frame, (int(left_shoulder.x * frame.shape[1]), int(left_shoulder.y * frame.shape[0])),
                     (int(right_shoulder.x * frame.shape[1]), int(right_shoulder.y * frame.shape[0])), (14, 237, 196), 2)

        # Display statuses on the frame
        cv2.putText(frame, f"Engagement Status: {current_engagement_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (217, 24, 242), 2)
        cv2.putText(frame, f"Distraction Status: {distraction_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Posture Status: {posture_status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (14, 237, 196), 2)

        # Encode frame for streaming
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/start_camera')
def start_camera_stream():
    return Response(start_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_running, capture
    camera_running = False
    if capture:
        capture.release()
    return jsonify({"status": "Camera stopped"})

@app.route('/status')
def get_status():
    return jsonify({
        "engagement_status": current_engagement_status,
        "dominant_emotion": current_dominant_emotion,
        "distraction_status": distraction_status,
        "posture_status": posture_status
    })

@app.route('/')
def index():
    return render_template('appindex.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
