import os
import sys
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import csv
import numpy as np
import tensorflow as tf # type: ignore
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO # type: ignore
from datetime import datetime

app = Flask(__name__)

labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Log
log_file = os.path.join('outputs', 'waste_log.csv')
os.makedirs('outputs', exist_ok=True)
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'date', 'label', 'confidence'])

# Cooldown
last_log_time = {label: 0 for label in labels}
LOG_COOLDOWN = 5.0 

base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
base_model.trainable = False
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
outputs = tf.keras.layers.Dense(len(labels), activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

try:
    model.load_weights(os.path.join('models', 'model.weights.h5'))
    print("AI Model & Weights Berhasil Terpasang!")
except Exception as e:
    print(f"Gagal memuat dikarenakan error: {e}")
    sys.exit()

yolo_eye = YOLO(os.path.join('models', 'yolov8n.pt'))

def log_detection(label, conf):
    now = datetime.now()
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([now.strftime("%H:%M:%S"), now.strftime("%Y-%m-%d"), label, f"{conf:.2f}"])

def generate_frames():
    camera = cv2.VideoCapture(0)
    frame_skip = 3
    frame_count = 0
    last_detections = []

    while True:
        success, frame = camera.read()
        if not success: break
        frame_count += 1

        if frame_count % frame_skip == 0:
            last_detections = []
        
            results = yolo_eye.predict(frame, verbose=False, conf=0.3)

            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    if cls == 0: continue 
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop_img = frame[y1:y2, x1:x2]

                    if crop_img.size > 0:
                        img = cv2.resize(crop_img, (224, 224))
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_rgb)
                        img_array = np.expand_dims(img_array, axis=0)

                        prediction = model.predict(img_array, verbose=0)[0]
                        idx = np.argmax(prediction)
                        conf = prediction[idx] * 100
                        main_label = labels[idx]

                        if conf > 30: 
                            current_time = time.time()
                            
                            if (current_time - last_log_time[main_label]) > LOG_COOLDOWN:
                                log_detection(main_label, conf)
                                last_log_time[main_label] = current_time

                            label_text = f"{main_label} ({conf:.1f}%)"
                            last_detections.append((x1, y1, x2, y2, label_text))

        for det in last_detections:
            x1, y1, x2, y2, label_text = det
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/chart_data')
def chart_data():
    counts = {label: 0 for label in labels}
    with open(log_file, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['label'] in counts:
                counts[row['label']] += 1
    return jsonify({'labels': list(counts.keys()), 'values': list(counts.values())})

@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/video_feed')
def video_feed(): 
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, port=5000)