# http://127.0.0.1:5000/
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf #type: ignore
import numpy as np
import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO #type: ignore

app = Flask(__name__)

labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(labels), activation='softmax', name='dense_1') 
])

try:
    model.load_weights('model_v1.keras', skip_mismatch=True)
    print("--- BERHASIL: Model Terpasang Tanpa Skipping! ---")
except Exception as e:
    print(f"Gagal load .keras, nyoba .h5... Info: {e}")
    model.load_weights('model_v1.h5', skip_mismatch=True, by_name=False)
    print("--- BERHASIL: Model Terpasang via .h5! ---")

# Ambil Mata YOLOv8
yolo_eye = YOLO('yolov8n.pt') 

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success: break
        
        # A. YOLO Cari Barang (conf diturunin biar lebih peka)
        results = yolo_eye.predict(frame, verbose=False, conf=0.3)
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls == 0: # JANGAN deteksi orang (Person)
                    continue 
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop_img = frame[y1:y2, x1:x2]
                
                if crop_img.size > 0:
                    # B. PREPROCESSING MobileNetV2
                    img = cv2.resize(crop_img, (224, 224))
                    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # C. Prediksi
                    prediction = model.predict(img_array, verbose=0)
                    idx = np.argmax(prediction)
                    conf = np.max(prediction) * 100
                    
                    # D. Tampilan Boxing + Label
                    if conf > 25: 
                        label_text = f"{labels[idx]} ({conf:.1f}%)"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, port=5000)