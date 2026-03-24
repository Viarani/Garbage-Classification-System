import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import tensorflow as tf #type: ignore
from flask import Flask, render_template, Response
from ultralytics import YOLO #type: ignore

app = Flask(__name__)

labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

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
    print("Arsitektur API + Bobot Berhasil Terpasang!")
except Exception as e:
    print(f"Gagal memuat dikarenakan error: {e}")
    sys.exit()

# YOLO
yolo_eye = YOLO('yolov8n.pt') 

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success: break
        
        results = yolo_eye.predict(frame, verbose=False, conf=0.3)
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls == 0: continue 
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop_img = frame[y1:y2, x1:x2]
                
                if crop_img.size > 0:
                    img = cv2.resize(crop_img, (224, 224))
                    # Convert BGR ke RGB biar tebakan gak ngaco
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_rgb)
                    img_array = np.expand_dims(img_array, axis=0)

                    prediction = model.predict(img_array, verbose=0)
                    idx = np.argmax(prediction)
                    conf = np.max(prediction) * 100

                    if conf > 30: 
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