# Garbage Classification: Smart Vision for Waste Management
Proyek ini merupakan pengaplikasian dari Computer Vision System yang dirancang untuk mengklasifikasikan 6 kategori sampah secara otomatis melalui kamera dengan memanfaatkan arsitektur Deep Learning yang dioptimasi untuk mengidentifikasi material secara real-time.

---

## Dataset Overview
Data yang digunakan berasal dari folder **`dataset`** yang bersumber dari Kaggle dengan total data gambar sebanyak 2527 gambar yang terbagi menjadi 6 kategori utama:
- `Carboard`
- `Glass`
- `Metal`
- `Paper`
- `Plastic`
- `Trash`

---

## Fitur Utama
- Real-time Recognition: Pengenalan objek sampah secara langsung melalui integrasi kamera.
- High-Precision Model: Menggunakan arsitektur MobileNetV2 yang telah melalui tahap Fine-Tuning mendalam.
- Intelligent Dashboard: Visualisasi persentase keyakinan model (probability score) menggunakan grafik interaktif.
- Deployment: Arsitektur ringan untuk penggunaan pada aplikasi berbasis web.

---

## Pengembangan Model
- Transfer Learning: Implementasi bobot pre-trained dari arsitektur MobileNetV2 untuk ekstraksi fitur tingkat tinggi.
- Aggressive Augmentation: Melatih sistem dengan manipulasi data (rotasi, zoom, flip) guna meningkatkan ketahanan terhadap variasi visual di lapangan.
- Fine-Tuning: Melakukan optimasi pada layer spesifik (100+) untuk menyesuaikan model dengan karakteristik unik material sampah.
- Learning Rate Optimization: Penerapan Exponential Decay untuk memastikan stabilitas model saat mencapai titik akurasi maksimal.

---

## Arsitektur Aplikasi
- Deep Learning Framework: TensorFlow & Keras.
- Web Backend: Flask
- Frontend: HTML, CSS, & JavaScript
- Data Visualization: Chart.js

---

## Cara Menjalankan Proyek
1. Persiapan Environment
- Buka terminal di dalam folder proyek kamu
- Aktifkan virtual environment

3. Instalasi Library
`pip install -r requirements.txt` Ini akan otomatis mengunduh TensorFlow, Flask, dan library lainnya.

4. Menjalankan Server
- Jalankan perintah: `python app.py`
- Tunggu sampai muncul tulisan Running on http://127.0.0.1:5000.

5. Akses Dashboard
Buka browser dan masuk ke alamat `http://127.0.0.1:5000` 
