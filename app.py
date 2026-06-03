import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from PIL import Image

app = FastAPI()

# 1. LOAD MODEL .H5 YANG SUDAH JADI
# Pastikan file h5 ini sudah kamu pindahkan dari Google Drive ke folder VPS yang sama dengan app.py
MODEL_PATH = "model_klasifikasi_makanan.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# 2. DAFTAR KELAS (Urutan HARUS persis sama dengan urutan folder saat training di Colab)
# Silakan ganti isi list di bawah ini dengan nama-nama kelas makanan yang ada di dataset kamu dulu
CLASS_NAMES = [
    'Apple Braeburn', 'Apple Granny Smith', 'Apricot', 'Avocado',
    'Banana', 'Blueberry', 'Cactus fruit', 'Cantaloupe',
    'Cherry', 'Clementine', 'Corn', 'Cucumber Ripe',
    'Grape Blue', 'Kiwi', 'Lemon', 'Limes',
    'Mango', 'Onion White', 'Orange', 'Papaya',
    'Passion Fruit', 'Peach', 'Pear', 'Pepper Green',
    'Pepper Red', 'Pineapple', 'Plum', 'Pomegranate',
    'Potato Red', 'Raspberry', 'Strawberry', 'Tomato',
    'Watermelon'
];

# Fungsi pembantu untuk memproses gambar mentah sebelum dimasukkan ke model CNN
def prepare_image(image_bytes):
    # Membuka gambar dari bytes stream dan memastikan formatnya RGB (3 channel)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Menyesuaikan ukuran gambar menjadi 224x224 sesuai konfigurasi IMG_SIZE di Colab kamu
    img = img.resize((224, 224))
    
    # Mengubah gambar menjadi array angka numpy
    img_array = np.array(img, dtype=np.float32)
    
    # Menambahkan dimensi batch di depan, dari (224, 224, 3) menjadi (1, 224, 224, 3)
    # Karena model CNN sekuensial selalu meminta format [Batch_Size, Height, Width, Channels]
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.get("/")
def home():
    return {"status": "AI Model Server is Running!"}

# Endpoint inilah yang nantinya akan ditembak/dipanggil oleh Express.js kamu
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Membaca file gambar yang dikirim oleh Express.js
        image_bytes = await file.read()
        
        # Jalankan fungsi preprocess gambar
        processed_image = prepare_image(image_bytes)
        
        # Proses Prediksi (Menebak gambar)
        # Catatan: Di kodingan Colab kamu sudah ada layers.Rescaling(1./255) di dalam model,
        # jadi di sini kita TIDAK PERLU membagi angka array dengan 255 lagi lewat Python biasa.
        predictions = model.predict(processed_image)
        
        # predictions menghasilkan array probabilitas, contoh: [0.1, 0.8, 0.05, 0.05]
        # Kita ambil index dengan probabilitas tertinggi menggunakan np.argmax
        highest_probability_index = np.argmax(predictions[0])
        
        # Ambil nama kelas berdasarkan index terbaik tersebut
        predicted_class = CLASS_NAMES[highest_probability_index]
        
        # Ambil skor tingkat kepercayaan (confidence score) dalam persen
        confidence = float(predictions[0][highest_probability_index]) * 100

        # Kembalikan hasil tebakan ke Express.js dalam bentuk JSON
        return {
            "success": True,
            "class": predicted_class,
            "confidence": f"{confidence:.2f}%"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}