import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import os
import gdown

# --- CONFIGURASI GOOGLE DRIVE ---
MODEL_FILE_ID = '14-qD4-i4uRSyO2AlkpVCC2CeRApEHVAk'
SCALER_FILE_ID = '1QLXwW9ckYbBoFmNYIgoghVRKUOZvB2q7'

@st.cache_resource
def load_model():
    # Buat folder models jika belum ada
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_path = 'models/model.pkl'
    scaler_path = 'models/scaler.pkl'

    # Download Model jika belum ada
    if not os.path.exists(model_path):
        with st.spinner("Mengunduh model (400MB)... Mohon tunggu, ini hanya dilakukan sekali."):
            url = f'https://drive.google.com/uc?id={MODEL_FILE_ID}'
            gdown.download(url, model_path, quiet=False)

    # Download Scaler jika belum ada
    if not os.path.exists(scaler_path):
        url = f'https://drive.google.com/uc?id={SCALER_FILE_ID}'
        gdown.download(url, scaler_path, quiet=False)

    # Load file
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

@st.cache_data
def load_mbti_info():
    # Pastikan file ini ada di repo GitHub kamu karena ukurannya kecil
    with open('mbti_data.json', 'r') as f:
        return json.load(f)

# --- UI STREAMLIT ---
st.set_page_config(page_title="Prediktor MBTI", page_icon="🧠", layout="centered")

# Load model dan data
try:
    model, scaler = load_model()
    mbti_info = load_mbti_info()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()

st.title("🧠 Prediksi Tipe Kepribadian MBTI")
st.markdown("Masukkan data profil di sidebar, lalu klik tombol prediksi untuk melihat hasilnya.")

# ... (Sisanya sama dengan code kamu sebelumnya) ...

st.sidebar.header("📝 Profil Responden")
age = st.sidebar.number_input("Umur", min_value=15, max_value=80, value=22)
gender = st.sidebar.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
education = st.sidebar.selectbox("Pendidikan", ["SMA / S1 / Sederajat", "Pascasarjana (S2/S3)"])

st.sidebar.header("📊 Skor Psikometri (0.0 - 10.0)")

# Menggunakan format (0: Lawan ↔ 10: Utama) agar arah slider jelas
introversion = st.sidebar.slider(
    "Introversion (0: Ekstrovert ↔ 10: Introvert)", 
    0.0, 10.0, 5.0,
    help="Skor rendah cenderung Ekstrovert, skor tinggi cenderung Introvert."
)

sensing = st.sidebar.slider(
    "Sensing (0: Intuisi ↔ 10: Sensing)", 
    0.0, 10.0, 5.0,
    help="Skor rendah cenderung menggunakan Intuisi (N), skor tinggi menggunakan Sensorik (S)."
)

thinking = st.sidebar.slider(
    "Thinking (0: Perasaan ↔ 10: Berpikir)", 
    0.0, 10.0, 5.0,
    help="Skor rendah cenderung menggunakan Perasaan (Feeling), skor tinggi menggunakan Logika (Thinking)."
)

judging = st.sidebar.slider(
    "Judging (0: Spontan ↔ 10: Terencana)", 
    0.0, 10.0, 5.0,
    help="Skor rendah cenderung Spontan (Perceiving), skor tinggi cenderung Terencana (Judging)."
)

st.sidebar.header("🎯 Minat Utama")
interest = st.sidebar.selectbox("Pilih Minat", ["Sports", "Arts", "Technology", "Others", "Unknown"])

if st.button("Prediksi MBTI", use_container_width=True):
    gender_val = 1 if gender == "Laki-laki" else 0
    edu_val = 1 if education == "Pascasarjana (S2/S3)" else 0

    int_others = 1 if interest == "Others" else 0
    int_sports = 1 if interest == "Sports" else 0
    int_tech = 1 if interest == "Technology" else 0
    int_unk = 1 if interest == "Unknown" else 0

    data_baru = pd.DataFrame([{
        'Age': age,
        'Gender': gender_val,
        'Education': edu_val,
        'Introversion Score': introversion,
        'Sensing Score': sensing,
        'Thinking Score': thinking,
        'Judging Score': judging,
        'Interest_Others': int_others,
        'Interest_Sports': int_sports,
        'Interest_Technology': int_tech,
        'Interest_Unknown': int_unk
    }])

    kolom_numerik = ['Age', 'Introversion Score', 'Sensing Score', 'Thinking Score', 'Judging Score']
    data_baru[kolom_numerik] = scaler.transform(data_baru[kolom_numerik])

    hasil = model.predict(data_baru)
    tipe = hasil[0]

    st.balloons()
    st.success(f"### Hasil Prediksi: **{tipe}**")

    if tipe in mbti_info:
        info = mbti_info[tipe]
        st.markdown(f"## {tipe} — *{info['julukan']}*")
        st.markdown(info['deskripsi'])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ Kekuatan:**")
            st.markdown(info['kekuatan'])
        with col2:
            st.markdown("**⚠️ Kelemahan:**")
            st.markdown(info['kelemahan'])

st.write("---")
st.caption("Random Forest Classifier — Akurasi ≥90%")