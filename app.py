import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json

@st.cache_resource
def load_model():
    model = joblib.load('models/model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

@st.cache_data
def load_mbti_info():
    with open('mbti_data.json', 'r') as f:
        return json.load(f)

try:
    model, scaler = load_model()
except FileNotFoundError:
    st.error("File model (.pkl) tidak ditemukan di folder models/. Jalankan notebook terlebih dahulu untuk mengekspor model.")
    st.stop()

mbti_info = load_mbti_info()

st.set_page_config(page_title="Prediktor MBTI", page_icon="🧠", layout="centered")
st.title("🧠 Prediksi Tipe Kepribadian MBTI")
st.markdown("Masukkan data profil di sidebar, lalu klik tombol prediksi untuk melihat hasilnya.")

st.sidebar.header("📝 Profil Responden")
age = st.sidebar.number_input("Umur", min_value=15, max_value=80, value=22)
gender = st.sidebar.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
education = st.sidebar.selectbox("Pendidikan", ["SMA / S1 / Sederajat", "Pascasarjana (S2/S3)"])

st.sidebar.header("📊 Skor Psikometri (0.0 - 10.0)")
introversion = st.sidebar.slider("Introversion Score", 0.0, 10.0, 5.0)
sensing = st.sidebar.slider("Sensing Score", 0.0, 10.0, 5.0)
thinking = st.sidebar.slider("Thinking Score", 0.0, 10.0, 5.0)
judging = st.sidebar.slider("Judging Score", 0.0, 10.0, 5.0)

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
