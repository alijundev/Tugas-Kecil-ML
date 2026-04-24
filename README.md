# Klasifikasi Tipe Kepribadian MBTI Menggunakan Machine Learning

Proyek ini memprediksi 1 dari **16 tipe kepribadian MBTI** berdasarkan karakteristik individu (usia, jenis kelamin, pendidikan, skor psikometri, dan minat) menggunakan algoritma Machine Learning.

**Performa Model Terbaik**: ≥90% (Random Forest Classifier)

---

## Struktur Direktori

```
Tugas-Kecil-ML/
├── data/
│   └── data.csv                # Dataset (Kaggle)
├── models/                     # Hasil export model (.pkl, di-generate dari notebook)
├── notebooks/
│   └── Tugas_ML_Project.ipynb  # Notebook utama (eksplorasi data s.d. kesimpulan)
├── app.py                      # Aplikasi web Streamlit
├── mbti_data.json              # Deskripsi 16 tipe MBTI
├── pyproject.toml              # Dependencies (uv)
└── README.md
```

---

## Cara Penggunaan

### 1. Menjalankan Notebook

1. Pastikan environment sudah aktif (`uv sync`)
2. Buka `notebooks/Tugas_ML_Project.ipynb` di Jupyter atau VSCode
3. Klik **Restart & Run All**
4. File model (`rf_juara.pkl` dan `scaler.pkl`) akan tersimpan di folder `models/`

### 2. Menjalankan Aplikasi Web

```bash
uv run streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501`. Masukkan data profil di sidebar dan klik **Prediksi MBTI**.

---

## Metodologi

1. **Data Understanding** — Eksplorasi dataset 43.744 responden dengan 9 fitur
2. **Data Preprocessing** — Penanganan duplikat, encoding (Label & One-Hot), standarisasi Z-Score
3. **EDA** — Distribusi target, heatmap korelasi
4. **Data Splitting** — 3 variasi rasio (70:30, 80:20, 90:10) dengan stratified sampling
5. **Model Building** — Logistic Regression, Decision Tree, Random Forest, XGBoost
6. **Hyperparameter Tuning** — GridSearchCV dan RandomizedSearchCV
7. **Evaluasi** — Classification report, confusion matrix, perbandingan akurasi

---

## Hasil

| Algoritma | Baseline | Setelah Tuning |
|:---|:---|:---|
| Logistic Regression | ~84% | ~85% |
| Decision Tree | ~86% | ~87% |
| **Random Forest** | **≥90%** | **≥90%** |

---

## Dependencies

- Python ≥ 3.14
- pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, streamlit, joblib