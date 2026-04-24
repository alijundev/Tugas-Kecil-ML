# Klasifikasi Tipe Kepribadian MBTI Menggunakan Machine Learning

Proyek tugas kecil mata kuliah Machine Learning. Menggunakan data psikometri dari Kaggle untuk memprediksi 1 dari 16 tipe kepribadian MBTI (multiclass classification).

Dataset: [Predict People Personality Types](https://www.kaggle.com/datasets/stealthtechnologies/predict-people-personality-types) — 43.744 responden, 9 fitur.

Akurasi terbaik: **≥90%** dengan Random Forest Classifier.

---

## Struktur Proyek

```
├── data/
│   └── data.csv               # Dataset
├── models/                     # Folder output model (.pkl), di-generate dari notebook
├── notebooks/
│   └── Tugas_ML_Project.ipynb  # Notebook utama
├── app.py                      # Aplikasi prediksi (Streamlit)
├── mbti_data.json              # Data deskripsi 16 tipe MBTI
├── pyproject.toml              # Daftar dependencies
└── README.md
```

---

## Prasyarat

Proyek ini menggunakan [uv](https://docs.astral.sh/uv/) sebagai package manager Python.

### Install uv

**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Setelah install, pastikan `uv` sudah terdaftar di terminal:
```bash
uv --version
```

---

## Cara Menjalankan

### 1. Clone dan install dependencies

```bash
git clone https://github.com/alijundev/Tugas-Kecil-ML.git
cd Tugas-Kecil-ML
uv sync
```

### 2. Jalankan Notebook

Buka file `notebooks/Tugas_ML_Project.ipynb` di Jupyter atau VSCode, lalu klik **Restart & Run All**.

Notebook akan melakukan seluruh proses mulai dari load data, preprocessing, training model, evaluasi, sampai export file `.pkl` ke folder `models/`.

### 3. Jalankan Aplikasi Web

Setelah notebook selesai dijalankan (file `.pkl` sudah ada di `models/`):

```bash
uv run streamlit run app.py
```

Buka `http://localhost:8501` di browser, isi data di sidebar, dan klik tombol prediksi.

---

## Pipeline

| Tahap | Keterangan |
|:---:|:---|
| 1 | Data Understanding — eksplorasi dataset, cek distribusi, tipe data |
| 2 | Preprocessing — hapus duplikat, encoding, standarisasi Z-Score |
| 3 | EDA — distribusi target, heatmap korelasi |
| 4 | Splitting — 3 variasi rasio (70:30, 80:20, 90:10) dengan stratified sampling |
| 5 | Model Building — Logistic Regression, Decision Tree, Random Forest, XGBoost |
| 6 | Hyperparameter Tuning — GridSearchCV, RandomizedSearchCV |
| 7 | Evaluasi — classification report, confusion matrix |
| 8 | Perbandingan — tabel akurasi sebelum vs sesudah tuning |

---

## Hasil Perbandingan Akurasi

| Algoritma | Sebelum Tuning | Sesudah Tuning |
|:---|:---:|:---:|
| Logistic Regression | ~84% | ~85% |
| Decision Tree | ~86% | ~87% |
| **Random Forest** | **≥90%** | **≥90%** |

---

## Tech Stack

- Python 3.14
- scikit-learn, XGBoost
- pandas, NumPy
- Matplotlib, Seaborn
- Streamlit
- uv (package manager)