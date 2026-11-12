# Klasifikasi Breast Cancer (Flask + PyTorch)

Aplikasi web sederhana untuk memprediksi tumor **jinak (benign)** atau **ganas (malignant)** dari fitur hasil
pengukuran sel (dataset Breast Cancer Wisconsin). Model berupa MLP PyTorch yang dilatih di Google Colab dan
disimpan sebagai `model/breast_cancer_model.pth`.

> Aplikasi ini untuk keperluan tugas kuliah, bukan alat diagnosis medis.

## Struktur proyek

```
.
├── app.py                      # aplikasi Flask (halaman UI + endpoint /api/predict)
├── model.py                    # arsitektur MLP dan pemuat file .pth
├── model/
│   └── breast_cancer_model.pth # model hasil training di Colab (letakkan di sini)
├── templates/index.html        # antarmuka pengguna
├── static/style.css
├── colab/
│   └── Training_PyTorch_Colab.ipynb   # notebook training; menghasilkan file .pth
├── tests/test_app.py           # uji API (tanpa PyTorch)
└── requirements.txt
```

## 1. Menghasilkan dan mengunduh model dari Google Colab

1. Buka <https://colab.research.google.com>, pilih **File > Upload notebook**, lalu unggah `colab/Training_PyTorch_Colab.ipynb`.
2. Jalankan semua sel (**Runtime > Run all**). Sel terakhir mengunduh `breast_cancer_model.pth` ke komputer Anda.
3. Pindahkan file tersebut ke folder `model/` pada proyek ini.

## 2. Menjalankan aplikasi

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Buka <http://127.0.0.1:5000>. Gunakan tombol **Isi contoh jinak / ganas** untuk mengisi form otomatis,
lalu tekan **Prediksi**.

Lokasi model dapat diubah dengan variabel lingkungan `MODEL_PATH`, dan port dengan `PORT`.

## 3. Menjalankan pengujian

```bash
python -m unittest discover -s tests
```

Pengujian memakai predictor palsu sehingga tidak membutuhkan PyTorch maupun file `.pth`.

## API

`POST /api/predict` dengan body JSON `{"features": {"mean radius": 14.1, ...}}` (semua fitur wajib, angka >= 0).
Respons berisi `label`, `probabilities` (benign dan malignant), `confidence`, dan `out_of_range`
(daftar fitur yang berada di luar rentang data latih). `GET /health` untuk cek status.

## Upload ke GitHub

```bash
git init
git add .
git commit -m "Aplikasi klasifikasi breast cancer (Flask + PyTorch)"
git branch -M main
git remote add origin https://github.com/<username>/<nama-repo>.git
git push -u origin main
```

Pastikan `model/breast_cancer_model.pth` ikut ter-commit (`git ls-files model` harus menampilkannya).
