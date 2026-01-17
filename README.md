# 🗑️ Sistem Deteksi Sampah Organik & Non-Organik

Aplikasi web untuk klasifikasi sampah organik dan non-organik menggunakan Deep Learning (MobileNetV2). Fitur lengkap: autentikasi, dashboard, riwayat deteksi, dan deteksi via webcam.

## ✨ Fitur Utama

### 🔐 Authentication System

- **Login/Register** - Sistem autentikasi lengkap dengan validasi
- **Session Management** - Manajemen session user
- **Password Hashing** - Keamanan password dengan Werkzeug

### 📊 Dashboard

- **Statistik User** - Total deteksi, organik, non-organik
- **Statistik Sistem** - Untuk admin (total users, total deteksi)
- **Recent Detections** - 5 deteksi terbaru
- **Quick Actions** - Tombol cepat ke fitur utama

### 🎯 Deteksi Sampah

- **Upload Gambar** - Drag & drop atau klik untuk upload
- **Webcam Detection** - Deteksi real-time menggunakan webcam
- **Preprocessing** - Resize, normalisasi, noise reduction otomatis
- **Confidence Score** - Menampilkan tingkat keyakinan prediksi
- **Processing Time** - Waktu processing ditampilkan

### 📜 History

- **Riwayat Lengkap** - Semua deteksi tersimpan di database
- **Pagination** - Navigasi halaman untuk history panjang
- **Filter** - Filter berdasarkan jenis sampah
- **Image Preview** - Preview gambar dengan modal

### 👤 Profile Management

- **Edit Profile** - Update nama dan email
- **Change Password** - Ubah password dengan validasi
- **User Stats** - Statistik personal user

### 🗄️ Database

- **SQLite Database** - Database lokal untuk development
- **User Management** - CRUD user
- **Detection History** - Penyimpanan semua hasil deteksi
- **System Statistics** - Statistik sistem otomatis

## 📋 Persyaratan

1. **Python 3.8+**
2. **Model file** (`waste_classification_model.h5` atau `best_model.h5`)

- File model harus ada di folder yang sama dengan `app.py` (sudah tersedia di folder utama)
- Model juga tersedia di folder `notebook/` untuk eksperimen

## 🚀 Instalasi

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Pastikan Model File Ada

- Model harus ada di folder yang sama dengan `app.py`
- File: `waste_classification_model.h5` atau `best_model.h5`

### 3. Jalankan Aplikasi

**Windows:**

```bash
python app.py
```

### 4. Akses Aplikasi

- Buka browser: `http://localhost:5000`
- **Default Admin Account:**
  - Username: `admin`
  - Password: `admin123`
  - ⚠️ **PENTING:** Ganti password admin di production!

## 📁 Struktur Folder

```
.
├── app.py                  # Main Flask app
├── models.py               # Database models
├── requirements.txt        # Dependencies
├── README.md               # Dokumentasi
├── best_model.h5           # Model file
├── waste_classification_model.h5   # Model file
├── yolov8n.pt              # YOLOv8 model (opsional)
├── static/
│   ├── css/
│   │   └── style.css       # Styling
│   │   └── style.css     # Styling (CalmGreen)
│   ├── js/
│   │   ├── main.js         # JS umum
│   │   └── detect.js       # JS deteksi
│   └── uploads/            # Folder upload gambar
├── templates/
│   ├── base.html
│   ├── dashboard.html
│   ├── detect.html
│   ├── history.html
│   ├── history_detail.html
│   ├── login.html
│   ├── profile.html
│   ├── register.html
│   └── admin/
│       ├── dashboard.html
│       ├── history.html
│       └── users.html
├── instance/
│   └── (folder instance Flask)
├── notebook/
│   ├── Klasifikasi_Sampah_Organik_NonOrganik.ipynb
│   ├── best_model.h5
│   ├── waste_classification_model.h5
│   ├── yolov8n.pt
│   └── dataset_sampah/
│       └── DATASET/
│           └── TRAIN/TEST/O/R
└── __pycache__/
```

## 🔄 Flowchart Sistem

```
Start
  ↓
Login/Register
  ↓
Validasi Kredensial
  ↓ (Jika berhasil)
Dashboard
  ↓
Deteksi (Upload/Webcam)
  ↓
Preprocessing (Resize, Normalisasi, Noise Reduction)
  ↓
Deteksi Objek & Klasifikasi
  ↓
Decision: Organik atau Non-Organik
  ↓
Simpan ke Database
  ↓
Tampilkan Hasil (Label, Confidence, Processing Time)
  ↓
History & Statistik
```

## 🎨 Tema & Design

- **Tema:** ClamGreen Professional
- **Responsive Design** - Mobile-friendly
- **Modern UI/UX** - Animasi dan transisi smooth
- **Font Awesome Icons** - Icons profesional

## 🔧 Konfigurasi

### Mengubah Port

Edit di bagian akhir `app.py`:

```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Ubah port di sini
```

### Mengubah Secret Key

Edit di bagian awal `app.py`:

```python
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
```

### Database

- Default: SQLite (`waste_detection.db`, auto-generated)
- Untuk production, ubah ke PostgreSQL/MySQL di `app.py`:

```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:pass@localhost/dbname'
```

## 🐛 Troubleshooting

### Model tidak ditemukan

- Pastikan file model ada di folder yang sama dengan `app.py` (atau di notebook untuk eksperimen)
- Nama file: `waste_classification_model.h5` atau `best_model.h5`

### Port sudah digunakan

- Ubah port di `app.py` atau tutup aplikasi lain yang menggunakan port 5000

### Webcam tidak bekerja

- Pastikan izin kamera sudah diberikan di browser
- Gunakan HTTPS untuk webcam (atau localhost)

### Database error

- Hapus file `waste_detection.db` dan restart aplikasi
- Database akan dibuat otomatis

## 📝 Catatan Penting

1. **Security:** Ganti `SECRET_KEY` dan password admin di production
2. **Model:** Menggunakan MobileNetV2 dengan transfer learning
3. **Input:** Gambar akan di-resize ke 224x224 pixels
4. **Format:** Mendukung PNG, JPG, JPEG, GIF, WEBP (max 16MB)

## 🚀 Production Deployment

Untuk production:

1. Set `debug=False` di `app.py`
2. Gunakan WSGI server (Gunicorn/uWSGI)
3. Setup reverse proxy (Nginx/Apache)
4. Gunakan database production (PostgreSQL/MySQL)
5. Setup SSL/HTTPS
6. Ganti secret key dan password admin

## 👨‍💻 Developer

Sistem deteksi sampah dengan Deep Learning untuk klasifikasi organik dan non-organik.

---

**Selamat menggunakan! 🎉**
