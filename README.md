# 🗑️ Sistem Deteksi Sampah Organik & Non-Organik

Sistem web profesional untuk klasifikasi sampah organik dan non-organik menggunakan Deep Learning (MobileNetV2) dengan fitur lengkap: authentication, dashboard, history, dan webcam detection.

Link Dataset : [Kaggle: Dataset Sampah](https://www.kaggle.com/datasets/eldadvikorian/dataset-sampah-organik-dan-anorganik)



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
   - File model harus ada di folder yang sama dengan `app.py`
   - Sudah termasuk dalam folder deployment ini

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

**Linux/Mac:**
```bash
python3 app.py
```

Atau double click `run.bat` (Windows)

### 4. Akses Aplikasi
- Buka browser: `http://localhost:5000`
- **Default Admin Account:**
  - Username: `admin`
  - Password: `admin123`
  - ⚠️ **PENTING:** Ganti password admin di production!

## 📁 Struktur Folder

```
deteksi_sampah_deploy/
├── app.py                 # Flask application (main)
├── models.py              # Database models
├── requirements.txt       # Dependencies
├── README.md             # Dokumentasi
├── run.bat               # Script untuk Windows
├── run.sh                # Script untuk Linux/Mac
├── waste_detection.db    # SQLite database (auto-generated)
├── templates/
│   ├── base.html         # Base template
│   ├── login.html        # Halaman login
│   ├── register.html     # Halaman register
│   ├── dashboard.html    # Dashboard utama
│   ├── detect.html       # Halaman deteksi
│   ├── history.html      # History deteksi
│   └── profile.html      # Profile user
├── static/
│   ├── css/
│   │   └── style.css       # Styling
│   │   └── style.css     # Styling (CalmGreen)
│   ├── js/
│   │   ├── main.js       # JavaScript umum
│   │   └── detect.js     # JavaScript deteksi
│   └── uploads/          # Folder untuk gambar upload
└── waste_classification_model.h5  # Model file
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
- **Tema:** ClamGreen Professional
- **Responsive Design** - Mobile-friendly
- **Modern UI/UX** - Animasi dan transisi smooth
- **Font Awesome Icons** - Icons profesional

## 🔧 Konfigurasi

### Mengubah Port
Edit di `app.py` baris terakhir:
```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Ubah port di sini
```

### Mengubah Secret Key
Edit di `app.py`:
```python
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
```

### Database
- Default: SQLite (`waste_detection.db`)
- Untuk production, ubah ke PostgreSQL/MySQL di `app.py`:
```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:pass@localhost/dbname'
```

## 🐛 Troubleshooting

### Model tidak ditemukan
- Pastikan file model ada di folder yang sama dengan `app.py`
- Cek nama file: `waste_classification_model.h5` atau `best_model.h5`

### Port sudah digunakan
- Ubah port di `app.py` atau tutup aplikasi yang menggunakan port 5000

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
2. Gunakan production WSGI server (Gunicorn, uWSGI)
3. Setup reverse proxy (Nginx)
4. Gunakan database production (PostgreSQL/MySQL)
5. Setup SSL/HTTPS
6. Ganti secret key dan password admin

## 👨‍💻 Developer

Sistem deteksi sampah dengan Deep Learning untuk klasifikasi organik dan non-organik.

---

**Selamat menggunakan! 🎉**

