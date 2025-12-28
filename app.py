"""
Flask Application untuk Sistem Deteksi Sampah Organik dan Non-Organik
Sistem lengkap dengan authentication, dashboard, dan history
"""

import os
import time
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, Response, send_file
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash
import tensorflow as tf
from tensorflow import keras
import cv2
from functools import wraps
from datetime import datetime, timedelta
import base64

# YOLO untuk bounding box
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO tidak tersedia. Install dengan: pip install ultralytics")

# Import models (harus setelah db.init_app)
try:
    from models import db, User, Detection, SystemStats
except ImportError:
    # Jika import gagal, definisikan di sini
    from flask_sqlalchemy import SQLAlchemy
    from werkzeug.security import generate_password_hash, check_password_hash
    from datetime import datetime
    
    db = SQLAlchemy()
    
    class User(db.Model):
        __tablename__ = 'users'
        id = db.Column(db.Integer, primary_key=True)
        username = db.Column(db.String(80), unique=True, nullable=False)
        email = db.Column(db.String(120), unique=True, nullable=False)
        password_hash = db.Column(db.String(255), nullable=False)
        full_name = db.Column(db.String(100))
        created_at = db.Column(db.DateTime, default=datetime.utcnow)
        is_admin = db.Column(db.Boolean, default=False)
        detections = db.relationship('Detection', backref='user', lazy=True, cascade='all, delete-orphan')
        
        def set_password(self, password):
            self.password_hash = generate_password_hash(password)
        
        def check_password(self, password):
            return check_password_hash(self.password_hash, password)
    
    class Detection(db.Model):
        __tablename__ = 'detections'
        id = db.Column(db.Integer, primary_key=True)
        user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
        predicted_class = db.Column(db.String(50), nullable=False)
        confidence = db.Column(db.Float, nullable=False)
        image_path = db.Column(db.String(255))
        image_size = db.Column(db.String(50))
        processing_time = db.Column(db.Float)
        created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    class SystemStats(db.Model):
        __tablename__ = 'system_stats'
        id = db.Column(db.Integer, primary_key=True)
        total_detections = db.Column(db.Integer, default=0)
        total_organic = db.Column(db.Integer, default=0)
        total_non_organic = db.Column(db.Integer, default=0)
        total_users = db.Column(db.Integer, default=0)
        last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        @classmethod
        def get_stats(cls):
            stats = cls.query.first()
            if not stats:
                stats = cls()
                db.session.add(stats)
                db.session.commit()
            return stats
        
        def update_stats(self):
            self.total_detections = Detection.query.count()
            self.total_organic = Detection.query.filter_by(predicted_class='Organik').count()
            self.total_non_organic = Detection.query.filter_by(predicted_class='Non-Organik').count()
            self.total_users = User.query.count()
            self.last_updated = datetime.utcnow()
            db.session.commit()

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///waste_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Initialize database
db.init_app(app)

# Buat folder uploads jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/uploads', exist_ok=True)

# Load model
MODEL_PATH = 'waste_classification_model.h5'
FALLBACK_MODEL = 'best_model.h5'

model = None
yolo_model = None
IMG_SIZE = 224

def load_model():
    """Load model klasifikasi dan YOLO"""
    global model, yolo_model
    try:
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            print(f"✅ Model loaded dari {MODEL_PATH}")
        elif os.path.exists(FALLBACK_MODEL):
            model = keras.models.load_model(FALLBACK_MODEL)
            print(f"✅ Model loaded dari {FALLBACK_MODEL}")
        else:
            print("❌ Error: Model tidak ditemukan!")
            return False
        
        # Load YOLO model untuk bounding box
        if YOLO_AVAILABLE:
            try:
                yolo_model = YOLO('yolov8n.pt')
                print("✅ YOLO model loaded untuk bounding box detection")
            except Exception as e:
                print(f"⚠️ YOLO model akan di-download otomatis saat pertama digunakan: {e}")
                yolo_model = None
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def allowed_file(filename):
    """Cek apakah file extension diizinkan"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    """Preprocess gambar untuk prediksi"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert BGR ke RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Normalisasi
        img = img.astype('float32') / 255.0
        
        # Expand dimensions
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"Error preprocessing: {e}")
        return None

# Informasi detail tentang jenis sampah
WASTE_INFO = {
    'Organik': {
        'name': 'Sampah Organik',
        'description': 'Sampah yang berasal dari makhluk hidup dan dapat terurai secara alami oleh mikroorganisme.',
        'examples': ['Sisa makanan', 'Daun-daunan', 'Kulit buah', 'Sayuran busuk', 'Kotoran hewan'],
        'recyclable': False,
        'recycling_info': 'Tidak dapat didaur ulang secara konvensional, namun dapat diolah menjadi kompos atau pupuk organik melalui proses pengomposan.',
        'disposal_tips': [
            'Pisahkan dari sampah non-organik',
            'Bisa dijadikan kompos di rumah',
            'Jangan dicampur dengan sampah plastik',
            'Buang di tempat sampah organik'
        ],
        'benefits': 'Dapat diubah menjadi kompos yang bermanfaat untuk tanaman dan mengurangi penggunaan pupuk kimia.',
        'icon': '🍃'
    },
    'Non-Organik': {
        'name': 'Sampah Non-Organik',
        'description': 'Sampah yang tidak dapat terurai secara alami dan membutuhkan waktu lama untuk terurai di alam.',
        'examples': ['Plastik', 'Kaca', 'Logam', 'Kertas', 'Styrofoam', 'Elektronik'],
        'recyclable': True,
        'recycling_info': 'Dapat didaur ulang melalui proses daur ulang di fasilitas khusus. Beberapa jenis seperti plastik, kaca, dan logam dapat diolah kembali menjadi produk baru.',
        'disposal_tips': [
            'Pisahkan berdasarkan jenisnya (plastik, kaca, logam)',
            'Cuci bersih sebelum dibuang',
            'Bawa ke bank sampah atau tempat daur ulang',
            'Kurangi penggunaan produk sekali pakai'
        ],
        'benefits': 'Daur ulang membantu mengurangi polusi, menghemat energi, dan mengurangi penggunaan sumber daya alam baru.',
        'icon': '♻️'
    }
}

def predict_image(image_path, use_yolo=False):
    """Prediksi kelas sampah dari gambar dengan atau tanpa YOLO"""
    if model is None:
        return None, "Model belum dimuat"
    
    start_time = time.time()
    
    # Baca gambar
    img = cv2.imread(image_path)
    if img is None:
        return None, "Error membaca gambar"
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = []
    
    # Jika menggunakan YOLO untuk bounding box
    if use_yolo and YOLO_AVAILABLE and yolo_model:
        try:
            results = yolo_model(img_rgb, conf=0.25, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        try:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Validasi koordinat
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(img.shape[1], x2)
                            y2 = min(img.shape[0], y2)
                            
                            if x2 > x1 and y2 > y1:
                                # Crop ROI
                                roi = img_rgb[y1:y2, x1:x2]
                                
                                if roi.size > 0:
                                    # Preprocess ROI
                                    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                                    roi_array = roi_resized.astype('float32') / 255.0
                                    roi_array = np.expand_dims(roi_array, axis=0)
                                    
                                    # Klasifikasi
                                    prediction = model.predict(roi_array, verbose=0)
                                    confidence = float(prediction[0][0])
                                    
                                    if confidence > 0.5:
                                        predicted_class = 'Non-Organik'
                                        class_confidence = confidence * 100
                                    else:
                                        predicted_class = 'Organik'
                                        class_confidence = (1 - confidence) * 100
                                    
                                    detections.append({
                                        'bbox': [x1, y1, x2, y2],
                                        'class': predicted_class,
                                        'confidence': round(class_confidence, 2)
                                    })
                        except:
                            continue
        except Exception as e:
            print(f"YOLO error: {e}")
    
    # Jika tidak ada deteksi YOLO atau tidak menggunakan YOLO, klasifikasi seluruh gambar
    if not detections:
        img_array = preprocess_image(image_path)
        if img_array is None:
            return None, "Error preprocessing gambar"
        
        prediction = model.predict(img_array, verbose=0)
        confidence = float(prediction[0][0])
        
        if confidence > 0.5:
            predicted_class = 'Non-Organik'
            class_confidence = confidence * 100
        else:
            predicted_class = 'Organik'
            class_confidence = (1 - confidence) * 100
        
        processing_time = time.time() - start_time
        
        # Tambahkan informasi detail
        result = {
            'class': predicted_class,
            'confidence': round(class_confidence, 2),
            'raw_confidence': round(confidence, 4),
            'processing_time': round(processing_time, 3),
            'detections': []
        }
        
        # Tambahkan info detail jika kelas terdeteksi
        if predicted_class in WASTE_INFO:
            result['info'] = WASTE_INFO[predicted_class]
        
        return result, None
    
    processing_time = time.time() - start_time
    
    # Hitung statistik semua deteksi
    if detections:
        total_detections = len(detections)
        organic_count = len([d for d in detections if d['class'] == 'Organik'])
        non_organic_count = len([d for d in detections if d['class'] == 'Non-Organik'])
        
        # Tentukan kelas utama (yang paling banyak)
        if organic_count >= non_organic_count:
            predicted_class = 'Organik'
            main_confidence = max([d['confidence'] for d in detections if d['class'] == 'Organik'] or [0])
        else:
            predicted_class = 'Non-Organik'
            main_confidence = max([d['confidence'] for d in detections if d['class'] == 'Non-Organik'] or [0])
    else:
        predicted_class = 'Unknown'
        main_confidence = 0
        total_detections = 0
        organic_count = 0
        non_organic_count = 0
    
    # Tambahkan informasi detail
    result = {
        'class': predicted_class,
        'confidence': main_confidence,
        'processing_time': round(processing_time, 3),
        'detections': detections,
        'has_bbox': True,
        'detection_summary': {
            'total': total_detections,
            'organic': organic_count,
            'non_organic': non_organic_count
        }
    }
    
    # Tambahkan info detail jika kelas terdeteksi
    if predicted_class in WASTE_INFO:
        result['info'] = WASTE_INFO[predicted_class]
    
    return result, None

def login_required(f):
    """Decorator untuk halaman yang memerlukan login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Silakan login terlebih dahulu', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Halaman utama - redirect ke login atau dashboard"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Halaman login"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Username dan password harus diisi', 'error')
            return render_template('login.html')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['is_admin'] = user.is_admin
            flash(f'Selamat datang, {user.username}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Username atau password salah', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Halaman register"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        full_name = request.form.get('full_name', '')
        
        # Validasi
        if not username or not email or not password:
            flash('Semua field harus diisi', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Password tidak cocok', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(username=username).first():
            flash('Username sudah digunakan', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email sudah digunakan', 'error')
            return render_template('register.html')
        
        # Buat user baru
        user = User(
            username=username,
            email=email,
            full_name=full_name
        )
        user.set_password(password)
        
        try:
            db.session.add(user)
            db.session.commit()
            flash('Registrasi berhasil! Silakan login', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error: {str(e)}', 'error')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    flash('Anda telah logout', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard utama"""
    user = User.query.get(session['user_id'])
    
    # Statistik user
    user_detections = Detection.query.filter_by(user_id=user.id).all()
    total_user_detections = len(user_detections)
    user_organic = len([d for d in user_detections if d.predicted_class == 'Organik'])
    user_non_organic = len([d for d in user_detections if d.predicted_class == 'Non-Organik'])
    
    # Statistik sistem (jika admin)
    system_stats = None
    if user.is_admin:
        stats = SystemStats.get_stats()
        stats.update_stats()
        system_stats = {
            'total_detections': stats.total_detections,
            'total_organic': stats.total_organic,
            'total_non_organic': stats.total_non_organic,
            'total_users': stats.total_users
        }
    
    # Recent detections
    recent_detections = Detection.query.filter_by(user_id=user.id)\
        .order_by(Detection.created_at.desc()).limit(5).all()
    
    return render_template('dashboard.html',
                         user=user,
                         total_detections=total_user_detections,
                         user_organic=user_organic,
                         user_non_organic=user_non_organic,
                         recent_detections=recent_detections,
                         system_stats=system_stats)

@app.route('/detect', methods=['GET', 'POST'])
@login_required
def detect():
    """Halaman deteksi sampah"""
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada file yang diupload'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Format file tidak didukung'}), 400
        
        if model is None:
            return jsonify({'error': 'Model belum dimuat'}), 500
        
        try:
            # Simpan file
            filename = secure_filename(f"{int(time.time())}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get image size
            img = cv2.imread(filepath)
            image_size = f"{img.shape[1]}x{img.shape[0]}" if img is not None else "Unknown"
            
            # Cek apakah menggunakan YOLO
            use_yolo = request.form.get('use_yolo', 'false').lower() == 'true'
            
            # Prediksi dengan YOLO jika diminta
            result, error = predict_image(filepath, use_yolo=use_yolo)
            
            if error:
                os.remove(filepath)
                return jsonify({'error': error}), 500
            
            # Jika ada bounding box, gambar di gambar
            if result.get('has_bbox') and result.get('detections'):
                img_with_bbox = img.copy()
                colors = {
                    'Organik': (0, 255, 0),      # Hijau (BGR)
                    'Non-Organik': (255, 0, 0)   # Merah (BGR)
                }
                
                for det in result['detections']:
                    x1, y1, x2, y2 = det['bbox']
                    color = colors[det['class']]
                    cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), color, 3)
                    
                    label = f"{det['class']} {det['confidence']}%"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    label_y = max(y1 - 10, label_size[1] + 10)
                    
                    cv2.rectangle(img_with_bbox, 
                                (x1, label_y - label_size[1] - 5),
                                (x1 + label_size[0] + 10, label_y + 5),
                                color, -1)
                    cv2.putText(img_with_bbox, label,
                              (x1 + 5, label_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                              (255, 255, 255), 2)
                
                # Simpan gambar dengan bounding box
                bbox_filename = f"bbox_{filename}"
                bbox_filepath = os.path.join(app.config['UPLOAD_FOLDER'], bbox_filename)
                cv2.imwrite(bbox_filepath, img_with_bbox)
                result['bbox_image'] = bbox_filename
                
                # Hitung statistik semua deteksi
                total_detections = len(result['detections'])
                organic_count = len([d for d in result['detections'] if d['class'] == 'Organik'])
                non_organic_count = len([d for d in result['detections'] if d['class'] == 'Non-Organik'])
                
                result['detection_summary'] = {
                    'total': total_detections,
                    'organic': organic_count,
                    'non_organic': non_organic_count
                }
            
            # Simpan ke database
            detection = Detection(
                user_id=session['user_id'],
                predicted_class=result['class'],
                confidence=result['confidence'],
                image_path=filename,
                image_size=image_size,
                processing_time=result['processing_time']
            )
            db.session.add(detection)
            db.session.commit()
            
            # Update system stats
            stats = SystemStats.get_stats()
            stats.update_stats()
            
            return jsonify({
                'success': True,
                'result': result,
                'detection_id': detection.id
            })
        
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Error: {str(e)}'}), 500
    
    return render_template('detect.html', yolo_available=YOLO_AVAILABLE)

@app.route('/history/detail/<int:detection_id>')
@login_required
def history_detail_page(detection_id):
    """Halaman detail history detection"""
    user = User.query.get(session['user_id'])
    detection = Detection.query.get_or_404(detection_id)
    
    # Cek apakah user adalah pemilik atau admin
    if detection.user_id != user.id and not user.is_admin:
        flash('Tidak memiliki izin untuk melihat detail ini', 'error')
        return redirect(url_for('history'))
    
    # Cek apakah video atau gambar
    is_video = False
    if detection.image_path:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        is_video = detection.image_path.startswith('output_') or \
                   any(detection.image_path.lower().endswith(ext) for ext in video_extensions)
    
    return render_template('history_detail.html', 
                         detection=detection, 
                         is_video=is_video,
                         user=user)

@app.route('/history')
@login_required
def history():
    """Halaman history deteksi"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    user = User.query.get(session['user_id'])
    detections = Detection.query.filter_by(user_id=user.id)\
        .order_by(Detection.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    return render_template('history.html', detections=detections)

@app.route('/history/delete/<int:detection_id>', methods=['POST'])
@login_required
def delete_history(detection_id):
    """Hapus history deteksi"""
    user = User.query.get(session['user_id'])
    detection = Detection.query.get_or_404(detection_id)
    
    # Cek apakah user adalah pemilik atau admin
    if detection.user_id != user.id and not user.is_admin:
        return jsonify({'error': 'Tidak memiliki izin untuk menghapus'}), 403
    
    try:
        # Hapus file gambar jika ada
        if detection.image_path:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], detection.image_path)
            if os.path.exists(image_path):
                os.remove(image_path)
        
        # Hapus bounding box image jika ada
        bbox_filename = f"bbox_{detection.image_path}"
        bbox_path = os.path.join(app.config['UPLOAD_FOLDER'], bbox_filename)
        if os.path.exists(bbox_path):
            os.remove(bbox_path)
        
        # Hapus dari database
        db.session.delete(detection)
        db.session.commit()
        
        # Update system stats
        stats = SystemStats.get_stats()
        stats.update_stats()
        
        return jsonify({'success': True, 'message': 'History berhasil dihapus'})
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/history/delete-multiple', methods=['POST'])
@login_required
def delete_multiple_history():
    """Hapus multiple history deteksi"""
    user = User.query.get(session['user_id'])
    data = request.get_json()
    detection_ids = data.get('ids', [])
    
    if not detection_ids:
        return jsonify({'error': 'Tidak ada history yang dipilih'}), 400
    
    try:
        deleted_count = 0
        for detection_id in detection_ids:
            detection = Detection.query.get(detection_id)
            if detection and (detection.user_id == user.id or user.is_admin):
                # Hapus file gambar jika ada
                if detection.image_path:
                    image_path = os.path.join(app.config['UPLOAD_FOLDER'], detection.image_path)
                    if os.path.exists(image_path):
                        os.remove(image_path)
                
                # Hapus bounding box image jika ada
                bbox_filename = f"bbox_{detection.image_path}"
                bbox_path = os.path.join(app.config['UPLOAD_FOLDER'], bbox_filename)
                if os.path.exists(bbox_path):
                    os.remove(bbox_path)
                
                db.session.delete(detection)
                deleted_count += 1
        
        db.session.commit()
        
        # Update system stats
        stats = SystemStats.get_stats()
        stats.update_stats()
        
        return jsonify({
            'success': True,
            'message': f'{deleted_count} history berhasil dihapus',
            'deleted_count': deleted_count
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """Halaman profile"""
    user = User.query.get(session['user_id'])
    
    if request.method == 'POST':
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        
        user.full_name = full_name
        user.email = email
        
        if new_password:
            if not current_password or not user.check_password(current_password):
                flash('Password lama salah', 'error')
                return render_template('profile.html', user=user)
            user.set_password(new_password)
            flash('Password berhasil diubah', 'success')
        
        try:
            db.session.commit()
            flash('Profile berhasil diupdate', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error: {str(e)}', 'error')
    
    return render_template('profile.html', user=user)

@app.route('/detect_video', methods=['POST'])
@login_required
def detect_video():
    """Endpoint untuk deteksi dari video file"""
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file video yang diupload'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
    
    # Cek format video
    video_extensions = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    
    if file_ext not in video_extensions:
        return jsonify({'error': 'Format video tidak didukung. Gunakan: MP4, AVI, MOV, MKV, WEBM'}), 400
    
    if model is None:
        return jsonify({'error': 'Model belum dimuat'}), 500
    
    try:
        start_time = time.time()
        
        # Simpan video
        filename = secure_filename(f"{int(time.time())}_{file.filename}")
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        
        # Proses video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            os.remove(video_path)
            return jsonify({'error': 'Tidak dapat membaca video'}), 500
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output video path - gunakan codec yang lebih compatible
        output_filename = f"output_{filename}"
        # Pastikan extension adalah .mp4 untuk kompatibilitas browser
        if not output_filename.lower().endswith('.mp4'):
            base_name = output_filename.rsplit('.', 1)[0] if '.' in output_filename else output_filename
            output_filename = base_name + '.mp4'
        
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        # Gunakan H.264 codec untuk kompatibilitas browser yang lebih baik
        # Coba beberapa codec sampai yang berhasil
        codecs_to_try = [
            ('H264', 'H.264'),
            ('XVID', 'XVID'),
            ('mp4v', 'MPEG-4'),
            ('avc1', 'H.264/AVC')
        ]
        
        out = None
        used_codec = None
        
        for codec_str, codec_name in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_str)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    used_codec = codec_name
                    print(f"Video writer created successfully with {codec_name} codec")
                    break
                else:
                    if out:
                        out.release()
                    out = None
            except Exception as e:
                print(f"Failed to create video writer with {codec_name}: {e}")
                if out:
                    out.release()
                    out = None
        
        if out is None or not out.isOpened():
            if os.path.exists(video_path):
                os.remove(video_path)
            return jsonify({'error': 'Tidak dapat membuat video writer. Codec tidak didukung.'}), 500
        
        detections_summary = {'Organik': 0, 'Non-Organik': 0}
        frame_count = 0
        processed_frames = 0
        colors = {
            'Organik': (0, 255, 0),
            'Non-Organik': (255, 0, 0)
        }
        
        # Proses setiap frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Deteksi dengan YOLO jika tersedia
            if YOLO_AVAILABLE and yolo_model:
                # YOLO expects RGB, but OpenCV uses BGR
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = yolo_model(frame_rgb, conf=0.25, verbose=False)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            try:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                x1 = max(0, x1)
                                y1 = max(0, y1)
                                x2 = min(frame.shape[1], x2)
                                y2 = min(frame.shape[0], y2)
                                
                                if x2 > x1 and y2 > y1:
                                    # Extract ROI from RGB frame for classification
                                    roi_rgb = frame_rgb[y1:y2, x1:x2]
                                    
                                    if roi_rgb.size > 0:
                                        roi_resized = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE))
                                        roi_array = roi_resized.astype('float32') / 255.0
                                        roi_array = np.expand_dims(roi_array, axis=0)
                                        
                                        prediction = model.predict(roi_array, verbose=0)
                                        confidence = float(prediction[0][0])
                                        
                                        if confidence > 0.5:
                                            predicted_class = 'Non-Organik'
                                            class_confidence = confidence * 100
                                        else:
                                            predicted_class = 'Organik'
                                            class_confidence = (1 - confidence) * 100
                                        
                                        detections_summary[predicted_class] += 1
                                        
                                        # Gambar bounding box pada frame BGR
                                        color = colors[predicted_class]
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                                        
                                        label = f"{predicted_class} {class_confidence:.1f}%"
                                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                        label_y = max(y1 - 10, label_size[1] + 10)
                                        
                                        cv2.rectangle(frame,
                                                    (x1, label_y - label_size[1] - 5),
                                                    (x1 + label_size[0] + 10, label_y + 5),
                                                    color, -1)
                                        cv2.putText(frame, label,
                                                  (x1 + 5, label_y),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                                  (255, 255, 255), 2)
                            except Exception as e:
                                print(f"Error processing box: {e}")
                                continue
            else:
                # Tanpa YOLO, klasifikasi full frame
                frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frame_array = frame_rgb.astype('float32') / 255.0
                frame_array = np.expand_dims(frame_array, axis=0)
                
                prediction = model.predict(frame_array, verbose=0)
                confidence = float(prediction[0][0])
                
                if confidence > 0.5:
                    predicted_class = 'Non-Organik'
                else:
                    predicted_class = 'Organik'
                
                detections_summary[predicted_class] += 1
                
                # Tampilkan label di tengah frame
                color = colors[predicted_class]
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (10, 10), (w-10, 100), color, -1)
                cv2.rectangle(frame, (10, 10), (w-10, 100), (255, 255, 255), 2)
                cv2.putText(frame, f"Kelas: {predicted_class}",
                          (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Confidence: {confidence*100:.1f}%",
                          (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            out.write(frame)
        
        cap.release()
        if out:
            out.release()
        
        # Pastikan video file sudah benar-benar tersimpan
        if not os.path.exists(output_path):
            if os.path.exists(video_path):
                os.remove(video_path)
            return jsonify({'error': 'Gagal menyimpan video hasil'}), 500
        
        # Cek ukuran file video
        video_size = os.path.getsize(output_path)
        if video_size == 0:
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(output_path):
                os.remove(output_path)
            return jsonify({'error': 'Video hasil kosong atau corrupt'}), 500
        
        print(f"Video berhasil disimpan: {output_path} ({video_size / (1024*1024):.2f} MB)")
        if 'used_codec' in locals():
            print(f"Codec yang digunakan: {used_codec}")
        
        # Hitung statistik
        total_detections = detections_summary['Organik'] + detections_summary['Non-Organik']
        processing_time = time.time() - start_time
        
        # Tentukan kelas utama berdasarkan yang paling banyak
        if detections_summary['Organik'] >= detections_summary['Non-Organik']:
            main_class = 'Organik'
            main_confidence = (detections_summary['Organik'] / total_detections * 100) if total_detections > 0 else 0
        else:
            main_class = 'Non-Organik'
            main_confidence = (detections_summary['Non-Organik'] / total_detections * 100) if total_detections > 0 else 0
        
        # Simpan ke database
        user = User.query.get(session['user_id'])
        detection = Detection(
            user_id=user.id,
            predicted_class=main_class,
            confidence=round(main_confidence, 2),
            image_path=output_filename,  # Simpan output video sebagai image_path
            image_size=f"{width}x{height}",
            processing_time=round(processing_time, 2)
        )
        db.session.add(detection)
        db.session.commit()
        
        # Update system stats
        stats = SystemStats.get_stats()
        stats.update_stats()
        
        # Hapus video input
        os.remove(video_path)
        
        return jsonify({
            'success': True,
            'output_video': output_filename,
            'summary': detections_summary,
            'total_frames': frame_count,
            'total_detections': total_detections,
            'main_class': main_class,
            'main_confidence': round(main_confidence, 2),
            'processing_time': round(processing_time, 2),
            'video_info': {
                'fps': fps,
                'width': width,
                'height': height,
                'duration': round(frame_count / fps, 2) if fps > 0 else 0
            },
            'detection_id': detection.id
        })
    
    except Exception as e:
        if os.path.exists(video_path):
            os.remove(video_path)
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/video/<filename>')
@login_required
def serve_video(filename):
    """Serve video file with proper headers for streaming"""
    # Decode filename jika ada encoding issues
    try:
        from urllib.parse import unquote
        filename = unquote(filename)
    except:
        pass
    
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Debug logging
    print(f"Looking for video at: {video_path}")
    print(f"File exists: {os.path.exists(video_path)}")
    
    if not os.path.exists(video_path):
        print(f"Video not found at: {video_path}")
        # Try alternative path
        alt_path = os.path.join('static', 'uploads', filename)
        if os.path.exists(alt_path):
            video_path = alt_path
            print(f"Found video at alternative path: {alt_path}")
        else:
            return jsonify({'error': f'Video tidak ditemukan: {filename}'}), 404
    
    # Determine MIME type based on extension
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'mp4'
    mime_types = {
        'mp4': 'video/mp4',
        'avi': 'video/x-msvideo',
        'mov': 'video/quicktime',
        'mkv': 'video/x-matroska',
        'webm': 'video/webm'
    }
    mimetype = mime_types.get(ext, 'video/mp4')
    
    # Get file size for range requests
    file_size = os.path.getsize(video_path)
    
    # Handle range requests for video streaming
    range_header = request.headers.get('Range', None)
    
    if range_header:
        # Parse range header
        byte_start = 0
        byte_end = file_size - 1
        
        range_match = range_header.replace('bytes=', '').split('-')
        if range_match[0]:
            byte_start = int(range_match[0])
        if len(range_match) > 1 and range_match[1]:
            byte_end = int(range_match[1])
        
        content_length = byte_end - byte_start + 1
        
        def generate_range():
            with open(video_path, 'rb') as f:
                f.seek(byte_start)
                remaining = content_length
                while remaining:
                    chunk_size = min(8192, remaining)
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk
        
        return Response(
            generate_range(),
            206,  # Partial Content
            mimetype=mimetype,
            headers={
                'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': str(content_length),
                'Content-Type': mimetype
            }
        )
    else:
        # Full file response - use send_file for better reliability
        try:
            return send_file(
                video_path,
                mimetype=mimetype,
                as_attachment=False,
                download_name=filename,
                conditional=True  # Support range requests
            )
        except Exception as e:
            print(f"Error using send_file, falling back to Response: {e}")
            # Fallback to Response generator
            def generate():
                with open(video_path, 'rb') as f:
                    while True:
                        chunk = f.read(8192)
                        if not chunk:
                            break
                        yield chunk
            
            return Response(
                generate(),
                mimetype=mimetype,
                headers={
                    'Content-Disposition': f'inline; filename="{filename}"',
                    'Accept-Ranges': 'bytes',
                    'Content-Length': str(file_size),
                    'Content-Type': mimetype,
                    'Cache-Control': 'public, max-age=3600'
                }
            )

@app.route('/admin')
@login_required
def admin_dashboard():
    """Admin dashboard"""
    user = User.query.get(session['user_id'])
    
    if not user.is_admin:
        flash('Akses ditolak. Hanya admin yang dapat mengakses halaman ini.', 'error')
        return redirect(url_for('dashboard'))
    
    # Statistik sistem
    stats = SystemStats.get_stats()
    stats.update_stats()
    
    # Total users
    total_users = User.query.count()
    
    # Recent detections (semua user)
    recent_detections = Detection.query\
        .order_by(Detection.created_at.desc()).limit(10).all()
    
    # Top users by detections
    from sqlalchemy import func
    top_users = db.session.query(
        User.username,
        User.full_name,
        func.count(Detection.id).label('detection_count')
    ).join(Detection).group_by(User.id).order_by(func.count(Detection.id).desc()).limit(5).all()
    
    return render_template('admin/dashboard.html',
                         stats=stats,
                         total_users=total_users,
                         recent_detections=recent_detections,
                         top_users=top_users)

@app.route('/admin/users')
@login_required
def admin_users():
    """Admin - Daftar semua user"""
    user = User.query.get(session['user_id'])
    
    if not user.is_admin:
        flash('Akses ditolak. Hanya admin yang dapat mengakses halaman ini.', 'error')
        return redirect(url_for('dashboard'))
    
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    users = User.query.order_by(User.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    return render_template('admin/users.html', users=users)

@app.route('/admin/history')
@login_required
def admin_history():
    """Admin - History semua deteksi"""
    user = User.query.get(session['user_id'])
    
    if not user.is_admin:
        flash('Akses ditolak. Hanya admin yang dapat mengakses halaman ini.', 'error')
        return redirect(url_for('dashboard'))
    
    page = request.args.get('page', 1, type=int)
    per_page = 30
    
    # Filter options
    filter_user = request.args.get('user_id', type=int)
    filter_class = request.args.get('class', '')
    
    query = Detection.query
    
    if filter_user:
        query = query.filter_by(user_id=filter_user)
    
    if filter_class:
        query = query.filter_by(predicted_class=filter_class)
    
    detections = query.order_by(Detection.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    # Get all users for filter dropdown
    all_users = User.query.order_by(User.username).all()
    
    return render_template('admin/history.html',
                         detections=detections,
                         all_users=all_users,
                         filter_user=filter_user,
                         filter_class=filter_class)

@app.route('/api/stats')
@login_required
def api_stats():
    """API untuk statistik"""
    user = User.query.get(session['user_id'])
    
    # Statistik user
    detections = Detection.query.filter_by(user_id=user.id).all()
    
    # Statistik per hari (7 hari terakhir)
    daily_stats = {}
    for i in range(7):
        date = (datetime.now() - timedelta(days=i)).date()
        day_detections = [d for d in detections if d.created_at.date() == date]
        daily_stats[date.isoformat()] = {
            'total': len(day_detections),
            'organic': len([d for d in day_detections if d.predicted_class == 'Organik']),
            'non_organic': len([d for d in day_detections if d.predicted_class == 'Non-Organik'])
        }
    
    return jsonify({
        'total': len(detections),
        'organic': len([d for d in detections if d.predicted_class == 'Organik']),
        'non_organic': len([d for d in detections if d.predicted_class == 'Non-Organik']),
        'daily_stats': daily_stats
    })

@app.route('/health')
def health():
    """Health check"""
    model_file = MODEL_PATH if os.path.exists(MODEL_PATH) else (FALLBACK_MODEL if os.path.exists(FALLBACK_MODEL) else 'Not found')
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'model_file': model_file,
        'model_exists': os.path.exists(MODEL_PATH) or os.path.exists(FALLBACK_MODEL)
    })

# ==================== INITIALIZE ====================

def init_db():
    """Initialize database"""
    with app.app_context():
        db.create_all()
        
        # Buat admin user jika belum ada
        if not User.query.filter_by(username='admin').first():
            admin = User(
                username='admin',
                email='admin@waste-detection.com',
                full_name='Administrator',
                is_admin=True
            )
            admin.set_password('admin123')  # Ganti password di production!
            db.session.add(admin)
            db.session.commit()
            print("✅ Admin user created: username='admin', password='admin123'")

if __name__ == '__main__':
    print("="*60)
    print("INISIALISASI SISTEM...")
    print("="*60)
    
    # Initialize database
    init_db()
    
    # Load model
    print("\nMEMUAT MODEL...")
    if load_model():
        print("\n" + "="*60)
        print("SERVER SIAP!")
        print("="*60)
        print("Buka browser dan akses: http://localhost:5000")
        print("Admin login: username='admin', password='admin123'")
        print("Tekan Ctrl+C untuk menghentikan server")
        print("="*60 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Gagal memuat model. Server tidak dapat dijalankan.")

