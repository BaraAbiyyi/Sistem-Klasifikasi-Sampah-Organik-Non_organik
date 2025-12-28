"""
Database Models untuk Sistem Deteksi Sampah
"""

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    """Model untuk user"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)
    
    # Relationship
    detections = db.relationship('Detection', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password"""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_admin': self.is_admin,
            'total_detections': len(self.detections)
        }

class Detection(db.Model):
    """Model untuk hasil deteksi"""
    __tablename__ = 'detections'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Hasil deteksi
    predicted_class = db.Column(db.String(50), nullable=False)  # Organik atau Non-Organik
    confidence = db.Column(db.Float, nullable=False)
    image_path = db.Column(db.String(255))  # Path ke gambar yang diupload
    
    # Metadata
    image_size = db.Column(db.String(50))  # e.g., "1920x1080"
    processing_time = db.Column(db.Float)  # Waktu processing dalam detik
    
    # Timestamp
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'predicted_class': self.predicted_class,
            'confidence': self.confidence,
            'image_path': self.image_path,
            'image_size': self.image_size,
            'processing_time': self.processing_time,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'user': self.user.username if self.user else None
        }

class SystemStats(db.Model):
    """Model untuk statistik sistem"""
    __tablename__ = 'system_stats'
    
    id = db.Column(db.Integer, primary_key=True)
    total_detections = db.Column(db.Integer, default=0)
    total_organic = db.Column(db.Integer, default=0)
    total_non_organic = db.Column(db.Integer, default=0)
    total_users = db.Column(db.Integer, default=0)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    @classmethod
    def get_stats(cls):
        """Get atau create stats"""
        stats = cls.query.first()
        if not stats:
            stats = cls()
            db.session.add(stats)
            db.session.commit()
        return stats
    
    def update_stats(self):
        """Update statistik dari database"""
        from models import Detection, User
        self.total_detections = Detection.query.count()
        self.total_organic = Detection.query.filter_by(predicted_class='Organik').count()
        self.total_non_organic = Detection.query.filter_by(predicted_class='Non-Organik').count()
        self.total_users = User.query.count()
        self.last_updated = datetime.utcnow()
        db.session.commit()

