from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    results = db.relationship('ScreeningResult', backref='user', lazy=True)

class ScreeningResult(db.Model):
    __tablename__ = 'screening_results'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    name = db.Column(db.String(100))
    age = db.Column(db.Float)
    gender = db.Column(db.String(20))
    occupation = db.Column(db.String(100))
    sleep_hours = db.Column(db.Float)
    physical_activity = db.Column(db.Float)
    social_support = db.Column(db.Float)
    work_life_balance = db.Column(db.Float)
    screen_time = db.Column(db.Float)
    stress_level = db.Column(db.Float)
    self_esteem = db.Column(db.Float)
    coping_skills = db.Column(db.Float)
    mindfulness_practice = db.Column(db.Float)
    family_history = db.Column(db.Integer)
    chronic_illness = db.Column(db.Integer)
    substance_use = db.Column(db.Integer)
    previous_treatment = db.Column(db.Integer)
    phq9_score = db.Column(db.Float)
    gad7_score = db.Column(db.Float)
    pss_score = db.Column(db.Float)
    risk_level = db.Column(db.String(20))
    risk_score = db.Column(db.Float)
    confidence_score = db.Column(db.Float)
    model_used = db.Column(db.String(100))
    
    # Probabilities
    prob_low = db.Column(db.Float)
    prob_moderate = db.Column(db.Float)
    prob_high = db.Column(db.Float)

class Resource(db.Model):
    __tablename__ = 'resources'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(50))
    content = db.Column(db.Text)
    icon = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
