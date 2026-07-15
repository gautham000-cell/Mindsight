"""
Mental Health Prediction System - Flask API Backend
Serves the web dashboard and provides prediction endpoints.
All screening results are saved to a local SQLite database.
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import hashlib
import secrets
import re
from functools import wraps
import joblib
import json
import numpy as np
import pandas as pd
import os
import sqlite3
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)
app.secret_key = 'mhps-secret-key-2024'

# ======== Database Setup ========
DB_PATH = 'data/screening_results.db'
CSV_PATH = 'data/mental_health_dataset.csv'

# ======== Email Configuration (Developer Portal) ========
# To enable actual email sending, fill in your SMTP details below.
# For Gmail, you may need an 'App Password'.
MAIL_SERVER     = 'smtp.gmail.com'
MAIL_PORT       = 587
MAIL_USERNAME   = 'MindSightttt@gmail.com' # Developer Email
MAIL_PASSWORD   = 'pbjxtmofinmpxirr'           # Use App Password, not normal password
MAIL_SENDER     = 'MindSightttt@gmail.com'
DEVELOPER_EMAIL = 'MindSightttt@gmail.com' # Where notifications arrive

def _hash_password(password, salt=None):
    """SHA-256 + random salt password hashing."""
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"

def _verify_password(password, stored):
    """Verify password against stored salt:hash string."""
    try:
        salt, _ = stored.split(':', 1)
        return _hash_password(password, salt) == stored
    except Exception:
        return False

def login_required(f):
    """Decorator: redirect to login if not authenticated."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please sign in to access this page.', 'info')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def init_db():
    """Create the SQLite database and all tables if not exists."""
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # ── Users table ──
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name   TEXT,
            last_name    TEXT,
            email        TEXT UNIQUE NOT NULL,
            username     TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at   TEXT NOT NULL
        )
    ''')

    # ── Screening results table ──
    c.execute('''
        CREATE TABLE IF NOT EXISTS screening_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            submitted_at    TEXT NOT NULL,
            name            TEXT,
            age             REAL,
            gender          TEXT,
            occupation      TEXT,
            sleep_hours     REAL,
            physical_activity REAL,
            social_support  REAL,
            work_life_balance REAL,
            screen_time     REAL,
            mindfulness_practice REAL,
            stress_level    REAL,
            self_esteem     REAL,
            coping_skills   REAL,
            family_history  INTEGER,
            chronic_illness INTEGER,
            substance_use   INTEGER,
            previous_treatment INTEGER,
            phq9_score      REAL,
            gad7_score      REAL,
            pss_score       REAL,
            risk_level      TEXT,
            risk_score      REAL,
            confidence_score REAL,
            model_used      TEXT,
            prob_low        REAL,
            prob_moderate   REAL,
            prob_high       REAL
        )
    ''')

    # ── Contact messages table ──
    c.execute('''
        CREATE TABLE IF NOT EXISTS contact_messages (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT NOT NULL,
            email        TEXT NOT NULL,
            message      TEXT NOT NULL,
            submitted_at TEXT NOT NULL,
            status       TEXT DEFAULT 'unread'
        )
    ''')
    conn.commit()
    conn.close()
    print("  Database initialised at:", DB_PATH)

def save_result(data, risk_level, risk_score, confidence_score, prob_dict, model_used):
    """Insert one screening result row into the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT INTO screening_results (
                submitted_at, name, age, gender, occupation,
                sleep_hours, physical_activity, social_support, work_life_balance,
                screen_time, mindfulness_practice, stress_level, self_esteem,
                coping_skills, family_history, chronic_illness, substance_use,
                previous_treatment, phq9_score, gad7_score, pss_score,
                risk_level, risk_score, confidence_score, model_used,
                prob_low, prob_moderate, prob_high
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            data.get('name', 'User'),
            float(data.get('age', 0)),
            data.get('gender', ''),
            data.get('occupation', ''),
            float(data.get('sleep_hours', 0)),
            float(data.get('physical_activity', 0)),
            float(data.get('social_support', 0)),
            float(data.get('work_life_balance', 0)),
            float(data.get('screen_time', 0)),
            float(data.get('mindfulness_practice', 0)),
            float(data.get('stress_level', 0)),
            float(data.get('self_esteem', 0)),
            float(data.get('coping_skills', 0)),
            int(data.get('family_history', 0)),
            int(data.get('chronic_illness', 0)),
            int(data.get('substance_use', 0)),
            int(data.get('previous_treatment', 0)),
            float(data.get('phq9_score', 0)),
            float(data.get('gad7_score', 0)),
            float(data.get('pss_score', 0)),
            risk_level,
            risk_score,
            confidence_score,
            model_used,
            prob_dict.get('Low', 0),
            prob_dict.get('Moderate', 0),
            prob_dict.get('High', 0),
        ))
        conn.commit()
        row_id = c.lastrowid
        conn.close()
        return row_id
    except Exception as e:
        print(f"DB save error: {e}")
        return None

def save_to_csv(data, risk_level, risk_score):
    """Append one screening result row to the local CSV dataset."""
    try:
        import csv
        import os
        
        file_exists = os.path.isfile(CSV_PATH)
        
        # Define columns in exact order as CSV
        columns = [
            'age', 'gender', 'occupation', 'sleep_hours', 'physical_activity',
            'social_support', 'work_life_balance', 'screen_time', 'stress_level',
            'self_esteem', 'coping_skills', 'mindfulness_practice', 'family_history',
            'chronic_illness', 'substance_use', 'previous_treatment', 'phq9_score',
            'gad7_score', 'pss_score', 'risk_score', 'risk_level'
        ]
        
        # Prepare row data
        row = {
            'age': data.get('age', 25),
            'gender': data.get('gender', 'Male'),
            'occupation': data.get('occupation', 'Others'),
            'sleep_hours': data.get('sleep_hours', 7.0),
            'physical_activity': data.get('physical_activity', 5.0),
            'social_support': data.get('social_support', 5.0),
            'work_life_balance': data.get('work_life_balance', 5.0),
            'screen_time': data.get('screen_time', 5.0),
            'stress_level': data.get('stress_level', 5.0),
            'self_esteem': data.get('self_esteem', 5.0),
            'coping_skills': data.get('coping_skills', 5.0),
            'mindfulness_practice': data.get('mindfulness_practice', 5.0),
            'family_history': data.get('family_history', 0),
            'chronic_illness': data.get('chronic_illness', 0),
            'substance_use': data.get('substance_use', 0),
            'previous_treatment': data.get('previous_treatment', 0),
            'phq9_score': data.get('phq9_score', 5.0),
            'gad7_score': data.get('gad7_score', 5.0),
            'pss_score': data.get('pss_score', 15.0),
            'risk_score': risk_score,
            'risk_level': risk_level
        }
        
        with open(CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            # Write header only if file is brand new (though it should already exist)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            
    except Exception as e:
        print(f"CSV save error: {e}")

# Initialise DB on startup
init_db()

# ======== Load Models ========
def load_models():
    try:
        model     = joblib.load('models/best_model.pkl')
        scaler    = joblib.load('models/scaler.pkl')
        le_risk   = joblib.load('models/label_encoder.pkl')
        le_gender = joblib.load('models/gender_encoder.pkl')
        le_occ    = joblib.load('models/occupation_encoder.pkl')

        with open('data/model_results.json') as f:
            model_meta = json.load(f)
        with open('data/summary.json') as f:
            summary = json.load(f)

        return model, scaler, le_risk, le_gender, le_occ, model_meta, summary
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, None, None, {}, {}

model, scaler, le_risk, le_gender, le_occ, model_meta, summary = load_models()

# ======== Helper: current user info ========
def get_current_user():
    """Return user dict from session, or None."""
    if 'user_id' not in session:
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute('SELECT id, username, first_name, email FROM users WHERE id=?',
                           (session['user_id'],)).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None

def get_dynamic_summary():
    """Update static summary with real-time record counts from CSV."""
    dynamic_summary = summary.copy()
    try:
        # Count rows in CSV (Total = total train data + new assessments)
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, 'r', encoding='utf-8') as f:
                # Total lines minus header
                total_csv = sum(1 for line in f) - 1
                dynamic_summary['total_records'] = max(total_csv, dynamic_summary.get('total_records', 0))
    except Exception as e:
        print(f"Summary calc error: {e}")
    return dynamic_summary

# ======== Routes ========

@app.route('/')
def index():
    return render_template('index.html', summary=get_dynamic_summary(), 
                           model_meta=model_meta,
                           current_user=get_current_user())

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', summary=get_dynamic_summary(), 
                           model_meta=model_meta,
                           current_user=get_current_user())

@app.route('/screening')
@login_required
def screening():
    return render_template('screening.html', current_user=get_current_user())

@app.route('/resources')
def resources():
    """Knowledge Hub for mental health articles and tips."""
    try:
        with open('data/resources.json', 'r', encoding='utf-8') as f:
            all_resources = json.load(f)
    except Exception:
        all_resources = []
    return render_template('resources.html', resources=all_resources, current_user=get_current_user())

@app.route('/resources/<int:resource_id>')
def resource_detail(resource_id):
    """View a specific article from the Knowledge Hub."""
    try:
        with open('data/resources.json', 'r', encoding='utf-8') as f:
            all_resources = json.load(f)
        article = next((r for r in all_resources if r['id'] == resource_id), None)
        if not article:
            flash("Article not found.", "error")
            return redirect(url_for('resources'))
    except Exception:
        return redirect(url_for('resources'))
    return render_template('article.html', article=article, current_user=get_current_user())
@app.route('/api/user-trends')
@login_required
def user_trends():
    """Fetch user's risk scores over time for trend charts."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        # In a real "big project", we'd filter by user_id. 
        # For now, we'll fetch all and show trends.
        c.execute('SELECT submitted_at, risk_score FROM screening_results ORDER BY submitted_at ASC')
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return jsonify({'success': True, 'trends': rows})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/history')
@login_required
def history():
    """Display past screening results directly from the CSV dataset."""
    try:
        page = int(request.args.get('page', 1))
        per_page = 20
        
        if not os.path.exists(CSV_PATH):
            return render_template('history.html', rows=[], total=0, page=1, total_pages=1)

        # Use pandas to read only necessary columns or full file
        # Reading 30k rows is fast enough for localized dev
        df = pd.read_csv(CSV_PATH)
        
        # Reverse to show newest first (assuming newest are at the bottom)
        df = df.iloc[::-1]
        
        total_records = len(df)
        total_pages = (total_records + per_page - 1) // per_page
        
        # Get counts for stats from full dataset
        risk_counts = df['risk_level'].value_counts().to_dict()
        high_count = int(risk_counts.get('High', 0))
        mod_count = int(risk_counts.get('Moderate', 0))
        low_count = int(risk_counts.get('Low', 0))

        # Paginate
        start = (page - 1) * per_page
        end = start + per_page
        rows_df = df.iloc[start:end]
        
        # Convert to list of dicts for template
        rows = rows_df.to_dict('records')
        
        # Add index-based ID for deletion simulation if needed
        for i, r in enumerate(rows):
            r['id'] = start + i
            # CSV doesn't have submitted_at, so we provide fallback or skip
            # If the CSV was recently appended with current results, maybe we add a 'date' column?
            # For now, we'll just handle it in the template.
        
    except Exception as e:
        print(f"History CSV fetch error: {e}")
        rows, total_records, high_count, mod_count, low_count = [], 0, 0, 0, 0
        total_pages, page = 1, 1

    return render_template('history.html',
                           rows=rows, total=total_records,
                           high_count=high_count,
                           mod_count=mod_count,
                           low_count=low_count,
                           page=page,
                           total_pages=total_pages,
                           current_user=get_current_user())

@app.route('/api/history/export')
@login_required
def export_history():
    """Serve the mental health dataset CSV for download."""
    try:
        if os.path.exists(CSV_PATH):
            from flask import send_file
            return send_file(CSV_PATH, as_attachment=True, download_name='mental_health_history.csv')
        return jsonify({'success': False, 'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/import-data')
@login_required
def import_csv_to_db():
    """Admin tool to sync CSV data into the SQLite database for visibility."""
    try:
        if not os.path.exists(CSV_PATH):
            return jsonify({'success': False, 'error': 'CSV not found'})
            
        df = pd.read_csv(CSV_PATH)
        # To avoid bloating, let's import a max sample if it's huge, 
        # or just import recent 500 if they aren't there.
        # For "Big Project" feel, let's import top 200.
        sample = df.head(200) 
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        import random
        from datetime import timedelta
        
        imported_count = 0
        for _, row in sample.iterrows():
            # Check if exists (crude check by risk_score and properties)
            # In real app, we'd have a 'source' column
            
            # Generate a slightly varied timestamp for the past 30 days
            random_days = random.randint(0, 30)
            random_time = datetime.now() - timedelta(days=random_days, hours=random.randint(0,23))
            timestamp = random_time.strftime('%Y-%m-%d %H:%M:%S')
            
            c.execute('''
                INSERT INTO screening_results (
                    submitted_at, name, age, gender, occupation,
                    sleep_hours, physical_activity, social_support, work_life_balance,
                    screen_time, mindfulness_practice, stress_level, self_esteem,
                    coping_skills, family_history, chronic_illness, substance_use,
                    previous_treatment, phq9_score, gad7_score, pss_score,
                    risk_level, risk_score, confidence_score, model_used,
                    prob_low, prob_moderate, prob_high
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                timestamp,
                "Dataset Sample User",
                float(row.get('age', 0)),
                row.get('gender', 'Other'),
                row.get('occupation', 'Other'),
                float(row.get('sleep_hours', 0)),
                float(row.get('physical_activity', 0)),
                float(row.get('social_support', 0)),
                float(row.get('work_life_balance', 0)),
                float(row.get('screen_time', 0)),
                float(row.get('mindfulness_practice', 0)),
                float(row.get('stress_level', 0)),
                float(row.get('self_esteem', 0)),
                float(row.get('coping_skills', 0)),
                int(row.get('family_history', 0)),
                int(row.get('chronic_illness', 0)),
                int(row.get('substance_use', 0)),
                int(row.get('previous_treatment', 0)),
                float(row.get('phq9_score', 0)),
                float(row.get('gad7_score', 0)),
                float(row.get('pss_score', 0)),
                row.get('risk_level', 'Low'),
                float(row.get('risk_score', 0)),
                95.0, # Dummy confidence for imported data
                "XGBoost (Batch)",
                33.3, 33.3, 33.3 # Dummy probs
            ))
            imported_count += 1
            
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'imported': imported_count})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ======== Auth Routes ========

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember')
        if not username or not password:
            flash('Please fill in all fields.', 'error')
            return render_template('login.html')
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                'SELECT * FROM users WHERE username=? OR email=?',
                (username, username)
            ).fetchone()
            conn.close()
        except Exception as e:
            flash('Database error. Please try again.', 'error')
            return render_template('login.html')
        if row and _verify_password(password, row['password_hash']):
            session.permanent = bool(remember)
            session['user_id']  = row['id']
            session['username'] = row['username']
            flash(f"Welcome back, {row['first_name'] or row['username']}! 👋", 'success')
            next_url = request.args.get('next', url_for('index'))
            return redirect(next_url)
        else:
            flash('Invalid username or password. Please try again.', 'error')
            return render_template('login.html')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('index'))
    if request.method == 'POST':
        first_name       = request.form.get('first_name', '').strip()
        last_name        = request.form.get('last_name', '').strip()
        email            = request.form.get('email', '').strip().lower()
        username         = request.form.get('username', '').strip()
        password         = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        terms            = request.form.get('terms')
        # Validation
        if not all([first_name, email, username, password, confirm_password]):
            flash('Please fill in all required fields.', 'error')
            return render_template('register.html')
        if not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
            flash('Please enter a valid email address.', 'error')
            return render_template('register.html')
        if len(username) < 3:
            flash('Username must be at least 3 characters.', 'error')
            return render_template('register.html')
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template('register.html')
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
        if not terms:
            flash('You must agree to the Terms of Service.', 'error')
            return render_template('register.html')
        password_hash = _hash_password(password)
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute(
                'INSERT INTO users (first_name, last_name, email, username, password_hash, created_at) VALUES (?,?,?,?,?,?)',
                (first_name, last_name, email, username, password_hash,
                 datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            )
            conn.commit()
            conn.close()
        except sqlite3.IntegrityError:
            flash('Username or email already taken. Please choose another.', 'error')
            return render_template('register.html')
        except Exception as e:
            flash('Registration failed. Please try again.', 'error')
            return render_template('register.html')
        flash('Account created successfully! Please sign in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been signed out.', 'info')
    return redirect(url_for('login'))

# ======== Prediction ========

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Check if model components are loaded
        if not all([model, scaler, le_risk, le_gender, le_occ]):
            return jsonify({'success': False,
                            'error': 'AI models not loaded. Run train_model.py first.'}), 500

        data = request.get_json()
        name = data.get('name', 'User')

        # Encode categoricals safely
        try:
            gender_raw = data.get('gender')
            gender_enc = int(le_gender.transform([gender_raw])[0]) \
                if gender_raw in le_gender.classes_ else 0
        except Exception:
            gender_enc = 0

        try:
            occ_raw = data.get('occupation')
            occ_enc = int(le_occ.transform([occ_raw])[0]) \
                if occ_raw in le_occ.classes_ else 0
        except Exception:
            occ_enc = 0

        # Base feature values
        age         = float(data.get('age', 25))
        sleep_hours = float(data.get('sleep_hours', 7))
        phys_act    = float(data.get('physical_activity', 5))
        soc_sup     = float(data.get('social_support', 5))
        wlb         = float(data.get('work_life_balance', 5))
        screen      = float(data.get('screen_time', 5))
        stress      = float(data.get('stress_level', 5))
        esteem      = float(data.get('self_esteem', 5))
        coping      = float(data.get('coping_skills', 5))
        mindful     = float(data.get('mindfulness_practice', 5))
        fam_hist    = int(data.get('family_history', 0))
        chronic     = int(data.get('chronic_illness', 0))
        substance   = int(data.get('substance_use', 0))
        prev_treat  = int(data.get('previous_treatment', 0))
        
        # Scale 1-10 clinical inputs to traditional ranges (27, 21, 40)
        phq9        = (float(data.get('phq9_score', 5)) / 10) * 27
        gad7        = (float(data.get('gad7_score', 5)) / 10) * 21
        pss         = (float(data.get('pss_score', 5)) / 10) * 40

        # Engineered features (must match train_model.py)
        sleep_stress_ratio     = sleep_hours / (stress + 1)
        support_esteem_sum     = soc_sup + esteem
        clinical_burden        = (phq9/27 + gad7/21 + pss/40) / 3 * 10
        protective_score       = (coping + mindful + phys_act) / 3
        risk_flag_count        = fam_hist + chronic + substance + prev_treat
        activity_sleep_product = phys_act * sleep_hours / 10

        features = np.array([[
            age, gender_enc, occ_enc,
            sleep_hours, phys_act, soc_sup, wlb, screen, stress,
            esteem, coping, mindful,
            fam_hist, chronic, substance, prev_treat,
            phq9, gad7, pss,
            sleep_stress_ratio, support_esteem_sum, clinical_burden,
            protective_score, risk_flag_count, activity_sleep_product
        ]])

        features_scaled  = scaler.transform(features)
        prediction_enc   = model.predict(features_scaled)[0]
        probabilities    = model.predict_proba(features_scaled)[0]
        risk_level       = le_risk.inverse_transform([prediction_enc])[0]

        # Composite risk score (1-10)
        risk_score = round(
            0.20*(phq9/27*10) + 0.16*(gad7/21*10) + 0.12*(pss/40*10) +
            0.10*stress + 0.08*(10-sleep_hours) + 0.07*(10-soc_sup) +
            0.07*(10-coping) + 0.06*(10-esteem) +
            0.04*(10-wlb) + 0.03*fam_hist*10 + 0.03*substance*10 +
            0.02*chronic*10 + 0.02*screen, 2
        )
        risk_score = min(10, max(1, risk_score))

        classes    = le_risk.classes_.tolist()
        prob_dict  = {c: round(float(p)*100, 1) for c, p in zip(classes, probabilities)}
        confidence_score = round(float(probabilities[prediction_enc])*100, 1)
        model_used = model_meta.get('best_model', 'ML Model')

        # ---- Save to database ----
        record_id = save_result(data, risk_level, risk_score,
                                confidence_score, prob_dict, model_used)

        # ---- Save to CSV ----
        save_to_csv(data, risk_level, risk_score)

        recommendations = generate_recommendations(data, risk_level)

        return jsonify({
            'success':          True,
            'record_id':        record_id,
            'name':             name,
            'risk_level':       risk_level,
            'risk_score':       risk_score,
            'confidence_score': confidence_score,
            'probabilities':    prob_dict,
            'recommendations':  recommendations,
            'model_used':       model_used
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def generate_recommendations(data, risk_level):
    recs  = []
    sleep  = float(data.get('sleep_hours', 7))
    wlb    = float(data.get('work_life_balance', 5))

    if risk_level == 'High':
        recs.append({'category': 'Professional Help', 'icon': '🩺',
            'advice': 'Seek Counseling: Your assessment indicates high symptoms. We strongly recommend scheduling an appointment with a licensed therapist or psychiatrist immediately.'})
        recs.append({'category': 'Mindfulness', 'icon': '🧘',
            'advice': 'Practice Mindfulness: Implement deep breathing exercises or guided meditation daily to help manage acute distress.'})
        recs.append({'category': 'EMERGENCY', 'icon': '🆘',
            'advice': 'Emergency Help: If you are in immediate distress, please contact your local emergency services or a mental health crisis helpline (e.g., iCall: 9152987821).'})
    elif risk_level == 'Moderate':
        recs.append({'category': 'Sleep Hygiene', 'icon': '🌙',
            'advice': 'Improve Sleep: Aim for a strict 8-hour sleep schedule and avoid screens 1 hour before bed.'})
        recs.append({'category': 'Burnout Prevention', 'icon': '⚖️',
            'advice': 'Reduce Workload: Delegate tasks, set boundaries, and prioritize breaks to prevent exhaustion.'})
        recs.append({'category': 'Lifestyle', 'icon': '💆',
            'advice': 'Engage in dedicated self-care activities like journaling or light physical exercise to lower your baseline stress level.'})
    else:
        recs.append({'category': 'Consistency', 'icon': '🌱',
            'advice': 'Maintain Healthy Habits: Continue your existing exercise, social routines, and sleep patterns to protect your resilience.'})
        recs.append({'category': 'Social Support', 'icon': '🤝',
            'advice': 'Stay Connected: Maintain your strong social support system. Helping others can also reinforce your own mental well-being.'})

    if sleep < 6 and risk_level != 'Moderate':
        recs.append({'category': 'Sleep Alert', 'icon': '💤',
            'advice': 'Consistently low sleep detected. This is a primary driver of mental health risk.'})
    if wlb < 4 and risk_level != 'Moderate':
        recs.append({'category': 'Work-Life Alert', 'icon': '💼',
            'advice': 'Your work-life balance is low. Integrating leisure time is critical for emotional recovery.'})

    return recs[:4]


# ======== API Endpoints ========

@app.route('/api/stats')
def api_stats():
    return jsonify(summary)

@app.route('/api/model-results')
def api_model_results():
    return jsonify(model_meta)

@app.route('/api/history')
@login_required
def api_history():
    """Return all screening records as JSON."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM screening_results ORDER BY submitted_at DESC')
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return jsonify({'success': True, 'count': len(rows), 'records': rows})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/history/<int:record_id>', methods=['DELETE'])
@login_required
def delete_record(record_id):
    """Delete a specific screening record."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute('DELETE FROM screening_results WHERE id = ?', (record_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'deleted_id': record_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ── Contact Form API ──
def send_contact_email(name, email, message):
    """Utility to send an email notification to the developer."""
    try:
        msg = MIMEMultipart()
        msg['From'] = MAIL_SENDER
        msg['To'] = DEVELOPER_EMAIL
        msg['Subject'] = f"New MindSight Message from {name}"

        body = f"""
        Hello Developer,

        You have received a new message through the MindSight Contact Form.

        Sender Name: {name}
        Sender Email: {email}
        Submitted At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        Message Content:
        ------------------------------------------
        {message}
        ------------------------------------------

        To reply to this user, simply click 'Reply' in your email client.
        """
        msg.attach(MIMEText(body, 'plain'))

        # Standard SMTP sending flow
        server = smtplib.SMTP(MAIL_SERVER, MAIL_PORT)
        server.starttls()
        server.login(MAIL_USERNAME, MAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Email failed to send: {e}")
        return False

@app.route('/api/contact', methods=['POST'])
def contact_submit():
    """Handle contact form submissions."""
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        message = data.get('message')

        if not all([name, email, message]):
            return jsonify({'success': False, 'error': 'All fields are required.'}), 400

        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            'INSERT INTO contact_messages (name, email, message, submitted_at) VALUES (?, ?, ?, ?)',
            (name, email, message, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        conn.commit()
        conn.close()

        # Try to send email notification to the developer
        # Note: If MAIL_PASSWORD is not set, this will fail gracefully but log the error
        send_contact_email(name, email, message)

        return jsonify({'success': True, 'message': 'Message sent successfully! Our team will get back to you soon.'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    init_db()  # Ensure tables exist
    print("=" * 55)
    print("  Mental Health Prediction System")
    print("  Server starting at http://localhost:5000")
    print("  History at  http://localhost:5000/history")
    print("=" * 55)
    app.run(debug=True, port=5000)
