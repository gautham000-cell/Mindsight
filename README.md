<div align="center">

# 🧠 MindSight
### AI-Powered Mental Health Prediction & Analytics System

An end-to-end **Machine Learning**, **Statistical Analysis**, and **Flask Web Application** that predicts mental health risk levels using validated clinical assessment scores, demographic information, and lifestyle factors.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web_App-black?style=for-the-badge&logo=flask)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange?style=for-the-badge&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Best_Model-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge)

</p>

**Machine Learning • Data Science • Statistical Analysis • Flask • Python**

</div>

---

# 📖 Overview

**MindSight** is an end-to-end Artificial Intelligence and Machine Learning application that predicts an individual's mental health risk level using validated clinical assessment scores, demographic information, lifestyle factors, and health history.

The project demonstrates the complete Machine Learning lifecycle—from data generation and preprocessing to model training, statistical analysis, deployment, and interactive visualization through a Flask web application.

---

# 🎯 Objectives

- Predict mental health risk levels accurately
- Compare multiple Machine Learning algorithms
- Perform statistical analysis on mental health indicators
- Visualize insights through interactive dashboards
- Deploy the trained model as a Flask web application

---

# ✨ Key Features

| Category | Description |
|----------|-------------|
| 🤖 Machine Learning | Four classification algorithms |
| 📊 Statistical Analysis | Descriptive Statistics, ANOVA, Correlation |
| 📈 Data Visualization | Seven analytical charts |
| 🌐 Web Dashboard | Interactive Flask application |
| 📝 Mental Health Screening | Multi-step prediction form |
| 📦 Model Persistence | Saved models using Joblib |
| 📉 Performance Evaluation | Accuracy, F1 Score, ROC-AUC |

---

# 🧠 Machine Learning Models

| Model | Type | Purpose |
|--------|------|----------|
| Logistic Regression | Linear Classifier | Baseline Model |
| Random Forest | Ensemble Learning | Robust Classification |
| Gradient Boosting | Boosting Ensemble | Improved Accuracy |
| ⭐ XGBoost | Gradient Boosting | Best Performing Model |

---

# 📊 Evaluation Metrics

| Metric | Description |
|---------|-------------|
| Accuracy | Overall prediction accuracy |
| Precision | Positive prediction quality |
| Recall | Detection capability |
| Weighted F1-Score | Balanced performance |
| ROC-AUC | Classification capability |
| Cross Validation | 5-Fold Stratified CV |

---

# 📋 Dataset Features

The prediction model uses **19 mental health indicators**.

| Category | Features |
|----------|----------|
| 👤 Demographics | Age, Gender, Occupation |
| 🏃 Lifestyle | Sleep Hours, Physical Activity, Screen Time, Social Support, Work-Life Balance |
| 🧠 Psychological | Stress Level, Self-Esteem, Coping Skills, Mindfulness |
| 🏥 Clinical | PHQ-9 Score, GAD-7 Score, PSS Score |
| ❤️ Health History | Family History, Chronic Illness, Substance Use, Previous Treatment |

---

# 📈 Statistical Analysis

| Analysis | Purpose |
|----------|---------|
| Descriptive Statistics | Feature Summary |
| Pearson Correlation | Relationship Analysis |
| One-Way ANOVA | Statistical Significance |
| Feature Importance | Model Interpretation |

---

# 📊 Generated Visualizations

| Visualization | Description |
|--------------|-------------|
| Risk Distribution | Mental health category distribution |
| Model Comparison | Accuracy comparison of ML models |
| Correlation Heatmap | Feature relationships |
| Feature Importance | Most influential variables |
| Confusion Matrix | Classification performance |
| ROC Curve | Model evaluation |
| Scatter Plot | Feature relationship analysis |

---

# 🌐 Web Application

| Page | Route | Description |
|------|-------|-------------|
| 🏠 Home | `/` | Project overview and model summary |
| 📊 Dashboard | `/dashboard` | Interactive analytics and charts |
| 📝 Screening | `/screening` | Mental health assessment form |

---

# 🏗 Project Workflow

```text
Dataset Generation
        │
        ▼
Data Preprocessing
        │
        ▼
Statistical Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Model Training
        │
        ▼
Cross Validation
        │
        ▼
Model Evaluation
        │
        ▼
Best Model Selection
        │
        ▼
Flask Deployment
        │
        ▼
Web Application
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/MindSight.git

cd MindSight
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

### Train Models

```bash
python train_model.py
```

### Launch Flask Application

```bash
python app.py
```

Open your browser and visit:

```text
http://localhost:5000
```

---

# 📁 Project Structure

```text
MindSight/
│
├── app.py
├── train_model.py
├── generate_dataset.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── mental_health_dataset.csv
│   ├── model_results.json
│   ├── statistical_summary.json
│   └── summary.json
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── gender_encoder.pkl
│   └── occupation_encoder.pkl
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   └── screening.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
└── LICENSE
```

---

# 🛠 Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python 3.11 |
| Backend | Flask |
| Machine Learning | Scikit-Learn, XGBoost |
| Data Analysis | Pandas, NumPy |
| Statistical Analysis | SciPy |
| Visualization | Matplotlib, Seaborn |
| Model Serialization | Joblib |
| Frontend | HTML5, CSS3, JavaScript |

---

# 📌 Future Enhancements

- Explainable AI (SHAP)
- Deep Learning Models
- PDF Report Generation
- User Authentication
- Docker Support
- Cloud Deployment
- Database Integration
- Real Clinical Dataset

---

# ⚠️ Disclaimer

This project is intended **solely for educational and research purposes**.

The predictions generated by this application should **not** be considered medical advice, diagnosis, or treatment. Individuals experiencing mental health concerns should seek guidance from qualified healthcare professionals.

---

# 🤝 Contributing

Contributions are welcome!

If you'd like to improve this project:

1. Fork the repository
2. Create a new feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

**Daksh_gp**

Machine Learning • Data Science • Python • Flask

---

<div align="center">

### ⭐ If you found this project useful, please consider giving it a Star!

**Built with ❤️ using Python, Machine Learning, Flask, and Data Science**

</div>
