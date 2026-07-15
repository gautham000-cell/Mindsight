import pandas as pd
import numpy as np
import os

def process_kaggle_dataset():
    print("Loading Kaggle 'Student Mental Health' dataset...")
    # Corrected path: the original file is 'Mental Health Dataset.csv'
    df_k = pd.read_csv('data/Mental Health Dataset.csv')
    
    # Kaggle cols:
    # ['Timestamp', 'Choose your gender', 'Age', 'What is your course?', 'Your current year of Study', 'What is your CGPA?', 'Marital status', 'Do you have Depression?', 'Do you have Anxiety?', 'Do you have Panic attack?', 'Did you seek any specialist for a treatment?']
    
    # Clean the Kaggle data
    df_k = df_k.rename(columns={
        'Choose your gender': 'gender',
        'Age': 'age',
        'Do you have Depression?': 'depression',
        'Do you have Anxiety?': 'anxiety',
        'Do you have Panic attack?': 'panic',
        'Did you seek any specialist for a treatment?': 'treatment',
        'Marital status': 'marital'
    })
    
    # Drop rows with NaN age
    df_k = df_k.dropna(subset=['age'])
    
    n_original = len(df_k)
    print(f"Kaggle dataset loaded: {n_original} records. Bootstrapping to 7000 to train robust models...")
    
    # Bootstrap to 7000 to match the previous robust model expectations
    df_b = df_k.sample(n=7000, replace=True, random_state=42).reset_index(drop=True)
    N = 7000
    
    # Now we map the Kaggle dataset variables onto our UI's required 19 base features!
    # By starting with REAL Kaggle labels, we ensure the correlations are grounded in the real student dataset.
    
    # Age & Gender
    age = df_b['age'].values
    gender = df_b['gender'].values  # 'Male', 'Female'

    # Occupation (Assuming all are students based on the dataset!)
    occupation = np.array(['Student'] * N)
    
    # Convert Yes/No to boolean for easier logic
    has_dep = (df_b['depression'].str.strip().str.lower() == 'yes').values
    has_anx = (df_b['anxiety'].str.strip().str.lower() == 'yes').values
    has_pan = (df_b['panic'].str.strip().str.lower() == 'yes').values
    treatment = (df_b['treatment'].str.strip().str.lower() == 'yes').astype(int).values
    
    # Latent risk u defined by the REAL Kaggle tags!
    # If they have depression, anxiety, and panic = high risk (u ~ 0.9)
    # If none = low risk (u ~ 0.1)
    base_u = (has_dep.astype(float) + has_anx.astype(float) + has_pan.astype(float)) / 3.0
    u = np.clip(base_u + np.random.normal(0, 0.2, N), 0, 1)

    # ---- Helper: correlated normal centred by u ----
    def corr_norm(low_mean, high_mean, std, u_arr, noise=0.6):
        mean = low_mean + (high_mean - low_mean) * u_arr
        return np.clip(mean + np.random.normal(0, std * noise, len(u_arr)), low_mean - 3*std, high_mean + 3*std)
        
    # Generate lifestyle based on Kaggle's actual latent risk signatures (u)
    sleep_hours      = np.clip(corr_norm(8.0, 4.5, 1.5, u),       2.0, 12.0)
    physical_activity= np.clip(corr_norm(7.5, 2.0, 2.0, u),       0.0, 10.0)
    social_support   = np.clip(corr_norm(8.0, 3.0, 1.8, u),       0.0, 10.0)
    work_life_balance= np.clip(corr_norm(7.5, 3.5, 1.8, u),       0.0, 10.0)
    screen_time      = np.clip(corr_norm(5.0, 10.0, 2.0, u),      0.0, 16.0)

    stress_level     = np.clip(corr_norm(3.0, 8.5, 1.5, u),       0.0, 10.0)
    self_esteem      = np.clip(corr_norm(7.5, 2.5, 1.8, u),       0.0, 10.0)
    coping_skills    = np.clip(corr_norm(7.5, 3.0, 1.8, u),       0.0, 10.0)
    mindfulness      = np.clip(corr_norm(7.0, 2.0, 2.0, u),       0.0, 10.0)

    family_history    = (np.random.uniform(0, 1, N) < (0.15 + 0.55 * u)).astype(int)
    chronic_illness   = (np.random.uniform(0, 1, N) < (0.10 + 0.45 * u)).astype(int)
    substance_use     = (np.random.uniform(0, 1, N) < (0.05 + 0.50 * u)).astype(int)
    previous_treatment= treatment # Directly from Kaggle!

    # Clinical validation scores mapped directly from Kaggle 'Yes/No' answers!
    # If Kaggle says 'Yes' to Depression, PHQ9 must be high (>10).
    phq9_raw = np.where(has_dep, np.random.normal(16, 4, N), np.random.normal(4, 3, N))
    phq9_score = np.clip(np.round(phq9_raw), 0, 27).astype(int)
    
    gad7_raw = np.where(has_anx, np.random.normal(15, 3, N), np.random.normal(3, 2, N))
    gad7_score = np.clip(np.round(gad7_raw), 0, 21).astype(int)
    
    pss_raw = np.where(has_pan, np.random.normal(28, 5, N), np.random.normal(12, 4, N))
    pss_score = np.clip(np.round(pss_raw), 0, 40).astype(int)

    # Recalculate robust Risk Score based on the Mindsight UI algorithm:
    risk_score = (
        0.20 * (phq9_score / 27.0 * 10) +  
        0.16 * (gad7_score / 21.0 * 10) +  
        0.12 * (pss_score  / 40.0 * 10) +  
        0.10 * stress_level +
        0.08 * (10 - sleep_hours) +
        0.07 * (10 - social_support) +
        0.07 * (10 - coping_skills) +
        0.06 * (10 - self_esteem) +
        0.04 * (10 - work_life_balance) +
        0.03 * family_history * 10 +
        0.03 * substance_use  * 10 +
        0.02 * chronic_illness * 10 +
        0.02 * screen_time
    )

    # Directly round and set thresholds
    risk_score = np.clip(np.round(risk_score, 2), 0, 10)

    def assign_risk(s):
        if s < 3.5:  return 'Low'
        if s < 6.5:  return 'Moderate'
        return 'High'

    risk_level = np.array([assign_risk(s) for s in risk_score])

    df_final = pd.DataFrame({
        'age':                 age.astype(int),
        'gender':              gender,
        'occupation':          occupation,
        'sleep_hours':         np.round(sleep_hours,       1),
        'physical_activity':   np.round(physical_activity, 1),
        'social_support':      np.round(social_support,    1),
        'work_life_balance':   np.round(work_life_balance, 1),
        'screen_time':         np.round(screen_time,       1),
        'stress_level':        np.round(stress_level,      1),
        'self_esteem':         np.round(self_esteem,       1),
        'coping_skills':       np.round(coping_skills,     1),
        'mindfulness_practice':np.round(mindfulness,       1),
        'family_history':      family_history,
        'chronic_illness':     chronic_illness,
        'substance_use':       substance_use,
        'previous_treatment':  previous_treatment,
        'phq9_score':          phq9_score,
        'gad7_score':          gad7_score,
        'pss_score':           pss_score,
        'risk_score':          risk_score,
        'risk_level':          risk_level
    })

    df_final.to_csv('data/mental_health_dataset.csv', index=False)
    print("Mapped real Kaggle dataset to Mindsight architecture. Overwritten data/mental_health_dataset.csv successfully.")

if __name__ == "__main__":
    process_kaggle_dataset()
