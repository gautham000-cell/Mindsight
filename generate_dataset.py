"""
Mental Health Prediction System - Enhanced Dataset Generator
Generates a highly realistic synthetic dataset with proper inter-feature
correlations, clinically grounded distributions, and strong class separability
for high model accuracy.
Dataset size: 7000 samples (increased from 2000)
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 50000  # Increased sample size for master-level model accuracy

def generate_dataset():
    print(f"Generating enhanced synthetic mental health dataset (N={N})...")

    # =========================================================
    # 1. DEMOGRAPHICS
    # =========================================================
    age        = np.random.randint(18, 65, N)
    gender     = np.random.choice(['Male', 'Female', 'Non-binary'], N, p=[0.48, 0.48, 0.04])
    occupation = np.random.choice(
        ['Student', 'Employed', 'Self-employed', 'Unemployed', 'Retired'],
        N, p=[0.30, 0.40, 0.15, 0.10, 0.05]
    )

    # =========================================================
    # 2. LATENT RISK FACTOR (hidden variable driving correlations)
    #    This is the key improvement: all features are drawn from
    #    a shared latent risk 'u' so that high-risk individuals
    #    consistently score high across ALL risk indicators.
    # =========================================================
    u = np.random.uniform(0, 1, N)   # latent risk (0=healthy, 1=high risk)

    # ---- Helper: correlated normal centred by u ----
    def corr_norm(low_mean, high_mean, std, u_arr, noise=0.6):
        """Draw from normal whose mean is linearly interpolated by u."""
        mean = low_mean + (high_mean - low_mean) * u_arr
        return np.clip(mean + np.random.normal(0, std * noise, len(u_arr)),
                       low_mean - 3*std, high_mean + 3*std)

    # =========================================================
    # 3. LIFESTYLE FACTORS  (high u → worse lifestyle)
    # =========================================================
    # Sleep hours: healthy=8h, high-risk=4.5h
    sleep_hours      = np.clip(corr_norm(8.5, 4.5, 1.5, u),       2.0, 12.0)
    # Physical activity: healthy=7, high-risk=2
    physical_activity= np.clip(corr_norm(7.5, 2.0, 2.0, u),       0.0, 10.0)
    # Social support: healthy=8, high-risk=2
    social_support   = np.clip(corr_norm(8.0, 2.0, 1.8, u),       0.0, 10.0)
    # Work-life balance: healthy=7.5, high-risk=2.5
    work_life_balance= np.clip(corr_norm(7.5, 2.5, 1.8, u),       0.0, 10.0)
    # Screen time: healthy=4h, high-risk=10h
    screen_time      = np.clip(corr_norm(4.0, 10.0, 2.0, u),      0.0, 16.0)

    # =========================================================
    # 4. PSYCHOLOGICAL FACTORS  (high u → worse psychology)
    # =========================================================
    # Stress: healthy=2, high-risk=8.5
    stress_level     = np.clip(corr_norm(2.0, 8.5, 1.5, u),       0.0, 10.0)
    # Self-esteem: healthy=8, high-risk=2 (inverse)
    self_esteem      = np.clip(corr_norm(8.0, 2.0, 1.8, u),       0.0, 10.0)
    # Coping skills: healthy=8, high-risk=2 (inverse)
    coping_skills    = np.clip(corr_norm(8.0, 2.0, 1.8, u),       0.0, 10.0)
    # Mindfulness: healthy=7, high-risk=1 (inverse)
    mindfulness      = np.clip(corr_norm(7.0, 1.0, 2.0, u),       0.0, 10.0)

    # =========================================================
    # 5. HEALTH & CLINICAL FLAGS  (high u → more likely Yes)
    # =========================================================
    family_history    = (np.random.uniform(0, 1, N) < (0.15 + 0.55 * u)).astype(int)
    chronic_illness   = (np.random.uniform(0, 1, N) < (0.10 + 0.45 * u)).astype(int)
    substance_use     = (np.random.uniform(0, 1, N) < (0.05 + 0.50 * u)).astype(int)
    previous_treatment= (np.random.uniform(0, 1, N) < (0.05 + 0.45 * u)).astype(int)

    # =========================================================
    # 6. VALIDATED CLINICAL QUESTIONNAIRE SCORES
    #    PHQ-9 (0–27), GAD-7 (0–21), PSS (0–40)
    #    Drawn from clinical distributions parameterised by u
    # =========================================================
    # PHQ-9: low-risk μ=2, high-risk μ=18
    phq9_raw  = corr_norm(2.0, 18.0, 4.0, u, noise=0.7)
    phq9_score= np.clip(np.round(phq9_raw), 0, 27).astype(int)

    # GAD-7: low-risk μ=1.5, high-risk μ=15
    gad7_raw  = corr_norm(1.5, 15.0, 3.5, u, noise=0.7)
    gad7_score= np.clip(np.round(gad7_raw), 0, 21).astype(int)

    # PSS: low-risk μ=6, high-risk μ=34
    pss_raw   = corr_norm(6.0, 34.0, 6.0, u, noise=0.7)
    pss_score = np.clip(np.round(pss_raw), 0, 40).astype(int)

    # =========================================================
    # 7. COMPOSITE RISK SCORE  (derived directly from u + features)
    #    This makes labels tightly aligned with actual features.
    # =========================================================
    risk_score = (
        0.20 * (phq9_score / 27.0 * 10) +   # Depression (strongest signal)
        0.16 * (gad7_score / 21.0 * 10) +   # Anxiety
        0.12 * (pss_score  / 40.0 * 10) +   # Stress scale
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

    # Do not Min-Max normalize. The raw sum is internally weighted to be roughly 0-10.
    risk_score = np.clip(risk_score + np.random.normal(0, 0.15, N), 0, 10)

    # =========================================================
    # 8. RISK LABELS — Hardcoded thresholds to match the UI visual scales
    # =========================================================
    def assign_risk(s):
        if s < 3.5:  return 'Low'
        if s < 6.5:  return 'Moderate'
        return 'High'

    risk_level = np.array([assign_risk(s) for s in risk_score])

    # =========================================================
    # 9. BUILD DATAFRAME
    # =========================================================
    df = pd.DataFrame({
        'age':                 age,
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
        'risk_score':          np.round(risk_score, 2),
        'risk_level':          risk_level
    })

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/mental_health_dataset.csv', index=False)
    print(f"Dataset saved: {len(df)} records, {df.shape[1]} features")
    print(f"Risk distribution:\n{df['risk_level'].value_counts()}")
    dist = df['risk_level'].value_counts()
    for lvl in ['Low', 'Moderate', 'High']:
        cnt = dist.get(lvl, 0)
        print(f"   {lvl:10s}: {cnt:5d} ({cnt/N*100:.1f}%)")
    return df


if __name__ == '__main__':
    generate_dataset()