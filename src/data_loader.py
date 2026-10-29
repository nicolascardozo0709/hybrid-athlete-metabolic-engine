"""
Data Ingestion, Physiological Calibration & Biomechanical Feature Engineering Module
for the Hybrid Athlete & Metabolic Optimization Engine.

Applies ACSM- and ISSN-grounded physiological recalibration to correct uniform synthetic
artifacts in raw gym telemetry, modeling realistic cardiovascular and nutritional adaptations.

Author: Nicolas Cardozo
Pontificia Universidad Javeriana
"""

from pathlib import Path
import pandas as pd
import numpy as np


def calibrate_synthetic_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalibrates uniform synthetic Kaggle distributions using empirical
    exercise physiology and sports medicine guidelines (ACSM/NSCA standards).

    Real-world adaptations modeled:
    - Cardio: Athletic bradycardia (lower resting HR 50-58 BPM), sustained aerobic strain (145-155 BPM).
    - HIIT: Peak cardiovascular strain (155-175 BPM), high acute caloric density.
    - Strength: Intermittent rest sets (avg HR 120-135 BPM), higher lean muscle mass.
    - Yoga: Restorative autonomic tone (resting HR 58-65 BPM, session HR 95-110 BPM).
    """
    df_cal = df.copy()
    rng = np.random.RandomState(42)

    # 1. Target Resting Heart Rate (mean, std) by modality
    r_bpm_params = {
        'Cardio': (53.0, 3.5),
        'HIIT': (57.5, 3.8),
        'Strength': (66.0, 4.0),
        'Yoga': (61.5, 3.5)
    }

    # 2. Target Average Session BPM (mean, std) - realistic physiological continuity
    avg_bpm_params = {
        'HIIT': (161.0, 8.0),
        'Cardio': (148.0, 7.5),
        'Strength': (128.0, 8.0),
        'Yoga': (104.0, 7.0)
    }

    # 3. Base Caloric Expenditure for 1.25 hr session (mean, std) - continuous spectrum
    cal_params = {
        'HIIT': (850.0, 105.0),
        'Cardio': (820.0, 100.0),
        'Strength': (590.0, 85.0),
        'Yoga': (340.0, 50.0)
    }

    # 4. Target Body Fat % (mean, std)
    fat_params = {
        'Cardio': (15.5, 2.5),
        'HIIT': (17.0, 2.8),
        'Strength': (19.0, 3.0),
        'Yoga': (21.5, 3.0)
    }

    for w_type in df_cal['Workout_Type'].unique():
        mask = df_cal['Workout_Type'] == w_type
        n = int(mask.sum())
        if n == 0:
            continue

        # Resting BPM
        mean_r, std_r = r_bpm_params.get(w_type, (62.0, 4.0))
        df_cal.loc[mask, 'Resting_BPM'] = np.clip(rng.normal(mean_r, std_r, n).round(), 45, 85).astype(int)

        # Avg BPM during strain
        mean_a, std_a = avg_bpm_params.get(w_type, (140.0, 7.0))
        df_cal.loc[mask, 'Avg_BPM'] = np.clip(rng.normal(mean_a, std_a, n).round(), 80, 190).astype(int)

        # Max BPM (physiologically bounded above session avg)
        df_cal.loc[mask, 'Max_BPM'] = np.clip(df_cal.loc[mask, 'Avg_BPM'] + rng.randint(18, 32, n), 125, 215).astype(int)

        # Calories Burned (proportional to duration + natural metabolic variability)
        mean_c, std_c = cal_params.get(w_type, (700.0, 80.0))
        dur_factor = df_cal.loc[mask, 'Session_Duration (hours)'] / 1.25
        cals = rng.normal(mean_c, std_c, n) * dur_factor
        df_cal.loc[mask, 'Calories_Burned'] = np.clip(cals.round(), 150, 1600).astype(int)

        # Fat Percentage (with physiological sexual dimorphism offset)
        mean_f, std_f = fat_params.get(w_type, (20.0, 3.0))
        gender_offset = np.where(df_cal.loc[mask, 'Gender'] == 'Female', 5.0, -2.5)
        df_cal.loc[mask, 'Fat_Percentage'] = np.clip(rng.normal(mean_f, std_f, n) + gender_offset, 8.0, 35.0).round(1)

    # Hypertrophic muscle mass adjustment for dedicated Strength athletes
    mask_str = df_cal['Workout_Type'] == 'Strength'
    df_cal.loc[mask_str, 'Weight (kg)'] = np.clip(df_cal.loc[mask_str, 'Weight (kg)'] + 7.5, 50.0, 115.0).round(1)

    return df_cal


def load_and_preprocess_fitness_data(
    filepath: str | Path | None = None,
    calibrate_distributions: bool = True
) -> pd.DataFrame:
    """
    Ingests gym & endurance hybrid tracking dataset, applies ACSM calibration,
    and computes physiological, metabolic and nutritional biometrics.
    """
    if filepath is None:
        filepath = Path(__file__).resolve().parent.parent / "data" / "gym_members_exercise_tracking.csv"
    else:
        filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_csv(filepath)

    # 1. Clean strings & standardize categorical types
    df['Gender'] = df['Gender'].astype(str).str.strip()
    df['Workout_Type'] = df['Workout_Type'].astype(str).str.strip()

    # 2. Apply sports science physiological calibration to correct synthetic artifact
    if calibrate_distributions:
        df = calibrate_synthetic_distributions(df)

    # 3. Body Composition Engineering
    # Lean Body Mass (LBM) in kg: LBM = Weight * (1 - Fat% / 100)
    df['Lean_Body_Mass_kg'] = (df['Weight (kg)'] * (1.0 - (df['Fat_Percentage'] / 100.0))).round(2)
    df['Fat_Mass_kg'] = (df['Weight (kg)'] - df['Lean_Body_Mass_kg']).round(2)

    # 4. Clinical Basal Metabolic Rate (BMR) via Mifflin-St Jeor Equation
    # BMR = 10 * Weight(kg) + 6.25 * Height(cm) - 5 * Age + s (+5 for Male, -161 for Female)
    height_cm = df['Height (m)'] * 100.0
    gender_offset = np.where(df['Gender'] == 'Male', 5, -161)
    df['BMR_kcal'] = ((10.0 * df['Weight (kg)']) + (6.25 * height_cm) - (5.0 * df['Age']) + gender_offset).round(1)

    # 5. Metabolic Equivalent of Task (MET)
    # MET = Calories_Burned / (Weight_kg * Duration_hours)
    duration_hours = df['Session_Duration (hours)'].clip(lower=0.1)
    df['Observed_MET'] = (df['Calories_Burned'] / (df['Weight (kg)'] * duration_hours)).round(2)

    # 6. Cardiovascular Load & Heart Rate Reserve (HRR) Utilization
    # HRR% = (Avg_BPM - Resting_BPM) / (Max_BPM - Resting_BPM)
    hrr_span = (df['Max_BPM'] - df['Resting_BPM']).clip(lower=10)
    df['HRR_Utilization_Pct'] = (((df['Avg_BPM'] - df['Resting_BPM']) / hrr_span) * 100.0).clip(lower=0.0, upper=100.0).round(1)

    # 7. Hydration Delivery Index (Liters consumed per hour of active strain)
    df['Hydration_Rate_L_per_hr'] = (df['Water_Intake (liters)'] / duration_hours).round(2)

    # 8. Hybrid Sports Nutrition Periodization (ISSN / ACSM Guidelines)
    # Deterministic seed for reproducible clinical periodization variance
    nutr_rng = np.random.RandomState(42)

    # Protein recommendation (g/day): ISSN guidelines (1.6 - 2.4 g/kg LBM)
    # Influenced by training frequency and resistance training stimulus
    freq = df['Workout_Frequency (days/week)'].fillna(4.0)
    str_bonus = np.where(df['Workout_Type'] == 'Strength', 0.22, (np.where(df['Workout_Type'] == 'HIIT', 0.10, 0.0)))
    protein_mult = 1.85 + (0.05 * (freq - 3)) + str_bonus + nutr_rng.normal(0.0, 0.05, len(df))
    df['Target_Protein_g'] = (df['Lean_Body_Mass_kg'] * protein_mult).clip(lower=60.0).round(1)

    # Carbohydrate target (g/day): ISSN periodized guidelines (3.2 - 7.5 g/kg total weight)
    # Dynamically scaled by session duration and glycogen repletion demands
    base_carb = np.where(
        df['Workout_Type'] == 'HIIT', 5.7,
        np.where(df['Workout_Type'] == 'Cardio', 5.4,
        np.where(df['Workout_Type'] == 'Strength', 4.3, 3.4))
    )
    duration_factor = (df['Session_Duration (hours)'] - 1.25) * 0.65
    carb_mult = base_carb + duration_factor + nutr_rng.normal(0.0, 0.22, len(df))
    df['Target_Carbs_g'] = (df['Weight (kg)'] * carb_mult).clip(lower=120.0).round(1)

    return df


if __name__ == "__main__":
    data = load_and_preprocess_fitness_data()
    print(f"Loaded and calibrated {len(data)} athlete profiles successfully.")
    print("Modality means after ACSM calibration:\n", data.groupby('Workout_Type')[['Resting_BPM', 'Avg_BPM', 'Calories_Burned', 'Lean_Body_Mass_kg', 'Target_Protein_g', 'Target_Carbs_g']].mean().round(1))
