"""
Data Ingestion and Biomechanical Feature Engineering Module
for the Hybrid Athlete & Metabolic Optimization Engine.
"""

from pathlib import Path
import pandas as pd
import numpy as np

def load_and_preprocess_fitness_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Ingests gym & endurance hybrid tracking dataset, cleans records,
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

    # 2. Body Composition Engineering
    # Lean Body Mass (LBM) in kg: LBM = Weight * (1 - Fat% / 100)
    df['Lean_Body_Mass_kg'] = df['Weight (kg)'] * (1.0 - (df['Fat_Percentage'] / 100.0))
    df['Fat_Mass_kg'] = df['Weight (kg)'] - df['Lean_Body_Mass_kg']

    # 3. Clinical Basal Metabolic Rate (BMR) via Mifflin-St Jeor Equation
    # BMR = 10 * Weight(kg) + 6.25 * Height(cm) - 5 * Age + s (+5 for Male, -161 for Female)
    height_cm = df['Height (m)'] * 100.0
    gender_offset = np.where(df['Gender'] == 'Male', 5, -161)
    df['BMR_kcal'] = (10.0 * df['Weight (kg)']) + (6.25 * height_cm) - (5.0 * df['Age']) + gender_offset

    # 4. Metabolic Equivalent of Task (MET)
    # MET = Calories_Burned / (Weight_kg * Duration_hours)
    # Note: 1 MET = 1 kcal / (kg * hour)
    duration_hours = df['Session_Duration (hours)'].clip(lower=0.1)
    df['Observed_MET'] = df['Calories_Burned'] / (df['Weight (kg)'] * duration_hours)

    # 5. Cardiovascular Load & Heart Rate Reserve (HRR) Utilization
    # HRR% = (Avg_BPM - Resting_BPM) / (Max_BPM - Resting_BPM)
    hrr_span = (df['Max_BPM'] - df['Resting_BPM']).clip(lower=10)
    df['HRR_Utilization_Pct'] = ((df['Avg_BPM'] - df['Resting_BPM']) / hrr_span) * 100.0
    df['HRR_Utilization_Pct'] = df['HRR_Utilization_Pct'].clip(lower=0.0, upper=100.0)

    # 6. Hydration Delivery Index (Liters consumed per hour of active strain)
    df['Hydration_Rate_L_per_hr'] = df['Water_Intake (liters)'] / duration_hours

    # 7. Hybrid Sports Nutrition Recommended Macronutrients
    # Protein recommendation for concurrent/hybrid training: 2.0g per kg of LBM
    df['Target_Protein_g'] = df['Lean_Body_Mass_kg'] * 2.0
    # Carbohydrate target based on intensity (g/kg):
    # Cardio/HIIT -> 5.5 g/kg, Strength/Yoga -> 4.0 g/kg
    carb_multiplier = np.where(df['Workout_Type'].isin(['Cardio', 'HIIT']), 5.5, 4.0)
    df['Target_Carbs_g'] = df['Weight (kg)'] * carb_multiplier

    return df

if __name__ == "__main__":
    data = load_and_preprocess_fitness_data()
    print(f"Loaded {len(data)} athlete profiles successfully.")
    print("Engineered features sample:\n", data[['Workout_Type', 'Lean_Body_Mass_kg', 'BMR_kcal', 'Observed_MET', 'Target_Protein_g']].head())
