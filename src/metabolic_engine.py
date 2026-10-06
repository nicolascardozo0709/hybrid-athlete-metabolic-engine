"""
Metabolic Modeling and First-Principles OLS Caloric Prediction Engine
for Concurrent & Hybrid Athletic Performance.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

class PureNumPyMetabolicRegressor:
    """
    Multivariate Ordinary Least Squares (OLS) Regression Engine
    implemented from first principles using NumPy matrix algebra.
    Solves beta = (X^T X)^(-1) X^T y.
    """
    def __init__(self):
        self.coefficients = None
        self.intercept = 0.0
        self.feature_names = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> "PureNumPyMetabolicRegressor":
        self.feature_names = feature_names
        X_mat = np.asarray(X, dtype=np.float64)
        y_vec = np.asarray(y, dtype=np.float64)

        # Prepend column of ones for intercept beta_0
        n_samples = X_mat.shape[0]
        X_design = np.hstack([np.ones((n_samples, 1)), X_mat])

        # Analytical solution via Normal Equations
        XtX = np.dot(X_design.T, X_design)
        Xty = np.dot(X_design.T, y_vec)
        beta_hat = np.linalg.solve(XtX, Xty)

        self.intercept = float(beta_hat[0])
        self.coefficients = beta_hat[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_mat = np.asarray(X, dtype=np.float64)
        return self.intercept + np.dot(X_mat, self.coefficients)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X)
        errors = y - preds
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        return {"R2": r2, "MAE": mae, "RMSE": rmse}


def analyze_concurrent_modalities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates physiological metrics across training modalities
    (Strength, Cardio, HIIT, Yoga) for hybrid training benchmarking.
    """
    summary = df.groupby('Workout_Type').agg(
        Athletes_Count=('Age', 'count'),
        Mean_Resting_BPM=('Resting_BPM', 'mean'),
        Mean_Avg_BPM=('Avg_BPM', 'mean'),
        Mean_Duration_Hrs=('Session_Duration (hours)', 'mean'),
        Mean_Calories_Burned=('Calories_Burned', 'mean'),
        Mean_MET=('Observed_MET', 'mean'),
        Mean_Fat_Pct=('Fat_Percentage', 'mean'),
        Mean_Water_Liters=('Water_Intake (liters)', 'mean'),
        Mean_Lean_Mass_kg=('Lean_Body_Mass_kg', 'mean')
    ).round(2)
    return summary


def train_metabolic_prediction_engine(df: pd.DataFrame) -> Tuple[PureNumPyMetabolicRegressor, Dict[str, float]]:
    """
    Trains pure NumPy regression model predicting Calories_Burned from
    session duration, mean heart rate, lean mass, fat %, and high-intensity indicator.
    """
    features = ['Session_Duration (hours)', 'Avg_BPM', 'Lean_Body_Mass_kg', 'Fat_Percentage']
    X = df[features].copy()
    # Add binary feature: High intensity endurance (Cardio or HIIT)
    X['Is_Endurance_HIIT'] = df['Workout_Type'].isin(['Cardio', 'HIIT']).astype(float)
    
    feature_cols = ['Duration_Hrs', 'Avg_BPM', 'Lean_Mass_kg', 'Fat_Pct', 'Is_Endurance_HIIT']
    X_mat = X.values
    y_vec = df['Calories_Burned'].values

    model = PureNumPyMetabolicRegressor()
    model.fit(X_mat, y_vec, feature_cols)
    metrics = model.evaluate(X_mat, y_vec)
    return model, metrics


def generate_hybrid_nutrition_guidelines(
    weight_kg: float,
    fat_pct: float,
    session_duration_hrs: float,
    avg_bpm: float,
    workout_type: str,
    model: PureNumPyMetabolicRegressor
) -> Dict[str, Any]:
    """
    Provides clinical sports nutrition and energy balance prescription
    for an individual athlete session.
    """
    lean_mass = weight_kg * (1.0 - (fat_pct / 100.0))
    is_cardio_hiit = 1.0 if workout_type in ['Cardio', 'HIIT'] else 0.0
    
    # Feature vector: [Duration_Hrs, Avg_BPM, Lean_Mass_kg, Fat_Pct, Is_Endurance_HIIT]
    x_input = np.array([[session_duration_hrs, avg_bpm, lean_mass, fat_pct, is_cardio_hiit]])
    predicted_burn = float(model.predict(x_input)[0])

    # Nutritional targets
    protein_daily_g = lean_mass * 2.0
    protein_post_workout_g = min(40.0, max(25.0, lean_mass * 0.4))
    carb_multiplier = 5.5 if is_cardio_hiit else 4.0
    carbs_daily_g = weight_kg * carb_multiplier
    water_recommended_L = session_duration_hrs * (0.8 if avg_bpm > 145 else 0.6)

    return {
        "Predicted_Caloric_Expenditure_kcal": round(predicted_burn, 1),
        "Lean_Body_Mass_kg": round(lean_mass, 1),
        "Daily_Protein_Target_g": round(protein_daily_g, 1),
        "Post_Workout_Protein_Pulse_g": round(protein_post_workout_g, 1),
        "Daily_Carbohydrate_Target_g": round(carbs_daily_g, 1),
        "Workout_Hydration_Target_L": round(water_recommended_L, 2)
    }
