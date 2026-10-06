"""
Main Execution Script for the Hybrid Athlete Metabolic
and Sports Nutrition Optimization Engine.
"""

from pathlib import Path
from src.data_loader import load_and_preprocess_fitness_data
from src.metabolic_engine import (
    analyze_concurrent_modalities,
    train_metabolic_prediction_engine,
    generate_hybrid_nutrition_guidelines
)
from src.visualizer import generate_all_hybrid_visuals

def main():
    print("=" * 68)
    print("   HYBRID ATHLETE METABOLIC & SPORTS NUTRITION ANALYTICS ENGINE")
    print("=" * 68)
    print()

    # 1. Ingest and engineer biometrics
    print("[1/4] Ingesting cohort telemetry & computing body composition metrics...")
    df = load_and_preprocess_fitness_data()
    print(f"      -> Successfully processed {len(df)} athlete profiles.")
    print(f"      -> Mean Lean Body Mass: {df['Lean_Body_Mass_kg'].mean():.1f} kg | Mean BMR: {df['BMR_kcal'].mean():.0f} kcal/day")
    print()

    # 2. Modality profiling
    print("[2/4] Benchmarking Concurrent & Hybrid Training Modalities...")
    modalities_summary = analyze_concurrent_modalities(df)
    print(modalities_summary[['Athletes_Count', 'Mean_Resting_BPM', 'Mean_Avg_BPM', 'Mean_Calories_Burned', 'Mean_MET']])
    print()

    # 3. Train pure NumPy OLS model
    print("[3/4] Fitting First-Principles Multivariate OLS Caloric Engine (NumPy)...")
    model, metrics = train_metabolic_prediction_engine(df)
    print(f"      - Coefficient of Determination (R²): {metrics['R2']:.3f}")
    print(f"      - Mean Absolute Error (MAE): {metrics['MAE']:.1f} kcal")
    print(f"      - Root Mean Squared Error (RMSE): {metrics['RMSE']:.1f} kcal")
    print("      - Model Equation Weights:")
    print(f"         * Intercept (beta_0): {model.intercept:.2f}")
    for name, weight in zip(model.feature_names, model.coefficients):
        print(f"         * {name}: {weight:+.3f}")
    print()

    # 4. Athlete Prescriptive Prescription Simulation
    print("      [+] Simulated Hybrid Athlete Prescription:")
    print("          Profile: 72.0 kg | 14.0% Body Fat | 1.25 hrs Session | Avg BPM 152 | Discipline: Cardio/HIIT")
    rx = generate_hybrid_nutrition_guidelines(
        weight_kg=72.0,
        fat_pct=14.0,
        session_duration_hrs=1.25,
        avg_bpm=152.0,
        workout_type="Cardio",
        model=model
    )
    for k, v in rx.items():
        print(f"          - {k.replace('_', ' ')}: {v}")
    print()

    # 5. Visualizer
    print("[4/4] Rendering high-resolution analytics dashboards...")
    visuals = generate_all_hybrid_visuals(df)
    for v in visuals:
        print(f"      -> Generated: {Path(v).name}")
    print()

    print("=" * 68)
    print("  Execution completed! Visuals saved in: outputs/")
    print("=" * 68)

if __name__ == "__main__":
    main()
