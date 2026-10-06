# 🏋️‍♂️ Hybrid Athlete Metabolic Engine & Sports Nutrition Analytics

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-darkgreen?logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Linear_Algebra-013243?logo=numpy)](https://numpy.org/)
[![Domain](https://img.shields.io/badge/Domain-Concurrent_Training_&_Sports_Nutrition-orange)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end data science and exercise physiology system analyzing **concurrent hybrid athletic training** (Strength, Endurance Cardio, HIIT, and Yoga) across a cohort of **973 active athletes**.

This project investigates the metabolic demands of hybrid training, cardiac adaptations (Resting BPM and heart rate reserve), body composition dynamics (Lean Body Mass vs Fat %), and implements a **multivariate Ordinary Least Squares (OLS) regression engine** from first principles in NumPy to forecast active caloric expenditure ($R^2 = 0.945$). It integrates clinical sports nutrition algorithms (Mifflin-St Jeor BMR, macronutrient periodization, and hydration replacement).

---

## 📌 Executive Summary & Key Results

| Metric / Dimension | Value | Physiological Significance |
| :--- | :--- | :--- |
| **Cohort Sample Size** | **973 Athletes** | Multi-discipline tracking (Strength: 258, Cardio: 255, Yoga: 239, HIIT: 221) |
| **Model Predictive Fit** | **$R^2 = 0.945$** | Exceptionally high variance explained via linear physiological modeling |
| **Caloric Error Margin** | **MAE = 51.5 kcal** | Precision forecasting for energy balance & nutritional planning |
| **Mean Lean Body Mass** | **55.7 kg** | Segmented by bioelectrical / DEXA-equivalent body fat percentages |
| **Mean Basal Metabolic Rate** | **1,548 kcal/day** | Calculated via clinical Mifflin-St Jeor formulation |
| **Concurrent Interference Insight** | **Cardio vs Strength** | Strength maintains competitive MET expenditure (10.7) with preserved cardiovascular efficiency |

---

## 🔬 Architecture & Methodology

```
    Cohort Sensor Telemetry & Biometrics (n = 973)
                          │
                          ▼
    [1] Data Engineering & Clinical Formulas (src/data_loader.py)
        ├── Body Composition: Lean Body Mass (LBM) & Fat Mass (kg)
        ├── Mifflin-St Jeor Basal Metabolic Rate (BMR)
        ├── Metabolic Equivalent of Task (MET) calculation
        └── Heart Rate Reserve (HRR) Utilization & Hydration Rate (L/hr)
                          │
                          ▼
    [2] Concurrent Modality Profiling (src/metabolic_engine.py)
        ├── Benchmarking Resting BPM across Strength vs Cardio vs HIIT
        ├── Session strain & energy output variance
        └── Sports Nutrition Prescription (Protein g/kg LBM, Glycogen repletion)
                          │
                          ▼
    [3] First-Principles NumPy OLS Regression Engine
        ├── Features: Duration (hrs), Avg BPM, Lean Mass, Fat %, Modality
        └── Solution: β̂ = (XᵀX)⁻¹ Xᵀy  --> R² = 0.945, MAE = 51.5 kcal
                          │
                          ▼
    [4] High-Resolution Publication Dashboards (src/visualizer.py)
```

---

## 📊 Analytical Insights & Visualizations

### 1. Concurrent Training Biometrics & Cardiovascular Adaptation
Comparing cardiac strain, energy output, and body composition across the four disciplines:

![Concurrent Training Biometrics](outputs/concurrent_training_biometrics.png)

> **Finding:** Endurance cardio and HIIT athletes demonstrated lower resting heart rates (~61.9 BPM), while strength-trained athletes maintained higher lean muscle mass and equivalent acute session energy expenditure.

---

### 2. Metabolic Energy Expenditure vs Workload & Lean Mass
Visualizing caloric burn as a function of duration, heart rate intensity, and lean mass:

![Metabolic Rate vs Workload](outputs/metabolic_rate_vs_workload.png)

> **Finding:** Caloric burn scales primarily with session duration and cardiovascular strain, but baseline metabolic rate is heavily anchored to active lean muscle mass.

---

### 3. Sports Nutrition & Hydration Target Matrix
Periodized macronutrient intake (protein for muscle protein synthesis + carbs for glycogen replenishment) alongside intra-workout fluid replenishment:

![Nutrition Hydration Matrix](outputs/nutrition_hydration_matrix.png)

---

## 🧮 Mathematical Modeling: OLS Caloric Prediction

Implemented analytically in pure NumPy without third-party black-box wrappers:

$$\mathbf{\hat{\beta}} = \left(\mathbf{X}^T \mathbf{X}\right)^{-1} \mathbf{X}^T \mathbf{y}$$

### Regression Weights:
* **Intercept ($\beta_0$):** `-873.52`
* **Session Duration ($\beta_1$):** `+684.23 kcal/hr`
* **Heart Rate Intensity ($\beta_2$):** `+6.16 kcal/bpm`
* **Lean Body Mass ($\beta_3$):** `+1.53 kcal/kg`
* **Fat Percentage ($\beta_4$):** `-2.12 kcal/%`
* **Endurance/HIIT Indicator ($\beta_5$):** `+1.99 kcal`

### Real-World Simulated Prescription:
For an athlete weighing **72 kg** with **14% body fat** performing a **1.25 hr** cardio/hybrid session at **152 BPM**:
* **Predicted Caloric Expenditure:** `985.6 kcal`
* **Lean Body Mass:** `61.9 kg`
* **Prescribed Daily Protein:** `123.8 g` (with a `25.0 g` post-workout recovery pulse)
* **Prescribed Carbohydrate Target:** `396.0 g`
* **Intra-workout Hydration Target:** `1.0 Liter`

---

## 🚀 Quickstart

### 1. Clone & Setup
```bash
git clone https://github.com/nicolascardozo0709/hybrid-athlete-metabolic-engine.git
cd hybrid-athlete-metabolic-engine
pip install -r requirements.txt
```

### 2. Run Pipeline
```bash
python main.py
```
*(On Windows, you can also simply double-click `run_pipeline.bat`)*.

---

## 📂 Repository Structure

```
hybrid-athlete-metabolic-engine/
├── data/
│   └── gym_members_exercise_tracking.csv   # 973 athlete cohort dataset
├── src/
│   ├── data_loader.py                      # Data cleaning & body composition metrics
│   ├── metabolic_engine.py                 # NumPy OLS engine & sports nutrition rules
│   └── visualizer.py                       # Matplotlib & Seaborn dashboard generators
├── outputs/
│   ├── concurrent_training_biometrics.png
│   ├── metabolic_rate_vs_workload.png
│   └── nutrition_hydration_matrix.png
├── main.py                                 # Automated execution pipeline
├── run_pipeline.bat                        # Double-click Windows batch runner
├── requirements.txt                        # Pinned dependencies
├── .gitignore                              # Git exclusion rules
└── README.md                               # Project documentation
```

---

## 👤 Author
**Nicolás Cardozo**  
*Undergraduate Data Science Student*  
Pontificia Universidad Javeriana — Bogotá, Colombia  
*Hybrid Athlete, Distance Runner & Exercise Physiology Technologist*  
[LinkedIn Profile](https://www.linkedin.com/in/nicolas-cardozo-a779b331a/) • [GitHub](https://github.com/nicolascardozo0709)
