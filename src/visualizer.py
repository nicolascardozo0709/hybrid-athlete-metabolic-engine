"""
Publication-Grade Visualizations for the Hybrid Athlete
& Sports Nutrition Analytics Platform.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_all_hybrid_visuals(df: pd.DataFrame, output_dir: str | Path | None = None) -> list[str]:
    """
    Renders 3 high-impact charts in output_dir.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "outputs"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    # Modern aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    palette = {'Strength': '#2B5C8F', 'Cardio': '#D9534F', 'HIIT': '#F0AD4E', 'Yoga': '#5CB85C'}

    # -------------------------------------------------------------
    # 1. Figure 1: Concurrent Training Biometrics Comparison
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#FAF9F6')

    # Subplot A: Resting BPM by Modality
    sns.barplot(data=df, x='Workout_Type', y='Resting_BPM', hue='Workout_Type', legend=False, ax=axes[0, 0], palette=palette, errorbar=None, capsize=0.1)
    axes[0, 0].set_title("Resting Heart Rate Adaptation by Discipline", fontsize=12, fontweight='bold', pad=10)
    axes[0, 0].set_ylabel("Resting BPM (Cardiovascular Efficiency)")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_facecolor('#FFFFFF')

    # Subplot B: Avg BPM During Strain
    sns.boxplot(data=df, x='Workout_Type', y='Avg_BPM', hue='Workout_Type', legend=False, ax=axes[0, 1], palette=palette)
    axes[0, 1].set_title("Session Cardiovascular Strain (Avg BPM)", fontsize=12, fontweight='bold', pad=10)
    axes[0, 1].set_ylabel("Heart Rate (BPM)")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_facecolor('#FFFFFF')

    # Subplot C: Calories Burned Distribution
    sns.violinplot(data=df, x='Workout_Type', y='Calories_Burned', hue='Workout_Type', legend=False, ax=axes[1, 0], palette=palette, cut=0)
    axes[1, 0].set_title("Total Energy Expenditure (kcal) per Session", fontsize=12, fontweight='bold', pad=10)
    axes[1, 0].set_ylabel("Calories Burned (kcal)")
    axes[1, 0].set_xlabel("Athletic Modality")
    axes[1, 0].set_facecolor('#FFFFFF')

    # Subplot D: Body Fat Percentage vs Lean Mass
    sns.scatterplot(data=df, x='Lean_Body_Mass_kg', y='Fat_Percentage', hue='Workout_Type', ax=axes[1, 1], palette=palette, alpha=0.7, s=50)
    axes[1, 1].set_title("Body Composition: Lean Mass vs Fat %", fontsize=12, fontweight='bold', pad=10)
    axes[1, 1].set_ylabel("Body Fat (%)")
    axes[1, 1].set_xlabel("Lean Body Mass (kg)")
    axes[1, 1].set_facecolor('#FFFFFF')

    plt.suptitle("HYBRID ATHLETE CONCURRENT TRAINING BIOMETRICS (n = 973)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fig1_path = output_dir / "concurrent_training_biometrics.png"
    plt.savefig(fig1_path, dpi=200, bbox_inches='tight')
    plt.close()
    generated_files.append(str(fig1_path))

    # -------------------------------------------------------------
    # 2. Figure 2: Metabolic Rate & Energy Expenditure vs Workload
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#FAF9F6')
    ax.set_facecolor('#FFFFFF')

    scatter = ax.scatter(
        df['Session_Duration (hours)'],
        df['Calories_Burned'],
        c=df['Avg_BPM'],
        cmap='coolwarm',
        s=df['Lean_Body_Mass_kg'] * 1.5,
        alpha=0.75,
        edgecolors='none'
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Cardiovascular Strain (Average BPM)", fontsize=11, fontweight='bold')

    # Overall linear trend
    sns.regplot(
        data=df,
        x='Session_Duration (hours)',
        y='Calories_Burned',
        scatter=False,
        ax=ax,
        color='#111111',
        label='Metabolic Regression Trend (OLS)',
        line_kws={'linewidth': 2, 'linestyle': '--'}
    )

    ax.set_title("Metabolic Expenditure as a Function of Duration, Heart Rate & Lean Mass", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Session Duration (Hours)", fontsize=12)
    ax.set_ylabel("Energy Burned (kcal)", fontsize=12)
    ax.legend(loc='upper left', frameon=True)

    fig2_path = output_dir / "metabolic_rate_vs_workload.png"
    plt.savefig(fig2_path, dpi=200, bbox_inches='tight')
    plt.close()
    generated_files.append(str(fig2_path))

    # -------------------------------------------------------------
    # 3. Figure 3: Sports Nutrition & Hydration Dynamics
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#FAF9F6')
    ax.set_facecolor('#FFFFFF')

    sns.scatterplot(
        data=df,
        x='Target_Protein_g',
        y='Target_Carbs_g',
        hue='Workout_Type',
        size='Water_Intake (liters)',
        sizes=(40, 200),
        palette=palette,
        alpha=0.8,
        ax=ax
    )

    ax.set_title("Sports Nutrition Target Matrix: Protein vs Carbohydrates vs Hydration Demand", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Prescribed Daily Protein Target (g) [2.0 g / kg Lean Mass]", fontsize=12)
    ax.set_ylabel("Prescribed Daily Carbohydrate Target (g) [Periodized by Modality]", fontsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)

    fig3_path = output_dir / "nutrition_hydration_matrix.png"
    plt.savefig(fig3_path, dpi=200, bbox_inches='tight')
    plt.close()
    generated_files.append(str(fig3_path))

    return generated_files
