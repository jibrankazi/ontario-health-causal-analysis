import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import os
import sys
from causalimpact import CausalImpact

# Set random seed for reproducibility
np.random.seed(42)

# Configurable path
DATA_PATH = os.environ.get("CAUSAL_DATA_PATH", "data/ontario_cases.csv")

# 1. Load and Prepare Data
try:
    df = pd.read_csv(DATA_PATH)
    print("Initial columns:", df.columns.tolist())
    # Rename columns immediately and assign back to df
    df = df.rename(columns={'incidence': 'y', 'treated': 'Treat'})
    print("Renamed columns:", df.columns.tolist())
    # Verify required columns after renaming
    if not all(col in df.columns for col in ['region', 'week', 'y', 'Treat']):
        missing = [col for col in ['region', 'week', 'y', 'Treat'] if col not in df.columns]
        raise ValueError(f"Missing columns after renaming: {missing}")
    # Check for missing values
    if df[['region', 'week', 'y', 'Treat']].isna().any().any():
        print("Warning: Missing values detected. Imputing 'y' with mean.")
        df['y'] = df['y'].fillna(df['y'].mean())
    df = df.sort_values('week')
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    sys.exit(1)
except ValueError as e:
    print(f"Data validation error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error loading data: {e}")
    sys.exit(1)

df['Post'] = (df['week'] >= pd.Timestamp('2021-02-01')).astype(int)
print("Data preparation completed.")

# Create output directories with absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 2. Difference-in-Differences
try:
    df['Treat:Post'] = df['Treat'] * df['Post']
    df['time_trend'] = (df['week'] - df['week'].min()).dt.days / 7
    X = sm.add_constant(df[['Treat', 'Post', 'Treat:Post', 'time_trend']])
    model = sm.OLS(df['y'], X).fit(cov_type='HC1')
    print(model.summary())
    did_att = model.params['Treat:Post']
    print("DiD ATT:", did_att)
except Exception as e:
    print(f"Error in DiD: {e}")
    sys.exit(1)

# 3. Propensity Score Matching (PSM) with SMD
try:
    X_psm = df.drop(columns=['week', 'y', 'Post'])
    y_psm = df['Treat']
    X_psm = pd.get_dummies(X_psm, columns=['region'], drop_first=True)
    logit = LogisticRegression(max_iter=1000)
    logit.fit(X_psm, y_psm)
    propensity_scores = pd.Series(logit.predict_proba(X_psm)[:, 1], index=df.index)
    treated_mask = df['Treat'] == 1
    control_mask = df['Treat'] == 0
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(propensity_scores[control_mask].values.reshape(-1, 1))
    distances, indices = nn.kneighbors(propensity_scores[treated_mask].values.reshape(-1, 1))
    matched_control_indices = np.where(control_mask)[0][indices.flatten()]
    matched_indices = np.concatenate([np.where(treated_mask)[0], matched_control_indices])
    matched_df = df.iloc[matched_indices].copy()
    
    def calculate_smd(df1, df2, cols):
        return np.abs((df1[cols].mean() - df2[cols].mean()) / np.sqrt((df1[cols].var() + df2[cols].var()) / 2))
    
    smd_before = calculate_smd(df[treated_mask], df[control_mask], X_psm.columns)
    smd_after = calculate_smd(matched_df[matched_df['Treat'] == 1], matched_df[matched_df['Treat'] == 0], X_psm.columns)
    print("SMD before matching:", {col: val for col, val in zip(X_psm.columns, smd_before)})
    print("SMD after matching:", {col: val for col, val in zip(X_psm.columns, smd_after)})
    psm_att = matched_df[matched_df['Treat'] == 1]['y'].mean() - matched_df[matched_df['Treat'] == 0]['y'].mean()
    print(f"PSM ATT: {psm_att:.3f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [smd_before.mean(), smd_after.mean()], marker='o', label='Mean SMD')
    plt.axhline(0.1, color='r', linestyle='--', label='Threshold')
    plt.xticks([0, 1], ['Before', 'After'])
    plt.ylabel('Standardized Mean Difference')
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_smd_balance.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("SMD balance plot saved.")
except ValueError as e:
    print(f"PSM value error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"PSM error: {e}")
    sys.exit(1)

# 4. Event-Study Plot
try:
    df['time_to_treat'] = (df['week'] - pd.Timestamp('2021-02-01')).dt.days / 7
    mean_by_time = df.groupby('time_to_treat')['y'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(mean_by_time['time_to_treat'], mean_by_time['y'], marker='o', label='Mean Incidence')
    plt.axvline(0, color='k', linestyle='--', label='Intervention')
    plt.title('Event-Study: Parallel Trends')
    plt.xlabel('Weeks Relative to Treatment')
    plt.ylabel('Mean Incidence')
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_event_study.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("Event-study plot saved.")
except KeyError as e:
    print(f"Event-study key error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Event-study error: {e}")
    sys.exit(1)

# 5. Bayesian Structural Time Series (BSTS) with CausalImpact
try:
    pre_period = [df['week'].min(), pd.Timestamp('2021-02-01')]
    post_period = [pd.Timestamp('2021-02-01'), df['week'].max()]
    control_means = df[df['Treat'] == 0].groupby('week')['y'].mean().reindex(df['week']).fillna(method='ffill')
    data_ci = pd.DataFrame({
        'y': df['y'].values,
        'control': control_means.values
    })
    impact = CausalImpact(data_ci, pre_period, post_period)
    print(impact.summary())
    impact.plot()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_causalimpact.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("BSTS plot saved.")
except ValueError as e:
    print(f"BSTS value error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"BSTS error: {e}")
    sys.exit(1)

# 6. Outcome Trends
try:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='week', y='y', hue='Treat', ci=None)
    plt.axvline(pd.Timestamp('2021-02-01'), color='r', linestyle='--', label='Intervention')
    plt.ylabel('Incidence Rate')
    plt.legend(title='Treatment')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig0_outcome_trends.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("Outcome trends plot saved.")
except Exception as e:
    print(f"Outcome trends error: {e}")
    sys.exit(1)

# 7. Placeholder Functions for Enhancements
def bootstrap_did(data):
    n_boot = 100
    boots = []
    for _ in range(n_boot):
        boot_data = data.sample(frac=1, replace=True)
        model = smf.ols('incidence ~ treated * post', data=boot_data).fit()
        boots.append(model.params['treated:post'])
    return np.mean(boots), np.std(boots)

def run_ml_causal(data):
    from sklearn.linear_model import LinearRegression
    X = pd.get_dummies(data.drop(columns=['incidence']), drop_first=True)
    y = data['incidence']
    model = LinearRegression().fit(X, y)
    return model.coef_[0]  # Simplified ATT

# 8. Run Enhancements
data = df.copy()
data = data.rename(columns={'y': 'incidence', 'Treat': 'treated', 'Post': 'post'})
boot_mean, boot_std = bootstrap_did(data)
ml_att = run_ml_causal(data)
print(f"Bootstrap ATT: {boot_mean:.3f} (SD: {boot_std:.3f})")
print(f"ML Causal ATT: {ml_att:.3f}")

# 9. HTML Report (Removed to avoid syntax warnings for now)
print("HTML report generation skipped to avoid syntax warnings. Update manually if needed.")
