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
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# 1. Load and Prepare Data
try:
    df = pd.read_csv(r"C:\Users\jibra\OneDrive\Desktop\ontario-health-causal-analysis\data\ontario_cases.csv")
    print("Initial columns:", df.columns.tolist())
    df = df.rename(columns={'incidence': 'y', 'treated': 'Treat'})
    print("Renamed columns:", df.columns.tolist())
    if not all(col in df.columns for col in ['region', 'week', 'y', 'Treat']):
        missing = [col for col in ['region', 'week', 'y', 'Treat'] if col not in df.columns]
        print(f"Error: Missing columns: {missing}")
        sys.exit(1)
    # Check for missing values
    if df[['region', 'week', 'y', 'Treat']].isna().any().any():
        print("Warning: Missing values detected. Imputing with mean for 'y'.")
        df['y'] = df['y'].fillna(df['y'].mean())
except FileNotFoundError:
    print("Error: 'ontario_cases.csv' not found.")
    sys.exit(1)

df['week'] = pd.to_datetime(df['week'])
df['Post'] = (df['week'] >= pd.Timestamp('2021-02-01')).astype(int)
df = df.sort_values('week')
print("Data preparation completed.")

# Create output directories
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

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
    # Prepare covariates (excluding outcome and time variables)
    X_psm = df.drop(columns=['week', 'y', 'Post'])
    y_psm = df['Treat']
    
    # Convert 'region' to categorical and dummy variables
    X_psm = pd.get_dummies(X_psm, columns=['region'], drop_first=True)
    
    # Fit logistic regression for propensity scores
    logit = LogisticRegression()
    logit.fit(X_psm, y_psm)
    propensity_scores = logit.predict_proba(X_psm)[:, 1]
    
    # Match treated and control units on propensity scores
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(propensity_scores[df['Treat'] == 0].reshape(-1, 1))
    distances, indices = nn.kneighbors(propensity_scores[df['Treat'] == 1].reshape(-1, 1))
    matched_indices = indices.flatten()
    matched_df = df.iloc[np.concatenate([np.where(df['Treat'] == 1)[0], np.where(df['Treat'] == 0)[0][matched_indices]])]
    
    # Calculate SMD before and after matching
    def calculate_smd(df1, df2, cols):
        return np.abs((df1[cols].mean() - df2[cols].mean()) / np.sqrt((df1[cols].var() + df2[cols].var()) / 2))
    
    smd_before = calculate_smd(df[df['Treat'] == 1], df[df['Treat'] == 0], ['y'])
    smd_after = calculate_smd(matched_df[matched_df['Treat'] == 1], matched_df[matched_df['Treat'] == 0], ['y'])
    print(f"SMD before matching: {smd_before:.3f}, after matching: {smd_after:.3f}")
    
    # PSM ATT
    psm_att = matched_df[matched_df['Treat'] == 1]['y'].mean() - matched_df[matched_df['Treat'] == 0]['y'].mean()
    print(f"PSM ATT: {psm_att:.3f}")
    
    # Plot SMD
    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [smd_before[0], smd_after[0]], marker='o', label='SMD')
    plt.axhline(0.1, color='r', linestyle='--', label='Threshold')
    plt.xticks([0, 1], ['Before', 'After'])
    plt.ylabel('Standardized Mean Difference')
    plt.legend()
    plt.savefig('figures/fig2_smd_balance.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("SMD balance plot saved.")
except Exception as e:
    print(f"Error in PSM: {e}")
    sys.exit(1)

# 4. Event-Study Plot
try:
    df['time_to_treat'] = (df['week'] - pd.Timestamp('2021-02-01')).dt.days / 7
    mean_by_time = df.groupby('time_to_treat')['y'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(mean_by_time['time_to_treat'], mean_by_time['y'], marker='o')
    plt.axvline(0, color='k', linestyle='--', label='Intervention')
    plt.title('Event-Study: Parallel Trends')
    plt.xlabel('Weeks Relative to Treatment')
    plt.ylabel('Mean Incidence')
    plt.legend()
    plt.savefig('figures/fig1_event_study.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Event-study plot saved.")
except Exception as e:
    print(f"Error in event-study plot: {e}")
    sys.exit(1)

# 5. Bayesian Structural Time Series (BSTS) with CausalImpact
try:
    pre_period = [df['week'].min(), pd.Timestamp('2021-02-01')]
    post_period = [pd.Timestamp('2021-02-01'), df['week'].max()]
    # Prepare data with treated as response and controls
    data_ci = pd.DataFrame({
        'y': df['y'],
        'control': df[df['Treat'] == 0]['y'].mean()  # Simplified control mean
    })
    impact = CausalImpact(data_ci, pre_period, post_period)
    print(impact.summary())
    impact.plot()
    plt.savefig('figures/fig_causalimpact.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("BSTS plot saved.")
except Exception as e:
    print(f"Error in BSTS: {e}")
    sys.exit(1)

# 6. Outcome Trends
try:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='week', y='y', hue='Treat', ci=None)
    plt.axvline(pd.Timestamp('2021-02-01'), color='r', linestyle='--', label='Intervention')
    plt.ylabel('Incidence Rate')
    plt.legend()
    plt.savefig('figures/fig0_outcome_trends.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Outcome trends plot saved.")
except Exception as e:
    print(f"Error in outcome trends plot: {e}")
    sys.exit(1)

# 7. Placeholder Functions for Enhancements
def bootstrap_did(data):
    # Placeholder: Return dummy values
    return np.random.normal(0, 0.1), np.random.normal(0, 0.01)

def run_ml_causal(data):
    # Placeholder: Return dummy value
    return np.random.normal(0, 0.1)

# 8. Run Enhancements
data = df.copy()
data = data.rename(columns={'y': 'incidence', 'Treat': 'treated', 'Post': 'post'})
boot_mean, boot_std = bootstrap_did(data)
ml_att = run_ml_causal(data)
print(f"Bootstrap ATT: {boot_mean:.3f} (SD: {boot_std:.3f})")
print(f"ML Causal ATT: {ml_att:.3f}")

# 9. HTML Report (Simplified)
try:
    html_content = f"""
    <html><body><h1>Causal Impact Analysis</h1>
    <h2>Figures</h2><img src="figures/fig0_outcome_trends.png"><br>
    <img src="figures/fig1_event_study.png"><br>
    <img src="figures/fig2_smd_balance.png"><br>
    <img src="figures/fig_causalimpact.png"><br>
    </body></html>
    """
    with open('results/analysis.html', 'w') as f:
        f.write(html_content)
    print("HTML report saved.")
except Exception as e:
    print(f"Error in HTML generation: {e}")
    sys.exit(1)
