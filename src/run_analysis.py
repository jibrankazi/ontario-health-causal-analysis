# -----------------------------------------------------------------------------
# ONTARIO HEALTH CAUSAL ANALYSIS (Reproducible Pipeline)
# Methods: DiD (Diff-in-Diff), PSM (Propensity Score Matching), BSTS (CausalImpact)
# Author: Kazi Jibran Rafat Samie
# Date: November 17, 2025
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import tempfile
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# Ignore irrelevant warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# 1. DETERMINISTIC SETUP
# =============================================================================
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seed()

# =============================================================================
# 2. CONFIGURATION & PATHS
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / "results.json"

cfg_path = ROOT / "config.yaml"
if cfg_path.exists():
    with open(cfg_path, "r", encoding="utf-8") as f:
        CFG = yaml.safe_load(f)
else:
    print("Warning: config.yaml not found. Using defaults.")
    CFG = {}

DATA_PATH = ROOT / CFG.get("data_path", "data/ontario_cases.csv")
POLICY_DATE = CFG.get("policy_date", "2021-02-01")
BSTS_ENABLED = bool((CFG.get("bsts") or {}).get("enabled", True))

# =============================================================================
# 3. DATA LOADING & PREPROCESSING
# =============================================================================
print(f"--- Starting Analysis [Policy Date: {POLICY_DATE}] ---")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"CRITICAL: Data file not found at {DATA_PATH}")

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    raise ValueError(f"Failed to read CSV. Error: {e}")

if "cases" in df.columns and "incidence" not in df.columns:
    print(">> Calculating incidence per 100k...")
    pop_map = {
        "Toronto": 2783000, "Peel": 1489000, "York": 1237000, "Durham": 697000,
        "Ottawa": 1010000, "Halton": 580000, "Hamilton": 569000, "Waterloo": 587000,
    }
    df["population"] = df["region"].map(pop_map).fillna(500000)
    df["incidence"] = df["cases"] * 100000 / df["population"]

required_cols = {"week", "region", "incidence", "treated"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Data missing required columns: {required_cols - set(df.columns)}")

df["week"] = pd.to_datetime(df["week"], errors="coerce")
df = df.dropna(subset=["week"]).sort_values("week")
df["post"] = (df["week"] >= pd.Timestamp(POLICY_DATE)).astype(int)

print(f">> Data Loaded: {len(df)} observations across {df['region'].nunique()} regions.")

# =============================================================================
# 4. METHOD 1: DIFFERENCE-IN-DIFFERENCES (DiD)
# =============================================================================
print("\n--- Running Method 1: DiD ---")
df["treat_post"] = df["treated"] * df["post"]
model = smf.ols("incidence ~ C(region) + C(week) + treat_post", data=df)
did_res = model.fit(cov_type="cluster", cov_kwds={"groups": df["region"]})
did_att = did_res.params.get("treat_post", np.nan)
did_se = did_res.bse.get("treat_post", np.nan)
print(f"   DiD ATT: {did_att:.2f} (SE: {did_se:.2f})")

# =============================================================================
# 5. METHOD 2: PROPENSITY SCORE MATCHING (PSM)
# =============================================================================
print("\n--- Running Method 2: PSM ---")
psm_att = None
psm_meta = {"status": "skipped", "matches": 0}

try:
    pre_data = df[df["post"] == 0].copy()
    exclude = {"week", "region", "incidence", "treated", "post", "treat_post", "population"}
    covars = [c for c in pre_data.columns if c not in exclude and pd.api.types.is_numeric_dtype(pre_data[c])]
    if not covars: covars = ["incidence"]

    X = pre_data[covars].fillna(pre_data[covars].median())
    y = pre_data["treated"]
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X, y)
    pre_data["ps"] = lr.predict_proba(X)[:, 1]

    ps_treat = pre_data[pre_data["treated"] == 1]["ps"]
    ps_ctrl = pre_data[pre_data["treated"] == 0]["ps"]
    min_sup = max(ps_treat.min(), ps_ctrl.min())
    max_sup = min(ps_treat.max(), ps_ctrl.max())
    supported = pre_data[pre_data["ps"].between(min_sup, max_sup)].copy()

    logit_ps = np.log(supported["ps"] / (1 - supported["ps"] + 1e-9))
    caliper = 0.2 * logit_ps.std()
    
    treat_grp = supported[supported["treated"] == 1]
    ctrl_grp = supported[supported["treated"] == 0]
    nn = NearestNeighbors(n_neighbors=1).fit(ctrl_grp[["ps"]])
    distances, indices = nn.kneighbors(treat_grp[["ps"]])

    matched_pairs = []
    dist_flat = distances.flatten()
    idx_flat = indices.flatten()
    for i, dist in enumerate(dist_flat):
        if dist <= caliper:
            matched_pairs.append((treat_grp.iloc[i], ctrl_grp.iloc[idx_flat[i]]))

    psm_meta["matches"] = len(matched_pairs)
    if not matched_pairs:
        print("   Warning: No matches found within caliper.")
    else:
        post_data = df[df["post"] == 1]
        post_means = post_data.groupby("region")["incidence"].mean()
        att_diffs = []
        for t_row, c_row in matched_pairs:
            t_reg = t_row["region"]
            c_reg = c_row["region"]
            if t_reg in post_means.index and c_reg in post_means.index:
                att_diffs.append(post_means[t_reg] - post_means[c_reg])
        
        if att_diffs:
            psm_att = float(np.mean(att_diffs))
            print(f"   PSM ATT: {psm_att:.2f} (Matches: {len(matched_pairs)})")

except Exception as e:
    print(f"   PSM Failed: {e}")

# =============================================================================
# 6. METHOD 3: BSTS (R/CausalImpact) - FINAL
# =============================================================================
print("\n--- Running Method 3: BSTS (R/CausalImpact) ---")
bsts_att = None
bsts_meta = {"enabled": BSTS_ENABLED}

if BSTS_ENABLED:
    try:
        ts_data = df.groupby("week", as_index=False)["incidence"].mean()
        policy_dt = pd.Timestamp(POLICY_DATE)
        pre_end_dt = policy_dt - pd.Timedelta(days=1)
        post_end_dt = ts_data["week"].max()

        fmt = "%Y-%m-%d"
        r_params = {
            "pre_start": ts_data["week"].min().strftime(fmt),
            "pre_end": pre_end_dt.strftime(fmt),
            "post_start": policy_dt.strftime(fmt),
            "post_end": post_end_dt.strftime(fmt)
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_csv = tmp_path / "data.csv"
            out_json = tmp_path / "out.json"
            ts_data.to_csv(data_csv, index=False)

            r_script = f"""
            suppressMessages(library(CausalImpact))
            suppressMessages(library(jsonlite))
            suppressMessages(library(zoo))
            
            df <- read.csv("{data_csv.as_posix()}")
            df$week <- as.Date(df$week)
            ts_data <- zoo(df$incidence, df$week)
            
            pre.period <- as.Date(c("{r_params['pre_start']}", "{r_params['pre_end']}"))
            post.period <- as.Date(c("{r_params['post_start']}", "{r_params['post_end']}"))
            
            impact <- CausalImpact(ts_data, pre.period, post.period)
            
            res <- list(bsts_att = as.numeric(impact$summary$AbsEffect["Average"]))
            write(toJSON(res, auto_unbox=TRUE), "{out_json.as_posix()}")
            """
            
            r_file = tmp_path / "script.R"
            r_file.write_text(r_script, encoding="utf-8")
            
            proc = subprocess.run(["Rscript", str(r_file)], capture_output=True, text=True)
            
            if proc.returncode != 0:
                raise RuntimeError(f"R Error output:\n{proc.stderr}")
                
            bsts_res = json.loads(out_json.read_text())
            raw_att = bsts_res.get("bsts_att")
            
            # Robustly handle the result (float or string)
            try:
                bsts_att = float(raw_att) if raw_att is not None else None
                if bsts_att is not None:
                    print(f"   BSTS ATT: {bsts_att:.2f}")
                else:
                    print("   BSTS ATT: None (Model returned null)")
            except ValueError:
                bsts_att = None
                print(f"   BSTS ATT: {raw_att} (Raw Output)")
            
    except Exception as e:
        bsts_meta["error"] = str(e)
        print(f"   BSTS Failed: {e}")
else:
    print("   BSTS Disabled in config.")

# =============================================================================
# 7. SAVING RESULTS
# =============================================================================
def safe_serialize(obj):
    if isinstance(obj, (np.generic, np.number)):
        return obj.item() if not np.isnan(obj) else None
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj

final_output = {
    "did": {"att": safe_serialize(did_att), "se": safe_serialize(did_se)},
    "psm": {"att": safe_serialize(psm_att), "meta": psm_meta},
    "bsts": {"att": safe_serialize(bsts_att), "meta": bsts_meta},
    "metadata": {"timestamp": datetime.now().isoformat(), "policy_date": str(POLICY_DATE)}
}

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=4, default=str)

print(f"\n>> Success! Results saved to: {RESULTS_PATH}")