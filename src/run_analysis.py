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

warnings.filterwarnings("ignore")

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
    CFG = {}

DATA_PATH = ROOT / CFG.get("data_path", "data/ontario_cases.csv")
POLICY_DATE = CFG.get("policy_date", "2021-02-01")
BSTS_ENABLED = bool((CFG.get("bsts") or {}).get("enabled", True))

# =============================================================================
# 3. DATA LOADING
# =============================================================================
print(f"--- Starting Analysis [Policy Date: {POLICY_DATE}] ---")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

if "cases" in df.columns and "incidence" not in df.columns:
    print(">> Calculating incidence per 100k...")
    pop_map = {
        "Toronto": 2783000, "Peel": 1489000, "York": 1237000, "Durham": 697000,
        "Ottawa": 1010000, "Halton": 580000, "Hamilton": 569000, "Waterloo": 587000,
    }
    df["population"] = df["region"].map(pop_map).fillna(500000)
    df["incidence"] = df["cases"] * 100000 / df["population"]

df["week"] = pd.to_datetime(df["week"], errors="coerce")
df = df.dropna(subset=["week"]).sort_values("week")
df["post"] = (df["week"] >= pd.Timestamp(POLICY_DATE)).astype(int)

print(f">> Data Loaded: {len(df)} rows.")

# =============================================================================
# 4. DiD
# =============================================================================
print("\n--- Running Method 1: DiD ---")
df["treat_post"] = df["treated"] * df["post"]
model = smf.ols("incidence ~ C(region) + C(week) + treat_post", data=df)
did_res = model.fit(cov_type="cluster", cov_kwds={"groups": df["region"]})
did_att = did_res.params.get("treat_post", np.nan)
did_se = did_res.bse.get("treat_post", np.nan)
print(f"   DiD ATT: {did_att:.2f} (SE: {did_se:.2f})")

# =============================================================================
# 5. PSM
# =============================================================================
print("\n--- Running Method 2: PSM ---")
psm_att = None
psm_meta = {"status": "skipped", "matches": 0}

try:
    pre_data = df[df["post"] == 0].copy()
    covars = ["incidence"]
    X = pre_data[covars].fillna(pre_data[covars].median())
    lr = LogisticRegression(max_iter=1000, random_state=42).fit(X, pre_data["treated"])
    pre_data["ps"] = lr.predict_proba(X)[:, 1]

    treat_grp = pre_data[pre_data["treated"] == 1]
    ctrl_grp = pre_data[pre_data["treated"] == 0]
    
    nn = NearestNeighbors(n_neighbors=1).fit(ctrl_grp[["ps"]])
    dists, idxs = nn.kneighbors(treat_grp[["ps"]])
    
    matched_pairs = []
    caliper = 0.2 * np.log(pre_data["ps"]/(1-pre_data["ps"])).std()
    
    for i, d in enumerate(dists.flatten()):
        if d <= caliper:
            matched_pairs.append((treat_grp.iloc[i], ctrl_grp.iloc[idxs.flatten()[i]]))
            
    psm_meta["matches"] = len(matched_pairs)
    
    if matched_pairs:
        post = df[df["post"] == 1].groupby("region")["incidence"].mean()
        diffs = []
        for t, c in matched_pairs:
            if t["region"] in post and c["region"] in post:
                diffs.append(post[t["region"]] - post[c["region"]])
        if diffs:
            psm_att = float(np.mean(diffs))
            print(f"   PSM ATT: {psm_att:.2f} (Matches: {len(matched_pairs)})")

except Exception as e:
    print(f"   PSM Failed: {e}")

# =============================================================================
# 6. METHOD 3: BSTS (R/CausalImpact) - EDGE GAP FIX
# =============================================================================
print("\n--- Running Method 3: BSTS (R/CausalImpact) ---")
bsts_att = None
bsts_meta = {"enabled": BSTS_ENABLED}

if BSTS_ENABLED:
    try:
        # 1. DATA CLEANING (CRITICAL STEP)
        # Resample to weekly -> Interpolate Gaps -> Backfill Start -> Forwardfill End
        ts = df.groupby("week")["incidence"].mean()
        ts = ts.resample("W-MON").mean()
        ts = ts.interpolate().bfill().ffill().reset_index()
        
        # 2. Calculate INDICES
        n_points = len(ts)
        policy_ts = pd.Timestamp(POLICY_DATE)
        
        # Find index
        post_start_idx = ts[ts["week"] >= policy_ts].index.min()
        
        if pd.isna(post_start_idx) or post_start_idx == 0:
            # Fallback logic if policy matches nothing
            post_start_idx = int(n_points * 0.3) 

        pre_end_r = int(post_start_idx) 
        post_start_r = int(post_start_idx) + 1
        
        print(f"   Debug: Total={n_points} weeks. Pre-Period ends at index {pre_end_r}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_csv = tmp_path / "data.csv"
            out_json = tmp_path / "out.json"
            ts.to_csv(data_csv, index=False)

            r_script = f"""
            suppressMessages(library(CausalImpact))
            suppressMessages(library(jsonlite))
            
            df <- read.csv("{data_csv.as_posix()}")
            values <- df$incidence
            
            # Define periods
            pre.period <- c(1, {pre_end_r})
            post.period <- c({post_start_r}, {n_points})
            
            # Run Model (Default settings, relying on clean data)
            impact <- CausalImpact(values, pre.period, post.period)
            
            res <- list(bsts_att = as.numeric(impact$summary$AbsEffect["Average"]))
            write(toJSON(res, auto_unbox=TRUE), "{out_json.as_posix()}")
            """
            
            r_file = tmp_path / "script.R"
            r_file.write_text(r_script, encoding="utf-8")
            
            proc = subprocess.run(["Rscript", str(r_file)], capture_output=True, text=True)
            
            if proc.returncode != 0:
                raise RuntimeError(f"R Error:\n{proc.stderr}")
                
            bsts_res = json.loads(out_json.read_text())
            raw_att = bsts_res.get("bsts_att")
            
            try:
                bsts_att = float(raw_att) if raw_att is not None else None
                print(f"   BSTS ATT: {bsts_att:.2f}" if bsts_att else f"   BSTS ATT: {raw_att}")
            except:
                print(f"   BSTS Result: {raw_att}")
            
    except Exception as e:
        bsts_meta["error"] = str(e)
        print(f"   BSTS Failed: {e}")

# =============================================================================
# 7. SAVE
# =============================================================================
def safe_serialize(obj):
    if isinstance(obj, (np.generic, np.number)):
        return obj.item() if not np.isnan(obj) else None
    return obj if isinstance(obj, float) and not np.isnan(obj) else obj

out = {
    "did": {"att": safe_serialize(did_att), "se": safe_serialize(did_se)},
    "psm": {"att": safe_serialize(psm_att)},
    "bsts": {"att": safe_serialize(bsts_att)},
    "metadata": {"policy": str(POLICY_DATE)}
}
with open(RESULTS_PATH, "w") as f: json.dump(out, f, indent=4, default=str)
print(f"\n>> Success! Saved: {RESULTS_PATH}")