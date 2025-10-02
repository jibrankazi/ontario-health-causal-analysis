import json, warnings, yaml, pathlib, math
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------- load config ----------
cfg = yaml.safe_load(open("config.yaml"))
DATA = pathlib.Path(cfg["data_path"])
date_col   = cfg["date_col"]
unit_col   = cfg["unit_col"]
y_col      = cfg["outcome_col"]
treat_col  = cfg["treat_col"]
covs       = cfg.get("covariates", []) or []
freq       = cfg.get("freq", "W-MON")
t_intended = pd.to_datetime(cfg["intervention_date"])

# ---------- load data ----------
df = pd.read_csv(DATA)
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values([unit_col, date_col]).copy()

# ---------- snap intervention to first available date >= intended ----------
t0 = df.loc[df[date_col] >= t_intended, date_col].min()
if pd.isna(t0):
    raise ValueError(f"No dates on/after {t_intended.date()} in {date_col}.")

# ---------- regularize to weekly (no-op if already weekly) ----------
df_reg = (
    df.set_index(date_col)
      .groupby(unit_col)
      .apply(lambda g: g.resample(freq).mean())
      .drop(columns=[unit_col], errors="ignore")
      .reset_index()
      .sort_values([unit_col, date_col])
)
df = df_reg
df["post"] = (df[date_col] >= t0).astype(int)
df["tp"] = df[treat_col] * df["post"]

Path("figures").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

# ---------- DiD (TWFE) with clustered SEs by unit ----------
d = pd.get_dummies(df, columns=[unit_col], drop_first=True, prefix="u").copy()
d["_time"] = d[date_col].dt.to_period("W").astype(str)
d = pd.get_dummies(d, columns=["_time"], drop_first=True, prefix="t")
X_cols = [treat_col, "post", "tp"] + [c for c in d.columns if c.startswith("u_") or c.startswith("t_")]
X = sm.add_constant(d[X_cols])
y = d[y_col].astype(float)
groups = df[cfg["unit_col"]].to_numpy()
model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})

did_att = float(model.params.get("tp", np.nan))
did_se  = float(model.bse.get("tp", np.nan)) if "tp" in model.bse else None
did_p   = float(model.pvalues.get("tp", np.nan)) if "tp" in model.pvalues else None

# ---------- Event-study (leads/lags) + pre-trend joint test ----------
df_es = df.copy()
k0 = df_es[date_col][df_es[date_col] >= t0].min()
df_es["_k"] = ((df_es[date_col] - k0).dt.days // 7).astype(int)
leads = list(range(-8, 0))
lags  = list(range(0, 9))
for k in leads + lags:
    df_es[f"I{k}"] = (df_es["_k"] == k).astype(int)
omit = -1
terms = [f"I{k}:{treat_col}" for k in leads + lags if k != omit]
formula = y_col + " ~ " + " + ".join(terms + [treat_col] + [f"C({cfg['unit_col']})", "C(_k)"])
es = ols(formula, data=df_es).fit(cov_type="cluster", cov_kwds={"groups": df_es[cfg["unit_col"]]})
pre_terms = [f"I{k}:{treat_col}" for k in leads if k != omit]
from statsmodels.stats.contrast import ContrastResults
pretest = es.f_test(" + ".join(pre_terms) + " = 0")
pretrend_p = float(pretest.pvalue)

# Coef path for plotting
coef_k, se_k = [], []
for k in leads + lags:
    if k == omit: 
        continue
    name = f"I{k}:{treat_col}"
    if name in es.params.index:
        coef_k.append((k, es.params[name]))
        se_k.append((k, es.bse[name]))
ks = [k for k,_ in coef_k]
vals = [v for _,v in coef_k]
ses  = [s for _,s in se_k]
plt.figure()
plt.errorbar(ks, vals, yerr=1.96*np.array(ses), fmt='-o')
plt.axvline(0, linestyle='--', alpha=0.6)
plt.axhline(0, color='k', linewidth=0.6)
plt.title("Event-study: treated x event-time (95% CI)")
plt.tight_layout()
plt.savefig("figures/fig1_event_study.png", dpi=150)

# ---------- PSM (optional) with balance + bootstrap CI ----------
def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    denom = np.sqrt((va + vb) / 2.0) if (va+vb) > 0 else np.nan
    return (np.nanmean(a) - np.nanmean(b)) / denom if denom and not np.isnan(denom) else np.nan

psm_att = None
psm_se  = None
if covs:
    pre = df[df["post"] == 0].copy()
    # pre-match SMDs
    prematch = {c: smd(pre.loc[pre[treat_col]==1, c], pre.loc[pre[treat_col]==0, c]) for c in covs}
    pd.Series(prematch, name="SMD_prematch").to_csv("results/psm_balance_prematch.csv")

    if not pre.empty and pre[treat_col].nunique() == 2:
        Z = pre[covs].astype(float).values
        scaler = StandardScaler().fit(Z)
        Zs = scaler.transform(Z)
        tmask = pre[treat_col].values == 1
        cmask = pre[treat_col].values == 0
        if tmask.any() and cmask.any():
            nn = NearestNeighbors(n_neighbors=1).fit(Zs[cmask])
            dists, idx = nn.kneighbors(Zs[tmask], return_distance=True)
            matched_ctrl = pre[cmask].iloc[idx.flatten()].copy()
            matched_treat = pre[tmask].copy()
            # post-match SMDs
            postmatch = {c: smd(matched_treat[c], matched_ctrl[c]) for c in covs}
            pd.Series(postmatch, name="SMD_postmatch").to_csv("results/psm_balance_postmatch.csv")

            # Simple post-period ATT and bootstrap CI
            post = df[df["post"] == 1]
            y_t = post[post[treat_col]==1][y_col].values
            y_c = post[post[treat_col]==0][y_col].values
            if len(y_t) and len(y_c):
                diff = y_t.mean() - y_c.mean()
                B=500
                atts=[]
                rng = np.random.default_rng(42)
                for _ in range(B):
                    bt = rng.choice(y_t, size=len(y_t), replace=True)
                    bc = rng.choice(y_c, size=len(y_c), replace=True)
                    atts.append(bt.mean() - bc.mean())
                psm_att = float(np.mean(atts))
                psm_se  = float(np.std(atts, ddof=1))

# ---------- Event trend plot by group (means) ----------
g = df.groupby([date_col, treat_col])[y_col].mean().reset_index()
plt.figure()
for grp, lab in [(1, "Treated"), (0, "Control")]:
    sub = g[g[treat_col]==grp]
    plt.plot(sub[date_col], sub[y_col], label=lab)
plt.axvline(t0, linestyle="--", alpha=0.6)
plt.title("Event trend (weekly means)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/fig1_event_trends.png", dpi=150)

# ---------- Save results ----------
out = {
    "did_att": did_att if math.isfinite(did_att) else None,
    "did_se": did_se if (did_se is not None and math.isfinite(did_se)) else None,
    "did_p":  did_p  if (did_p  is not None and math.isfinite(did_p))  else None,
    "pretrend_p": pretrend_p if math.isfinite(pretrend_p) else None,
    "psm_att": psm_att if (psm_att is not None and math.isfinite(psm_att)) else None,
    "psm_se":  psm_se  if (psm_se  is not None and math.isfinite(psm_se))  else None,
    "note": "Clustered DiD, event-study pretrend test, PSM balance+bootstrap. Python-only."
}
with open("results/results.json","w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))


# ---------- NUMERIC SAFETY FOR OLS / CLUSTER VARS ----------
_num_cols = []
for _c in [y_col, "post", treat_col, "tp"]:
    if _c in df.columns:
        _num_cols.append(_c)
df[_num_cols] = df[_num_cols].apply(pd.to_numeric, errors="coerce")

# drop rows with NAs in key fields
df = df.dropna(subset=_num_cols).copy()

# cluster groups: use integer codes (not object dtype)
if unit_col in df.columns:
    groups = df[unit_col].astype("category").cat.codes
else:
    # fallback single cluster
    groups = pd.Series(0, index=df.index)

# Design matrix (TWFE-lite main terms)
X = df[["post", treat_col, "tp"]].astype(float)
X = sm.add_constant(X, has_constant="add")
y = df[y_col].astype(float)
# ---------- END NUMERIC SAFETY ----------
