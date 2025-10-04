# src/impact_py.py
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt


@dataclass
class ImpactResult:
    att: Optional[float]
    ci: Tuple[Optional[float], Optional[float]]
    p: Optional[float]
    relative_effect: Optional[float]
    timeline: List[Dict[str, Any]]  # rows with date, actual, predicted, lower, upper
    notes: Optional[str] = None
    plot_path: Optional[str] = None
    summary_path: Optional[str] = None


def _weekly_grid(min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    weeks = pd.date_range(min_date.normalize(), max_date.normalize(), freq="W-MON")
    return pd.DataFrame({"week": weeks})


def build_ci_matrix(panel: pd.DataFrame,
                    policy_date: pd.Timestamp,
                    min_corr: float = 0.25,
                    top_min: int = 2) -> pd.DataFrame:
    """
    Build a weekly DataFrame with:
      - y: mean incidence across treated regions
      - x_*: control region weekly incidences for selected controls
    Fills holes forward/backward to avoid missing values in models.
    """
    df = panel.copy()
    df["week"] = pd.to_datetime(df["week"])

    treated_agg = (df[df["treated"] == 1]
                   .groupby("week", as_index=False)["incidence"].mean()
                   .rename(columns={"incidence": "y"}))

    ctrl = df[df["treated"] == 0].copy()
    ctrl["var"] = "x_" + ctrl["region"].astype(str)
    ctrl_wide = ctrl.pivot_table(index="week", columns="var",
                                 values="incidence", aggfunc="mean").reset_index()

    grid = _weekly_grid(df["week"].min(), df["week"].max())
    m = (grid.merge(treated_agg, on="week", how="left")
              .merge(ctrl_wide, on="week", how="left")
              .sort_values("week")
              .reset_index(drop=True))

    # Fill missing values both directions (safe for weekly data we’re only averaging)
    m = m.ffill().bfill()

    # Select informative controls by pre-period correlation with y
    pre_mask = m["week"] < policy_date
    y_pre = m.loc[pre_mask, "y"].to_numpy(float)
    Xcols = [c for c in m.columns if c.startswith("x_")]

    if len(Xcols) == 0:
        return m[["week", "y"]]

    cors = []
    for c in Xcols:
        v = m.loc[pre_mask, c].to_numpy(float)
        if len(y_pre) > 1 and not np.all(np.isnan(v)):
            with np.errstate(all="ignore"):
                r = np.corrcoef(y_pre, v)[0, 1]
        else:
            r = np.nan
        cors.append((c, r))
    cors = [(c, r) for c, r in cors if np.isfinite(r)]
    cors.sort(key=lambda t: abs(t[1]), reverse=True)

    keep = [c for c, r in cors if abs(r) >= min_corr]
    if len(keep) < top_min and len(cors) >= top_min:
        keep = [c for c, _ in cors[:top_min]]
    if len(keep) == 0 and len(cors) > 0:
        keep = [cors[0][0]]

    cols = ["week", "y"] + keep
    return m[cols]


def _try_python_causalimpact(df: pd.DataFrame,
                             pre_period: Tuple[pd.Timestamp, pd.Timestamp],
                             post_period: Tuple[pd.Timestamp, pd.Timestamp]) -> Optional[ImpactResult]:
    """
    Try causalimpact (PyPI) or tfcausalimpact (PyPI).
    Both accept a DataFrame with first column as response and others as covariates.
    """
    data = df.drop(columns=["week"]).copy()
    data.index = df["week"]

    def _extract(ci_obj) -> ImpactResult:
        # Most Python ports provide an 'inferences' DataFrame
        inf = getattr(ci_obj, "inferences", None)
        if inf is None:
            # tfcausalimpact returns a DataFrame from ci.run() or stores at .impact
            inf = getattr(ci_obj, "impact", None)
        if inf is None:
            raise RuntimeError("Could not locate inferences DataFrame in causalimpact object.")

        # Try common column names used by ports
        # (point_pred may be 'point.pred', same for lower/upper)
        def pick(cands):
            for c in cands:
                if c in inf.columns:
                    return c
            return None

        col_eff = pick(["point_effect", "point.effect", "effect"])
        col_lo  = pick(["point_effect_lower", "point.effect.lower", "effect.lower"])
        col_hi  = pick(["point_effect_upper", "point.effect.upper", "effect.upper"])
        col_y   = pick(["response", "y", "actual"])
        col_pred= pick(["point_pred", "point.pred", "pred"])

        # mask for post period on the index
        post_mask = (inf.index >= post_period[0]) & (inf.index <= post_period[1])

        if col_eff is None and col_y and col_pred:
            inf["__eff__"] = inf[col_y] - inf[col_pred]
            col_eff = "__eff__"

        post = inf.loc[post_mask]
        if post.empty:
            raise RuntimeError("Post-period slice is empty in inferences DataFrame.")

        att = float(np.nanmean(post[col_eff])) if col_eff else None
        lo  = float(np.nanmean(post[col_lo])) if col_lo else None
        hi  = float(np.nanmean(post[col_hi])) if col_hi else None

        # relative effect = mean(effect)/mean(pred)
        reff = None
        if col_pred and col_eff:
            mu_pred = float(np.nanmean(post[col_pred]))
            if math.isfinite(mu_pred) and mu_pred != 0 and math.isfinite(att):
                reff = att / mu_pred

        # p-value – some ports expose .p_value or .p_value_ (optional)
        p = None
        for attr in ("p_value", "p_value_", "pvalue"):
            if hasattr(ci_obj, attr):
                val = getattr(ci_obj, attr)
                try:
                    p = float(val)
                except Exception:
                    p = None
                break

        # Timeline for plotting / JSON
        timeline = []
        # Try to reconstruct predicted CI as well
        col_pred_lo = pick(["point_pred_lower", "point.pred.lower", "pred.lower"])
        col_pred_hi = pick(["point_pred_upper", "point.pred.upper", "pred.upper"])

        for idx, row in inf.iterrows():
            d = {
                "date": pd.Timestamp(idx).date().isoformat(),
                "actual": float(row[col_y]) if col_y in inf.columns else None,
                "predicted": float(row[col_pred]) if col_pred in inf.columns else None,
                "lower": float(row.get(col_pred_lo)) if col_pred_lo in inf.columns else None,
                "upper": float(row.get(col_pred_hi)) if col_pred_hi in inf.columns else None,
            }
            timeline.append(d)

        return ImpactResult(att=att, ci=(lo, hi), p=p,
                            relative_effect=reff, timeline=timeline,
                            notes="python-causalimpact")

    # Try causalimpact (Willian Fuks)
    try:
        from causalimpact import CausalImpact as CI  # type: ignore
        ci = CI(data, pre_period=pre_period, post_period=post_period)
        return _extract(ci)
    except Exception:
        pass

    # Try tfcausalimpact (TensorFlow Probability backend)
    try:
        from tfcausalimpact import CausalImpact as TFCI  # type: ignore
        ci = TFCI(data, pre_period=pre_period, post_period=post_period)
        # Some versions require ci.run()
        if hasattr(ci, "run"):
            ci = ci.run()
        return _extract(ci)
    except Exception:
        return None


def _sarimax_counterfactual(df: pd.DataFrame,
                            pre_period: Tuple[pd.Timestamp, pd.Timestamp],
                            post_period: Tuple[pd.Timestamp, pd.Timestamp]) -> ImpactResult:
    """
    Frequentist fallback: SARIMAX with exogenous controls.
    Fit on PRE, forecast into POST, compute mean effect and CI.
    """
    # Set index
    data = df.set_index("week").copy()
    y = data["y"].astype(float)
    X = data.drop(columns=["y"])

    pre_mask = (data.index >= pre_period[0]) & (data.index <= pre_period[1])
    post_mask= (data.index >= post_period[0]) & (data.index <= post_period[1])
    y_pre, X_pre = y.loc[pre_mask], X.loc[pre_mask] if not X.empty else None
    y_post, X_post = y.loc[post_mask], X.loc[post_mask] if not X.empty else None

    # Simple ARIMA(1,0,0) + optional weekly seasonal (52)
    mod = SARIMAX(y_pre, exog=X_pre, order=(1, 0, 0),
                  seasonal_order=(1, 0, 0, 52) if len(y_pre) > 60 else (0, 0, 0, 0),
                  enforce_stationarity=False, enforce_invertibility=False)
    fit = mod.fit(disp=False)

    steps = len(y_post)
    fc = fit.get_forecast(steps=steps, exog=X_post)
    pred = fc.predicted_mean
    ci   = fc.conf_int(alpha=0.05)  # columns: lower y, upper y
    lower = ci.iloc[:, 0]
    upper = ci.iloc[:, 1]

    effect = y_post - pred
    # Convert forecast CI into effect CI (actual - pred bounds)
    eff_lo = y_post - upper
    eff_hi = y_post - lower

    att = float(effect.mean())
    lo  = float(eff_lo.mean())
    hi  = float(eff_hi.mean())

    # relative effect vs predicted mean
    reff = None
    mu_pred = float(np.nanmean(pred))
    if math.isfinite(mu_pred) and mu_pred != 0:
        reff = att / mu_pred

    timeline = []
    for idx in y_post.index:
        timeline.append({
            "date": pd.Timestamp(idx).date().isoformat(),
            "actual": float(y_post.loc[idx]),
            "predicted": float(pred.loc[idx]),
            "lower": float(lower.loc[idx]),
            "upper": float(upper.loc[idx]),
        })

    return ImpactResult(att=att, ci=(lo, hi), p=None,
                        relative_effect=reff, timeline=timeline,
                        notes="sarimax-fallback")


def estimate_causal_impact_python(panel: pd.DataFrame,
                                  policy_date: pd.Timestamp,
                                  fig_path: str = "figures/fig_causalimpact.png",
                                  txt_path: str = "results/causalimpact_summary.txt"
                                  ) -> ImpactResult:
    """
    High-level entry point used by run_analysis.py.
    Tries a Python CausalImpact implementation; falls back to SARIMAX.
    """
    Path("figures").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    df = build_ci_matrix(panel, policy_date)
    pre_period  = (df["week"].min(), policy_date - pd.Timedelta(days=7))
    post_period = (policy_date, df["week"].max())

    # Try Python CausalImpact variants
    ci_res = _try_python_causalimpact(df, pre_period, post_period)

    # Fallback to SARIMAX(+exog)
    if ci_res is None:
        ci_res = _sarimax_counterfactual(df, pre_period, post_period)

    # Save summary (plain text)
    att, (lo, hi), p, reff = ci_res.att, ci_res.ci, ci_res.p, ci_res.relative_effect
    lines = [
        "Posterior/Counterfactual inference (Python)",
        f"ATT (mean effect): {att if att is not None else 'NA'}",
        f"95% CI: [{lo if lo is not None else 'NA'}, {hi if hi is not None else 'NA'}]",
        f"Relative effect: {reff if reff is not None else 'NA'}",
        f"p-value: {p if p is not None else 'NA'}",
        f"Notes: {ci_res.notes or ''}",
    ]
    Path(txt_path).write_text("\n".join(lines), encoding="utf-8")
    ci_res.summary_path = txt_path

    # Save plot
    # Create a simple two-panel plot: actual vs pred, and point effect.
    t = pd.to_datetime([r["date"] for r in ci_res.timeline])
    actual = pd.Series([r["actual"] for r in ci_res.timeline], index=t, dtype="float64")
    predicted = pd.Series([r["predicted"] for r in ci_res.timeline], index=t, dtype="float64")
    lower = pd.Series([r["lower"] for r in ci_res.timeline], index=t, dtype="float64")
    upper = pd.Series([r["upper"] for r in ci_res.timeline], index=t, dtype="float64")

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.plot(actual.index, actual.values, label="Actual")
    if predicted.notna().any():
        ax.plot(predicted.index, predicted.values, label="Predicted")
    if lower.notna().any() and upper.notna().any():
        ax.fill_between(lower.index, lower.values, upper.values, alpha=0.2, label="Pred. 95% CI")
    ax.axvline(policy_date, linestyle="--", alpha=0.8)
    ax.set_title("Causal impact (Python)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Incidence")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    ci_res.plot_path = fig_path

    return ci_res
