"""
Python implementation of causal impact estimation.
This module provides BSTS-style causal impact analysis using SARIMAX as fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass
class CausalImpactResult:
    """Results from causal impact estimation."""
    
    att: float | None
    ci: tuple[float | None, float | None]
    p: float | None
    relative_effect: float | None
    timeline: list[Mapping[str, Any]] | None
    notes: str | None
    plot_path: str | None = None
    summary_path: str | None = None


def estimate_causal_impact_python(
    panel: pd.DataFrame,
    policy_date: pd.Timestamp,
    output_dir: Path | None = None,
) -> CausalImpactResult:
    """
    Estimate causal impact using SARIMAX-based counterfactual.
    
    This is a Python-native implementation that doesn't require R or CausalImpact package.
    Uses SARIMAX to model pre-period treated-control differences and forecast counterfactual.
    
    Args:
        panel: Panel dataframe with columns: week, region, incidence, treated
        policy_date: Date of intervention
        output_dir: Optional directory to save plots and summaries
        
    Returns:
        CausalImpactResult object with ATT, CI, and optional artifacts
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        return CausalImpactResult(
            att=None,
            ci=(None, None),
            p=None,
            relative_effect=None,
            timeline=None,
            notes=f"Required packages not available: {e}",
        )
    
    try:
        # Prepare time series
        weekly = panel.copy()
        weekly["week"] = pd.to_datetime(weekly["week"], errors="coerce")
        weekly = weekly.dropna(subset=["week"])

        # Aggregate treated and control groups
        treated = (
            weekly[weekly["treated"] == 1]
            .groupby("week", as_index=False)["incidence"]
            .mean()
            .rename(columns={"incidence": "treated_inc"})
        )
        control = (
            weekly[weekly["treated"] == 0]
            .groupby("week", as_index=False)["incidence"]
            .mean()
            .rename(columns={"incidence": "control_inc"})
        )

        # Merge and compute difference
        series = pd.merge(treated, control, on="week", how="inner")
        series = series.sort_values("week").reset_index(drop=True)
        
        if series.empty:
            raise RuntimeError("No overlapping treated/control weeks.")

        series["diff"] = series["treated_inc"] - series["control_inc"]

        # Split into pre and post periods
        pre = series[series["week"] < policy_date].copy()
        post = series[series["week"] >= policy_date].copy()
        
        if pre.empty:
            raise RuntimeError("No pre-intervention observations.")
        if post.empty:
            raise RuntimeError("No post-intervention observations.")

        y_pre = pre["diff"].to_numpy(dtype=float)
        
        if len(y_pre) < 8:
            raise RuntimeError(f"Insufficient pre-period observations: {len(y_pre)} (need >= 8)")

        # Fit SARIMAX model with multiple specifications
        best_model = None
        best_aic = float("inf")
        candidates = [
            (1, 0, 0),  # AR(1)
            (0, 1, 1),  # MA(1) with differencing
            (1, 1, 0),  # ARIMA(1,1,0)
            (0, 1, 0),  # Random walk with drift
            (1, 1, 1),  # ARIMA(1,1,1)
            (2, 1, 0),  # ARIMA(2,1,0)
            (0, 1, 2),  # ARIMA(0,1,2)
        ]
        
        for order in candidates:
            try:
                model = SARIMAX(
                    y_pre,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False, maxiter=200)
                
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_model = result
            except Exception:
                continue

        if best_model is None:
            raise RuntimeError("SARIMAX failed to converge for all candidate orders.")

        # Forecast post-period
        horizon = len(post)
        forecast = best_model.get_forecast(steps=horizon)
        predicted = np.asarray(forecast.predicted_mean, dtype=float)
        
        # Get confidence intervals
        conf_int = forecast.conf_int(alpha=0.05)
        conf_arr = np.asarray(conf_int, dtype=float)
        lower = conf_arr[:, 0]
        upper = conf_arr[:, -1]
        
        # Compute effects
        actual_post = post["diff"].to_numpy(dtype=float)
        effect = actual_post - predicted
        
        # Average treatment effect
        att = float(np.mean(effect))
        
        # Confidence interval for ATT (based on forecast uncertainty)
        eff_lo = actual_post - upper
        eff_hi = actual_post - lower
        ci_lo = float(np.mean(eff_lo))
        ci_hi = float(np.mean(eff_hi))
        
        # Relative effect (if meaningful)
        mean_predicted = np.mean(predicted)
        if abs(mean_predicted) > 1e-6:
            relative_effect = att / abs(mean_predicted)
        else:
            relative_effect = None
        
        # Build timeline
        timeline = []
        for i, week in enumerate(post["week"].to_list()):
            timeline.append({
                "date": pd.Timestamp(week).date().isoformat(),
                "actual": float(actual_post[i]),
                "predicted": float(predicted[i]),
                "lower": float(lower[i]),
                "upper": float(upper[i]),
                "effect": float(effect[i]),
            })
        
        # Generate plots if output directory provided
        plot_path = None
        summary_path = None
        
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create plot
            try:
                sns.set_theme(style="whitegrid")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Top panel: Observed vs predicted
                all_dates = series["week"]
                all_actual = series["diff"]
                
                ax1.plot(all_dates, all_actual, label="Observed", color="black", linewidth=2)
                ax1.plot(post["week"], predicted, label="Counterfactual", 
                        color="blue", linewidth=2, linestyle="--")
                ax1.fill_between(post["week"], lower, upper, 
                                color="blue", alpha=0.2, label="95% CI")
                ax1.axvline(policy_date, color="red", linestyle=":", 
                           linewidth=2, label="Intervention")
                ax1.set_ylabel("Treated - Control Difference")
                ax1.set_title("Causal Impact Analysis: Observed vs Counterfactual")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Bottom panel: Point-wise effect
                ax2.plot(post["week"], effect, color="green", linewidth=2, label="Causal Effect")
                ax2.fill_between(post["week"], eff_lo, eff_hi, 
                                color="green", alpha=0.2, label="95% CI")
                ax2.axhline(0, color="black", linestyle="-", linewidth=0.5)
                ax2.axvline(policy_date, color="red", linestyle=":", linewidth=2)
                ax2.set_xlabel("Date")
                ax2.set_ylabel("Point-wise Effect")
                ax2.set_title(f"Average Treatment Effect: {att:.2f} [{ci_lo:.2f}, {ci_hi:.2f}]")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                fig.tight_layout()
                plot_file = output_dir / "fig_causalimpact.png"
                fig.savefig(plot_file, dpi=300, bbox_inches="tight")
                plt.close(fig)
                plot_path = plot_file.as_posix()
            except Exception as e:
                print(f"Warning: Failed to generate plot: {e}")
            
            # Create summary text
            try:
                summary_lines = [
                    "=" * 70,
                    "CAUSAL IMPACT ANALYSIS (Python SARIMAX Implementation)",
                    "=" * 70,
                    "",
                    "Model Specification:",
                    f"  Best SARIMAX order: {best_model.specification['order']}",
                    f"  AIC: {best_aic:.2f}",
                    f"  Pre-period observations: {len(y_pre)}",
                    f"  Post-period observations: {horizon}",
                    "",
                    "Average Treatment Effect (ATT):",
                    f"  Point estimate: {att:.4f}",
                    f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]",
                ]
                
                if relative_effect is not None:
                    summary_lines.append(f"  Relative effect: {relative_effect:.2%}")
                
                summary_lines.extend([
                    "",
                    "Interpretation:",
                    f"  The intervention caused an average {'increase' if att > 0 else 'decrease'}",
                    f"  of {abs(att):.2f} in the treated-control difference.",
                    "",
                    "=" * 70,
                ])
                
                summary_file = output_dir / "causalimpact_summary.txt"
                summary_file.write_text("\n".join(summary_lines))
                summary_path = summary_file.as_posix()
            except Exception as e:
                print(f"Warning: Failed to generate summary: {e}")
        
        return CausalImpactResult(
            att=att,
            ci=(ci_lo, ci_hi),
            p=None,  # P-value not directly available from SARIMAX
            relative_effect=relative_effect,
            timeline=timeline,
            notes=None,
            plot_path=plot_path,
            summary_path=summary_path,
        )
        
    except Exception as e:
        return CausalImpactResult(
            att=None,
            ci=(None, None),
            p=None,
            relative_effect=None,
            timeline=None,
            notes=f"Causal impact estimation failed: {e}",
        )
