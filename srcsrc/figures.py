"""Utilities for generating analysis figures and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

__all__ = [
    "FigureArtifacts",
    "generate_analysis_figures",
]


@dataclass
class FigureArtifacts:
    """Paths (relative to the project root) for generated figures."""

    event_study: str | None = None
    balance: str | None = None
    impact: str | None = None

    def as_list(self) -> list[str]:
        return [p for p in (self.event_study, self.balance, self.impact) if p]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _prepare_week_index(df: pd.DataFrame) -> pd.Series:
    week = df["week"]
    if not pd.api.types.is_datetime64_any_dtype(week):
        week = pd.to_datetime(week, errors="coerce")
    return week


def _plot_event_study(panel: pd.DataFrame, policy_date: pd.Timestamp, out_path: Path) -> bool:
    weekly = (
        panel.assign(week=_prepare_week_index(panel))
        .dropna(subset=["week"])
        .groupby(["week", "treated"], as_index=False)["incidence"]
        .mean()
    )
    if weekly.empty:
        return False

    pivot = weekly.pivot(index="week", columns="treated", values="incidence").sort_index()
    pivot = pivot.rename(columns={0: "Control", 1: "Treated"})
    if pivot.empty:
        return False

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    if "Treated" in pivot:
        ax.plot(pivot.index, pivot["Treated"], label="Treated", linewidth=2.0)
    if "Control" in pivot:
        ax.plot(pivot.index, pivot["Control"], label="Control", linewidth=2.0, linestyle="--")
    if "Treated" in pivot and "Control" in pivot:
        diff = pivot["Treated"] - pivot["Control"]
        ax.fill_between(diff.index, diff, 0, color="tab:blue", alpha=0.1, label="Treated - Control")

    ax.axvline(policy_date, color="tab:red", linestyle=":", linewidth=2, label="Intervention")
    ax.set_xlabel("Week")
    ax.set_ylabel("Mean incidence")
    ax.set_title("Weekly incidence: treated vs. control regions")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True


def _compute_smd(treated: pd.DataFrame, control: pd.DataFrame, covariates: Sequence[str]) -> Mapping[str, float]:
    smd: dict[str, float] = {}
    for cov in covariates:
        if cov not in treated.columns or cov not in control.columns:
            continue
        t = pd.to_numeric(treated[cov], errors="coerce").dropna()
        c = pd.to_numeric(control[cov], errors="coerce").dropna()
        if t.empty or c.empty:
            continue
        mean_t = t.mean()
        mean_c = c.mean()
        var_t = t.var(ddof=1)
        var_c = c.var(ddof=1)
        pooled = np.sqrt(max(1e-12, (var_t + var_c) / 2.0))
        smd[cov] = float((mean_t - mean_c) / pooled)
    return smd


def _plot_balance(
    pre: pd.DataFrame,
    matched_treated: pd.DataFrame | None,
    matched_control: pd.DataFrame | None,
    covariates: Sequence[str],
    out_path: Path,
) -> bool:
    treated_pre = pre[pre["treated"] == 1]
    control_pre = pre[pre["treated"] == 0]
    smd_pre = _compute_smd(treated_pre, control_pre, covariates)

    smd_matched: Mapping[str, float] = {}
    if matched_treated is not None and matched_control is not None:
        smd_matched = _compute_smd(matched_treated, matched_control, covariates)

    if not smd_pre and not smd_matched:
        return False

    rows: list[dict[str, object]] = []
    for cov in covariates:
        if cov in smd_pre:
            rows.append({"covariate": cov, "Stage": "Pre-match", "SMD": smd_pre[cov]})
        if cov in smd_matched:
            rows.append({"covariate": cov, "Stage": "Matched", "SMD": smd_matched[cov]})
    if not rows:
        return False

    plot_df = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(covariates) + 2)))
    sns.barplot(data=plot_df, x="SMD", y="covariate", hue="Stage", ax=ax)
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(0.1, color="tab:red", linestyle=":", linewidth=1)
    ax.axvline(-0.1, color="tab:red", linestyle=":", linewidth=1)
    ax.set_title("Standardized mean differences")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True


def _plot_impact(
    series: Iterable[Mapping[str, object]], policy_date: pd.Timestamp, out_path: Path
) -> bool:
    series_df = pd.DataFrame(list(series))
    if series_df.empty:
        return False
    if "date" not in series_df.columns:
        return False

    series_df["date"] = pd.to_datetime(series_df["date"], errors="coerce")
    series_df = series_df.dropna(subset=["date"]).sort_values("date")
    if series_df.empty:
        return False

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    if "actual" in series_df.columns:
        ax.plot(
            series_df["date"],
            series_df["actual"],
            label="Observed treated-control difference",
            linewidth=2.0,
        )
    if "predicted" in series_df.columns:
        ax.plot(
            series_df["date"],
            series_df["predicted"],
            label="SARIMAX counterfactual",
            linewidth=2.0,
            linestyle="--",
        )
    if {"lower", "upper"}.issubset(series_df.columns):
        ax.fill_between(
            series_df["date"],
            series_df["lower"],
            series_df["upper"],
            color="tab:blue",
            alpha=0.15,
            label="95% forecast interval",
        )

    ax.axvline(policy_date, color="tab:red", linestyle=":", linewidth=2, label="Intervention")
    ax.set_xlabel("Week")
    ax.set_ylabel("Treated - control incidence")
    ax.set_title("Observed vs. counterfactual difference (SARIMAX)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True


def generate_analysis_figures(
    *,
    panel: pd.DataFrame,
    policy_date: pd.Timestamp,
    pre_period: pd.DataFrame,
    covariates: Sequence[str],
    matched_treated: pd.DataFrame | None,
    matched_control: pd.DataFrame | None,
    impact_series: Iterable[Mapping[str, object]] | None,
    root: Path,
) -> FigureArtifacts:
    """Generate analysis figures and return their relative paths."""

    figures_dir = root / "figures"
    _ensure_dir(figures_dir)
    artifacts = FigureArtifacts()

    event_path = figures_dir / "fig1_event_study.png"
    if _plot_event_study(panel, policy_date, event_path):
        artifacts.event_study = event_path.relative_to(root).as_posix()

    balance_path = figures_dir / "fig2_smd_balance.png"
    if covariates and _plot_balance(pre_period, matched_treated, matched_control, covariates, balance_path):
        artifacts.balance = balance_path.relative_to(root).as_posix()

    if impact_series:
        impact_path = figures_dir / "fig3_impact_counterfactual.png"
        if _plot_impact(impact_series, policy_date, impact_path):
            artifacts.impact = impact_path.relative_to(root).as_posix()

    return artifacts
