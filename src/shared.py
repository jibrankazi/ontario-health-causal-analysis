"""
Shared utilities and configuration loading for causal analysis pipeline.
This module was referenced but missing in the original codebase.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Project root (src/..)
ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses ROOT/config.yaml
        
    Returns:
        Dictionary containing configuration
    """
    if config_path is None:
        config_path = ROOT / "config.yaml"
    
    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}, using defaults")
        return {
            "data_path": "data/ontario_cases.csv",
            "policy_date": "2021-02-01",
            "covariates": [],
        }
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config if config is not None else {}
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}: {e}")
        return {}


def resolve_intervention_date(config: dict[str, Any]) -> pd.Timestamp:
    """
    Resolve intervention date from configuration.
    
    Supports multiple config key names for backwards compatibility:
    - intervention_date
    - policy_date
    - treatment_date
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Timestamp of intervention date
    """
    # Try multiple possible config keys
    for key in ["intervention_date", "policy_date", "treatment_date"]:
        if key in config:
            date_val = config[key]
            if isinstance(date_val, str):
                return pd.Timestamp(date_val)
            elif isinstance(date_val, pd.Timestamp):
                return date_val
            elif hasattr(date_val, "isoformat"):  # datetime-like object
                return pd.Timestamp(date_val)
    
    # Default fallback
    print("Warning: No intervention date found in config, using default 2021-02-01")
    return pd.Timestamp("2021-02-01")


def validate_panel_data(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate that panel data has required structure.
    
    Args:
        df: Panel dataframe to validate
        
    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []
    
    # Check required columns
    required = {"week", "region", "incidence", "treated"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
    
    # Check data types if columns exist
    if "week" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["week"]):
            try:
                pd.to_datetime(df["week"], errors="raise")
            except Exception:
                errors.append("'week' column cannot be converted to datetime")
    
    if "treated" in df.columns:
        unique_vals = set(df["treated"].dropna().unique())
        if not unique_vals.issubset({0, 1}):
            errors.append(f"'treated' column must contain only 0/1, found: {unique_vals}")
    
    if "incidence" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["incidence"]):
            errors.append("'incidence' column must be numeric")
    
    # Check for reasonable data volume
    if len(df) == 0:
        errors.append("DataFrame is empty")
    elif len(df) < 10:
        errors.append(f"DataFrame has only {len(df)} rows, which may be insufficient")
    
    if "region" in df.columns:
        n_regions = df["region"].nunique()
        if n_regions < 2:
            errors.append(f"Need at least 2 regions, found {n_regions}")
    
    return (len(errors) == 0, errors)


def summarize_panel(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generate summary statistics for panel data.
    
    Args:
        df: Panel dataframe
        
    Returns:
        Dictionary of summary statistics
    """
    summary = {
        "n_rows": len(df),
        "n_regions": df["region"].nunique() if "region" in df.columns else None,
        "date_range": None,
        "n_treated": None,
        "n_control": None,
        "pre_periods": None,
        "post_periods": None,
    }
    
    if "week" in df.columns:
        weeks = pd.to_datetime(df["week"], errors="coerce").dropna()
        if not weeks.empty:
            summary["date_range"] = (
                weeks.min().date().isoformat(),
                weeks.max().date().isoformat(),
            )
    
    if "treated" in df.columns:
        summary["n_treated"] = int((df["treated"] == 1).sum())
        summary["n_control"] = int((df["treated"] == 0).sum())
    
    if "post" in df.columns:
        summary["pre_periods"] = int((df["post"] == 0).sum())
        summary["post_periods"] = int((df["post"] == 1).sum())
    
    return summary


def format_results_table(results: dict[str, Any]) -> str:
    """
    Format results dictionary as a readable table.
    
    Args:
        results: Results dictionary from analysis
        
    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 60)
    lines.append("CAUSAL ANALYSIS RESULTS")
    lines.append("=" * 60)
    
    # DiD results
    if "did" in results:
        did = results["did"]
        lines.append("\nDifference-in-Differences:")
        lines.append(f"  ATT:    {did.get('att', 'N/A')}")
        lines.append(f"  SE:     {did.get('se', 'N/A')}")
        if did.get("notes"):
            lines.append(f"  Notes:  {did['notes']}")
    
    # PSM results
    if "psm" in results:
        psm = results["psm"]
        lines.append("\nPropensity Score Matching:")
        lines.append(f"  ATT:    {psm.get('att', 'N/A')}")
        if psm.get("diagnostics"):
            diag = psm["diagnostics"]
            lines.append(f"  Matched pairs: {diag.get('n_matched', 'N/A')}")
            lines.append(f"  Covariates: {', '.join(diag.get('covariates', []))}")
        if psm.get("notes"):
            lines.append(f"  Notes:  {psm['notes']}")
    
    # BSTS/Impact results
    if "bsts" in results:
        bsts = results["bsts"]
        lines.append("\nBayesian Structural Time Series / Causal Impact:")
        lines.append(f"  ATT:    {bsts.get('att', 'N/A')}")
        ci = bsts.get("ci", [])
        if ci and len(ci) == 2:
            lines.append(f"  95% CI: [{ci[0]}, {ci[1]}]")
        if bsts.get("p") is not None:
            lines.append(f"  P-value: {bsts['p']}")
        if bsts.get("notes"):
            lines.append(f"  Notes:  {bsts['notes']}")
    
    # Metadata
    if "metadata" in results:
        meta = results["metadata"]
        lines.append("\nMetadata:")
        lines.append(f"  Intervention date: {meta.get('intervention_date', 'N/A')}")
        lines.append(f"  N regions:         {meta.get('n_regions', 'N/A')}")
        lines.append(f"  N observations:    {meta.get('n_rows', 'N/A')}")
        lines.append(f"  Generated at:      {meta.get('generated_at', 'N/A')}")
    
    # Artifacts
    if "artifacts" in results:
        artifacts = results["artifacts"]
        lines.append("\nGenerated figures:")
        for name, path in artifacts.items():
            if path:
                lines.append(f"  {name:15s} -> {path}")
    
    lines.append("=" * 60)
    return "\n".join(lines)


def safe_json_serializer(obj: Any) -> Any:
    """
    JSON serializer for objects not serializable by default json module.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable representation
    """
    if pd.isna(obj):
        return None
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if hasattr(obj, "isoformat"):  # datetime-like
        return obj.isoformat()
    if hasattr(obj, "__float__"):  # numpy types
        return float(obj)
    if hasattr(obj, "__int__"):
        return int(obj)
    
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
