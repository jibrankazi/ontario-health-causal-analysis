# ---------------------------
# Save & Print Results
# ---------------------------

def _jsonable(x):
    """Make objects JSON-serializable without extra deps."""
    import numpy as _np
    import pandas as _pd
    from pathlib import Path as _Path

    # Primitives pass through
    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    # NumPy scalars / arrays
    if isinstance(x, _np.generic):
        return x.item()
    if isinstance(x, _np.ndarray):
        return x.tolist()

    # Pandas Timestamp / NA
    if isinstance(x, (_pd.Timestamp,)):
        return x.isoformat()
    if x is _pd.NaT:
        return None

    # Paths
    if isinstance(x, _Path):
        return str(x)

    # Sets / tuples
    if isinstance(x, (set, tuple)):
        return [_jsonable(v) for v in x]

    # Dicts / lists
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]

    # Datetime
    from datetime import date, datetime
    if isinstance(x, (date, datetime)):
        return x.isoformat()

    # Fallback to string
    return str(x)


results_path = results_dir / "results.json"

did_results = {
    "att": None if math.isnan(did_att) else float(did_att),
    "se": None if math.isnan(did_se) else float(did_se),
    # Extras you might find useful for quick checks:
    "n_obs": int(len(df)),
    "n_regions": int(df["region"].nunique()),
}

psm_results = {
    "att": None if psm_att is None else float(psm_att),
    "reason": psm_reason,            # why PSM may have been skipped
    "diagnostics": {
        # ensure plain Python types:
        "n_treat_pre": int(psm_diag.get("n_treat_pre", 0)) if "n_treat_pre" in psm_diag else None,
        "n_ctrl_pre": int(psm_diag.get("n_ctrl_pre", 0)) if "n_ctrl_pre" in psm_diag else None,
        "n_treat_pre_cs": int(psm_diag.get("n_treat_pre_cs", 0)) if "n_treat_pre_cs" in psm_diag else None,
        "n_ctrl_pre_cs": int(psm_diag.get("n_ctrl_pre_cs", 0)) if "n_ctrl_pre_cs" in psm_diag else None,
        "ps_overlap_low": float(psm_diag.get("ps_overlap_low")) if "ps_overlap_low" in psm_diag else None,
        "ps_overlap_high": float(psm_diag.get("ps_overlap_high")) if "ps_overlap_high" in psm_diag else None,
        "caliper": float(psm_diag.get("caliper")) if "caliper" in psm_diag else None,
        "n_matched": int(psm_diag.get("n_matched", 0)) if "n_matched" in psm_diag else None,
        "covars": psm_diag.get("covars"),
    },
}

bsts_results = {
    "att": None if bsts_att is None else float(bsts_att),
    "reason": bsts_reason,
}

artifacts = {
    "results_path": str(results_path),
    "data_path": str(data_path),
}

metadata = {
    "policy_date": str(cfg.get("policy_date", "2021-02-01")),
    "bsts_enabled": bool(bsts_enabled),
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}

output_data = {
    "did": did_results,
    "psm": psm_results,
    "bsts": bsts_results,
    "artifacts": artifacts,
    "metadata": metadata,
}

# Write JSON (guaranteed serializable)
results_dir.mkdir(parents=True, exist_ok=True)
results_path.write_text(json.dumps(_jsonable(output_data), indent=4), encoding="utf-8")

print("\n--- Analysis Complete ---")
print(json.dumps(_jsonable(output_data), indent=4))
