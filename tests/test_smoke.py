from pathlib import Path
import json
import math

def _pick_att(r, method):
    flat = r.get(f"{method}_att")
    if flat is not None:
        return flat
    nested = r.get(method)
    if isinstance(nested, dict):
        return nested.get("att")
    return None

def test_results_json_exists():
    f = Path("results/results.json")
    assert f.exists(), f"Error: The results file {f} does not exist."
    try:
        r = json.loads(f.read_text())
    except json.JSONDecodeError as e:
        raise AssertionError(f"Error: Could not decode {f} as JSON. Error: {e}")

    # Accept either schema: flat keys or nested sections.
    has_flat   = all(k in r for k in ["did_att","psm_att","bsts_att"])
    has_nested = all(k in r for k in ["did","psm","bsts"])
    assert has_flat or has_nested, "results.json must have either flat keys (did_att/psm_att/bsts_att) or nested keys (did/psm/bsts)."

def test_results_sections_nonempty():
    r = json.loads(Path("results/results.json").read_text())
    for method in ["did","psm","bsts"]:
        val = _pick_att(r, method)
        # allow None for methods that didn't run (e.g., BSTS), but require at least one numeric ATT overall
        if val is not None:
            try:
                float(val)
            except Exception:
                raise AssertionError(f"{method}_att is not numeric: {val!r}")

    # ensure at least one estimator produced a numeric ATT
    numeric_any = any(
        isinstance(_pick_att(r, m), (int, float)) and not (isinstance(_pick_att(r, m), float) and math.isnan(_pick_att(r, m)))
        for m in ["did","psm","bsts"]
    )
    assert numeric_any, "At least one ATT must be numeric (DiD, PSM, or BSTS)."
