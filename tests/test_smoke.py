from pathlib import Path
import json
import math

def _is_numeric(x):
    try:
        f = float(x)
        return not (isinstance(f, float) and math.isnan(f))
    except Exception:
        return False

def _pick_att(r, method: str):
    """
    Support both schemas:
      - flat:   did_att / psm_att / bsts_att
      - nested: {"did":{"att":...}}, {"psm":{"att":...}}, {"bsts":{"att":...}}
    """
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

    # Accept either schema
    has_flat   = all(k in r for k in ["did_att","psm_att","bsts_att"])
    has_nested = all(k in r for k in ["did","psm","bsts"])
    assert has_flat or has_nested, \
        "results.json must have either flat keys (did_att/psm_att/bsts_att) or nested keys (did/psm/bsts)."

def test_at_least_one_numeric_att():
    r = json.loads(Path("results/results.json").read_text())
    vals = [_pick_att(r, m) for m in ["did","psm","bsts"]]
    assert any(_is_numeric(v) for v in vals), \
        "At least one ATT must be numeric among DiD/PSM/BSTS."

def test_meta_present_and_well_formed():
    r = json.loads(Path("results/results.json").read_text())
    meta = r.get("meta")
    assert isinstance(meta, dict), "meta must exist and be a dict."
