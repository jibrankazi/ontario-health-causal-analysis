from pathlib import Path
import json
import math

def _is_num(x):
    try:
        f = float(x)
        return not math.isnan(f)
    except Exception:
        return False

def test_results_json_exists():
    f = Path("results/results.json")
    assert f.exists(), f"Error: The results file {f} does not exist."
    try:
        r = json.loads(f.read_text())
    except json.JSONDecodeError as e:
        raise AssertionError(f"Error: Could not decode {f} as JSON. Error: {e}")

    # Expect flat schema
    for k in ["did_att", "psm_att", "bsts_att"]:
        assert k in r, f"Error: Key '{k}' missing from results.json."

def test_results_sections_nonempty():
    r = json.loads(Path("results/results.json").read_text())
    # At least one numeric ATT should be present (BSTS may be None if it failed)
    numeric_any = any(_is_num(r.get(k)) for k in ["did_att", "psm_att", "bsts_att"])
    assert numeric_any, "At least one ATT must be numeric (did_att, psm_att, or bsts_att)."
