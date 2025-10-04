from pathlib import Path
import json
import math

def _is_numeric(x):
    try:
        f = float(x)
        return not (isinstance(f, float) and math.isnan(f))
    except Exception:
        return False


def test_results_json_exists():
    """
    Smoke test: ensure results.json exists and has the essential top-level keys.
    Accepts the current flat schema (did_att, psm_att, bsts_att, meta).
    """
    f = Path("results/results.json")
    assert f.exists(), f"Error: The results file {f} does not exist."

    try:
        r = json.loads(f.read_text())
    except json.JSONDecodeError as e:
        raise AssertionError(f"Error: Could not decode {f} as JSON. Error: {e}")

    # Flat schema keys written by the pipeline
    essential_keys = ["did_att", "psm_att", "bsts_att", "meta"]
    missing = [k for k in essential_keys if k not in r]
    assert not missing, f"Error: Missing keys in results.json: {missing}"


def test_at_least_one_numeric_att():
    """
    Ensure at least one ATT is numeric (BSTS may legitimately be None).
    """
    r = json.loads(Path("results/results.json").read_text())

    did  = r.get("did_att")
    psm  = r.get("psm_att")
    bsts = r.get("bsts_att")

    any_numeric = any(_is_numeric(v) for v in [did, psm, bsts])
    assert any_numeric, "At least one ATT must be numeric among did_att, psm_att, bsts_att."


def test_meta_present_and_well_formed():
    """
    Basic sanity check on meta block (exists and is a dict).
    """
    r = json.loads(Path("results/results.json").read_text())
    meta = r.get("meta")
    assert isinstance(meta, dict), "meta must exist and be a JSON object (dict)."
