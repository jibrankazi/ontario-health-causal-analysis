# tests/test_smoke.py
from pathlib import Path
import json


def test_results_json_exists():
    """
    Smoke test: ensure results.json exists and has the essential top-level keys.
    """
    f = Path("results/results.json")

    # 1. Assert the file exists.
    assert f.exists(), f"Error: The results file {f} does not exist."

    # 2. Read the file content and parse it as JSON.
    try:
        r = json.loads(f.read_text())
    except json.JSONDecodeError as e:
        # Raise an AssertionError to fail the test cleanly on a JSON decoding issue.
        raise AssertionError(f"Error: Could not decode {f} as JSON. Error: {e}")

    # 3. Assert essential keys are present in the top-level dictionary.
    essential_keys = ["did", "psm", "bsts"]
    for key in essential_keys:
        assert key in r, f"Error: Essential key '{key}' missing from results.json."


def test_results_sections_nonempty():
    """
    A stricter test to ensure the essential sections in results.json 
    are not empty (i.e., not None, [], or {}).

    NOTE: This test assumes test_results_json_exists passes first, 
    so it doesn't re-check for file existence or JSON decoding.
    """
    try:
        r = json.loads(Path("results/results.json").read_text())
    except Exception:
        # Skip this check if the file couldn't be loaded
        return

    sections_to_check = ["did", "psm", "bsts"]

    for key in sections_to_check:
        # Fails if value is None, empty list/dict/string, or 0.
        assert r.get(key), f"Error: Essential section '{key}' is empty or None."
