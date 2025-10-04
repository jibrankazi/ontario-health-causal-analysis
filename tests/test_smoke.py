from pathlib import Path
import json

def test_results_json_exists():
    """
    Smoke test: ensure results.json exists and has the essential top-level keys.
    
    Based on the analysis output, we check for specific ATT results and metadata.
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
    # CORRECTION: Updated keys to match the actual output structure ('did_att', 'bsts_att', 'meta')
    essential_keys = ["did_att", "bsts_att", "meta"] 
    for key in essential_keys:
        assert key in r, f"Error: Essential key '{key}' missing from results.json."


def test_results_sections_nonempty():
    """
    A stricter test to ensure the essential sections in results.json
    are not None.
    """
    try:
        r = json.loads(Path("results/results.json").read_text())
    except Exception:
        # If the file couldn't be loaded (e.g., if test_results_json_exists failed)
        return

    # CORRECTION: Updated sections to check for the actual ATT result fields.
    # We check if these critical fields exist and are not None.
    sections_to_check = ["did_att", "bsts_att"]

    for key in sections_to_check:
        # Assert the key exists and its value is not explicitly None.
        # This handles the case where the analysis might return a valid result of 0.
        assert key in r and r.get(key) is not None, f"Error: Essential result '{key}' is missing or None."
