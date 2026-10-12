"""Regression checks for actual report execution and truthful interpretation."""
import json
from pathlib import Path

import pytest

import scripts.regenerate_figures as figure


def test_missing_bsts_is_not_plotted_as_zero():
    assert figure.available_estimates({
        "did": {"att": -108.61},
        "psm": {"att": -119.85},
        "bsts": {"att": None},
    }) == {"DiD": -108.61, "PSM": -119.85}


def test_nan_or_no_actual_result_fails_closed():
    with pytest.raises(ValueError, match="finite"):
        figure.available_estimates({"did": {"att": float("nan")}})
    with pytest.raises(ValueError, match="No finite"):
        figure.available_estimates({"bsts": {"att": None}})


def test_running_figure_module_writes_real_png_not_empty_placeholder(tmp_path, monkeypatch):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({
        "did": {"att": -12.5}, "psm": {"att": -9.0}, "bsts": {"att": None}
    }))
    monkeypatch.setattr(figure, "RES", path)
    monkeypatch.setattr(figure, "FIG", tmp_path / "figures")
    output = figure.main()
    assert output.exists() and output.stat().st_size > 1000
    assert output.read_bytes()[:8] == bytes.fromhex("89504e470d0a1a0a")


def test_actual_automated_report_contains_image_and_source_caveat():
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "results/results.json").read_text())
    assert raw["bsts"]["att"] is None or isinstance(raw["bsts"]["att"], (int, float))
    page = (root / "reports/report.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in page, "Figure was never generated or embedded"
    assert "not established" in page.lower() or "not validated" in page.lower()
    assert "not been reconciled" in page.lower()
    assert "−245.82" not in page, "Stale unverified old report result"


def test_all_checked_in_public_html_pages_do_not_repeat_stale_causal_claims():
    root = Path(__file__).resolve().parents[1]
    for filename in ("docs/index.html", "docs/analysis.html"):
        page = (root / filename).read_text(encoding="utf-8").lower()
        assert "not established" in page
        assert "245.82" not in page
        assert "verified" in page
