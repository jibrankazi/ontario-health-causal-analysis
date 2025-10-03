"""Shared helpers for configuration-driven analysis scripts (Python-only)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

__all__ = ["ROOT", "load_config", "resolve_intervention_date"]

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load YAML configuration, returning an empty dict when missing."""
    from yaml import safe_load

    cfg_path = path or (ROOT / "config.yaml")
    try:
        return safe_load(cfg_path.read_text())
    except FileNotFoundError:
        return {}


def resolve_intervention_date(config: Mapping[str, Any], default: str = "2021-02-01") -> pd.Timestamp:
    """Return the intervention cutoff, honoring the legacy ``policy_date`` key."""
    for key in ("intervention_date", "policy_date"):
        value = config.get(key)
        if value:
            return pd.Timestamp(value)
    return pd.Timestamp(default)
