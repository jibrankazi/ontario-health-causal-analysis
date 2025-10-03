"""Shared helpers for configuration-driven analysis scripts."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

__all__ = [
    "ROOT",
    "load_config",
    "resolve_intervention_date",
    "resolve_rscript",
]

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


def resolve_rscript(config: Mapping[str, Any]) -> str:
    """Locate an ``Rscript`` executable via config hints, env vars, or PATH."""
    def _candidate(path: Path | str | None) -> str | None:
        if not path:
            return None
        p = Path(path).expanduser()
        return p.as_posix() if p.exists() else None

    explicit = _candidate(config.get("rscript_path"))
    if explicit:
        return explicit

    r_home = config.get("r_home") or os.environ.get("R_HOME")
    if r_home:
        bin_name = "Rscript.exe" if os.name == "nt" else "Rscript"
        for suffix in (Path("bin") / bin_name, Path("bin") / "x64" / bin_name):
            resolved = _candidate(Path(r_home) / suffix)
            if resolved:
                return resolved

    which_name = "Rscript.exe" if os.name == "nt" else "Rscript"
    via_path = shutil.which(which_name)
    if via_path:
        return Path(via_path).as_posix()

    if os.name == "nt":
        search_roots = []
        for env_key in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"):
            base = os.environ.get(env_key)
            if base:
                search_roots.append(Path(base) / "R")
        candidates: list[Path] = []
        for root in search_roots:
            if not root.exists():
                continue
            candidates.extend(sorted(root.glob("R-*/bin/Rscript.exe")))
            candidates.extend(sorted(root.glob("R-*/bin/x64/Rscript.exe")))
        if candidates:
            candidates.sort()
            return candidates[-1].as_posix()

    raise FileNotFoundError(
        "Could not locate Rscript. Set `rscript_path` or `r_home` in config.yaml, set R_HOME, or add R to PATH."
    )
