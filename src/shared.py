"""Shared helpers for configuration-driven analysis scripts."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Mapping

__all__ = ["resolve_intervention_date", "resolve_rscript"]


def _as_existing_path(path_like: str | os.PathLike[str] | None) -> str | None:
    """Return the path as a POSIX string if it exists on disk."""
    if not path_like:
        return None
    candidate = Path(path_like).expanduser()
    if candidate.exists():
        return candidate.as_posix()
    return None


def resolve_intervention_date(config: Mapping[str, Any], default: str = "2021-02-01") -> str:
    """Fetch the intervention cutoff date honoring legacy configuration keys."""
    for key in ("intervention_date", "policy_date"):
        value = config.get(key)
        if value:
            return str(value)
    return default


def resolve_rscript(config: Mapping[str, Any]) -> str:
    """Locate a usable Rscript executable based on config hints and platform defaults."""
    # 1. Explicit config override
    explicit = _as_existing_path(config.get("rscript_path"))
    if explicit:
        return explicit

    # 2. R_HOME hint (config overrides environment)
    r_home = config.get("r_home") or os.environ.get("R_HOME")
    if r_home:
        bin_name = "Rscript.exe" if os.name == "nt" else "Rscript"
        for rel in (Path("bin") / bin_name, Path("bin") / "x64" / bin_name):
            resolved = _as_existing_path(Path(r_home) / rel)
            if resolved:
                return resolved

    # 3. PATH lookup
    which_name = "Rscript.exe" if os.name == "nt" else "Rscript"
    via_path = shutil.which(which_name)
    if via_path:
        return Path(via_path).as_posix()

    # 4. Windows well-known install locations
    if os.name == "nt":
        candidates: list[Path] = []
        for env_key in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"):
            base = os.environ.get(env_key)
            if not base:
                continue
            root = Path(base) / "R"
            if not root.exists():
                continue
            candidates.extend(sorted(root.glob("R-*/bin/Rscript.exe")))
            candidates.extend(sorted(root.glob("R-*/bin/x64/Rscript.exe")))
        if candidates:
            candidates.sort()
            return candidates[-1].as_posix()

    raise FileNotFoundError(
        "Could not find Rscript. Set rscript_path or r_home in config.yaml, set R_HOME, or add R to PATH."
    )
