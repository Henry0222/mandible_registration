"""Stable writable storage for selections in source and installed runs."""
from __future__ import annotations

import os
from pathlib import Path


def application_output_root() -> Path:
    override = os.environ.get("MANDIBLE_REGISTRATION_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").is_file():
        return source_root / "outputs"
    working_root = Path.cwd().resolve()
    if (working_root / "pyproject.toml").is_file():
        return working_root / "outputs"
    local = os.environ.get("LOCALAPPDATA")
    return (Path(local) if local else Path.home()) / "MandibleRegistration"
