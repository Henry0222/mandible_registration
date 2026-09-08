from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import uuid

import numpy as np
import open3d as o3d


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: str | Path, payload: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(
                _json_safe(payload),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_mesh(path: str | Path, mesh: o3d.geometry.TriangleMesh) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.parent / f".{destination.stem}.{uuid.uuid4().hex}{destination.suffix}"
    try:
        # Open3D on Windows can fail when it writes directly through a path
        # containing Chinese characters. Write to an ASCII temporary path and
        # atomically move the bytes into the requested destination.
        with tempfile.TemporaryDirectory(prefix="mandible_registration_") as temporary:
            safe = Path(temporary) / f"mesh{destination.suffix.lower()}"
            if not o3d.io.write_triangle_mesh(str(safe), mesh, write_ascii=False):
                raise OSError(f"Open3D 无法写入网格：{destination}")
            shutil.copyfile(safe, staged)
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)
    return destination


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamped_run_directory(root: str | Path) -> Path:
    base = Path(root).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().astimezone().strftime("mandible_%Y%m%d_%H%M%S")
    candidate = base / stem
    suffix = 1
    while candidate.exists():
        candidate = base / f"{stem}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate
