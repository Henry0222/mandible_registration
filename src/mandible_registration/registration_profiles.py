"""Lightweight paths and labels for saved T_CT priority selections."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .storage import application_output_root


ROLE_LABELS = {"baseline_lower": "下颌口扫.1", "ct_dentition": "全牙列"}


def profile_path(mesh_path, role):
    if role not in ROLE_LABELS:
        raise ValueError("未知配准选区模型")
    normalized = os.path.normcase(str(Path(mesh_path).resolve()))
    key = hashlib.sha256(f"{role}|{normalized}".encode("utf-8")).hexdigest()
    return application_output_root() / "registration_selections" / f"{role}_{key}.json"


def condyle_profile_path(mesh_path):
    normalized = os.path.normcase(str(Path(mesh_path).resolve()))
    key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return application_output_root() / "condyle_selections" / f"{key}.json"


def saved_selection_keys(paths) -> set[str]:
    """Read small profile files only; expensive STL hashes remain a run-time gate."""
    result = set()
    bone = paths.get("ct_mandible")
    if bone is not None:
        try:
            data = json.loads(condyle_profile_path(bone).read_text(encoding="utf-8"))
            if data.get("kind") == "mandible_condyle_regions" and any(
                int(region.get("triangle_count", 0)) > 0
                for region in data.get("regions", {}).values()
            ):
                result.add("ct_mandible")
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    for role in ROLE_LABELS:
        mesh = paths.get(role)
        if mesh is None:
            continue
        try:
            data = json.loads(profile_path(mesh, role).read_text(encoding="utf-8"))
            if (
                data.get("kind") == "registration_priority_faces"
                and data.get("role") == role
                and int(data.get("selected_triangle_count", 0)) > 0
            ):
                result.add(role)
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    return result
