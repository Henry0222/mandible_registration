"""Checks for preserving the original relative pose of CT-derived meshes."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _value(facts, key):
    return facts[key] if isinstance(facts, Mapping) else getattr(facts, key)


def ct_frame_compatibility(dentition_facts, mandible_facts) -> dict[str, float | bool]:
    """Conservatively detect CT files whose original relative pose was lost.

    A full dentition and its mandible need not have similar shapes, but their
    axis-aligned boxes must occupy a meaningful common volume in the shared CT
    export frame. Degenerate synthetic meshes are left unassessed.
    """
    dentition_min = np.asarray(_value(dentition_facts, "bounds_min"), dtype=float)
    dentition_max = np.asarray(_value(dentition_facts, "bounds_max"), dtype=float)
    mandible_min = np.asarray(_value(mandible_facts, "bounds_min"), dtype=float)
    mandible_max = np.asarray(_value(mandible_facts, "bounds_max"), dtype=float)
    dentition_extent = dentition_max - dentition_min
    mandible_extent = mandible_max - mandible_min
    smaller_extent = np.minimum(dentition_extent, mandible_extent)
    if np.any(smaller_extent <= 1e-6):
        return {"assessed": False, "compatible": True}
    shared_extent = np.maximum(
        0.0, np.minimum(dentition_max, mandible_max) - np.maximum(dentition_min, mandible_min)
    )
    overlap_fraction = float(np.prod(shared_extent / smaller_extent))
    center_distance = float(np.linalg.norm(
        (dentition_min + dentition_max) / 2 - (mandible_min + mandible_max) / 2
    ))
    reference_diagonal = max(
        float(_value(dentition_facts, "diagonal_mm")),
        float(_value(mandible_facts, "diagonal_mm")),
        1e-9,
    )
    center_distance_ratio = center_distance / reference_diagonal
    compatible = overlap_fraction >= 0.03 or center_distance_ratio <= 0.35
    return {
        "assessed": True,
        "compatible": compatible,
        "aabb_overlap_fraction": overlap_fraction,
        "center_distance_mm": center_distance,
        "center_distance_ratio": center_distance_ratio,
    }
