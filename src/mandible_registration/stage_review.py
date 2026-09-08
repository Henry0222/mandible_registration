"""Adapt each workflow stage to the existing general-registration viewer."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .core_bridge import ensure_registration_core
from .exporter import sha256_file, write_json, write_mesh
from .transforms import transformed_mesh

ensure_registration_core()

from auto_alignment.integration.review import (  # noqa: E402
    RegistrationReviewSpec,
    build_review_manifest,
    load_review_manifest,
    run_general_result_viewer,
    validate_review_manifest,
)


def export_stage_review(directory, stage_key, target_path, source_mesh, result, *, accepted=True, extra_warnings=()):
    if stage_key not in {"T_CT", "T_UPPER", "T_DELTA"}:
        raise ValueError("未知配准阶段")
    root = Path(directory).resolve()
    folder = root / "stage_views" / stage_key
    aligned_path = write_mesh(folder / "aligned.stl", transformed_mesh(source_mesh, result.transformation))
    target_path = Path(target_path).resolve(strict=True)
    quality = getattr(result, "quality", None)
    confidence = quality.position_confidence.value if quality is not None else None
    spec = RegistrationReviewSpec(
        target_path=target_path,
        aligned_path=aligned_path.name,
        status=result.status if accepted else "failed",
        position_confidence=confidence,
        confidence_display=result.confidence,
        warnings=tuple(result.warnings) + tuple(extra_warnings),
        review_only=(not accepted or result.status == "failed"),
        target_sha256=sha256_file(target_path),
        aligned_sha256=sha256_file(aligned_path),
        annotations_path="viewer_annotations.json",
        minimum_nominal_mm=-0.05,
        maximum_nominal_mm=0.05,
    )
    payload = build_review_manifest(spec)
    payload["stage_key"] = stage_key
    payload["transformation"] = np.asarray(result.transformation, dtype=float).tolist()
    if target_path.is_relative_to(root):
        payload["target_mesh"]["archived_path"] = os.path.relpath(target_path, folder)
    validate_review_manifest(payload)
    manifest = write_json(folder / "results.json", payload)
    load_review_manifest(manifest)
    return manifest


def read_stage_reviews(directory):
    paths, states, tips = {}, {}, {}
    for key in ("T_CT", "T_UPPER", "T_DELTA"):
        path = Path(directory) / "stage_views" / key / "results.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        paths[key] = path
        registration = payload["registration"]
        states[key] = "failed" if payload.get("review_only") else registration["status"]
        tips[key] = f"{key} · {registration['status']} · {registration['confidence']}\n点击查看配准彩虹图"
        if payload.get("review_only"):
            tips[key] += "\n未通过质量检查：仅查看候选，不传递到颌骨。"
    return paths, states, tips


def show_stage_review(path):
    # Reuse the complete viewer, including signed scale, nominal band, critical
    # limits, projection, model rotation, overlays and persisted local annotations.
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    target_path = Path(payload["target_mesh"]["path"])
    archived_target = payload["target_mesh"].get("archived_path")
    if archived_target and (path.parent / archived_target).is_file():
        target_path = path.parent / archived_target
    elif payload.get("stage_key") == "T_DELTA" and (path.parent / "target.stl").is_file():
        # Early stage adapters saved an absolute target path only.
        target_path = path.parent / "target.stl"
    aligned_path = Path(payload["outputs"]["aligned_current_stl"])
    if not aligned_path.is_absolute():
        aligned_path = path.parent / aligned_path
    for mesh_path, expected in ((target_path, payload["target_mesh"].get("sha256")), (aligned_path, payload.get("aligned_sha256"))):
        if expected and sha256_file(mesh_path) != expected:
            raise ValueError(f"模型内容已改变，不能按原配准结果显示：{mesh_path}")
    run_general_result_viewer(path, target_override=target_path, aligned_override=aligned_path)


def prepare_project_stage(project_path, stage_key):
    """Backfill a viewer-only adapter for older runs, without re-registration."""
    project_path = Path(project_path).resolve()
    data = json.loads(project_path.read_text(encoding="utf-8"))
    roles = {"T_CT": ("inputs", "baseline_lower", "ct_dentition_t0"),
             "T_UPPER": ("inputs", "baseline_upper", "followup_upper_in_t0"),
             "T_DELTA": ("outputs", "followup_lower_in_t0", "baseline_lower_at_t1")}
    group, target_key, aligned_key = roles[stage_key]

    def mesh_record(section, key):
        record = data[section][key]
        path = Path(record["path"])
        if not path.is_absolute():
            path = project_path.parent / path
        local = project_path.parent / "meshes" / path.name
        if section == "outputs" and local.is_file():
            path = local
        if record.get("sha256") and sha256_file(path) != record["sha256"]:
            raise ValueError(f"模型内容与历史配准不一致：{path}")
        return path.resolve(), sha256_file(path)

    target, target_hash = mesh_record(group, target_key)
    aligned, aligned_hash = mesh_record("outputs", aligned_key)
    stage = next(stage for stage in data["stages"] if stage["key"] == stage_key)
    destination = project_path.parent / "stage_views" / stage_key / "results.json"
    if destination.exists():
        return destination
    quality = stage.get("quality") or {}
    spec = RegistrationReviewSpec(
        target_path=target,
        aligned_path=os.path.relpath(aligned, destination.parent),
        status=stage.get("status", "unknown"),
        position_confidence=quality.get("position_confidence"),
        confidence_display=stage.get("confidence"),
        warnings=tuple(stage.get("warnings", ())),
        review_only=stage.get("status") == "failed",
        target_sha256=target_hash,
        aligned_sha256=aligned_hash,
        annotations_path="viewer_annotations.json",
        minimum_nominal_mm=-0.05,
        maximum_nominal_mm=0.05,
    )
    payload = build_review_manifest(spec)
    payload["stage_key"] = stage_key
    payload["registration"] = {**stage, **payload["registration"]}
    if "transformation" in stage:
        payload["transformation"] = stage["transformation"]
    if group == "outputs":
        payload["target_mesh"]["archived_path"] = os.path.relpath(target, destination.parent)
    validate_review_manifest(payload)
    manifest = write_json(destination, payload)
    load_review_manifest(manifest)
    return manifest
