from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
import traceback
from typing import Callable

import numpy as np
import open3d as o3d

from .core_bridge import ensure_registration_core
from .ct_coordinates import ct_frame_compatibility
from .exporter import sha256_file, timestamped_run_directory, write_json, write_mesh
from .models import StudyInputs
from .transforms import compose, rotation_degrees, transformed_mesh, translation_mm
from .stage_review import export_stage_review


ensure_registration_core()

from auto_alignment.integration import GENERAL_MODEL_REGISTRATION_VERSION as registration_core_version  # noqa: E402
from auto_alignment.integration.core import (  # noqa: E402
    AlignmentConfig,
    MeshFacts,
    RegistrationResult,
    load_mesh,
    register_meshes,
)


ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class StageUpdate:
    """One stage transition; independent of human-readable progress messages."""

    key: str
    status: str
    run_directory: Path
    review_path: Path | None = None
    confidence: str = ""
    ready_outputs: tuple[str, ...] = ()


StageCallback = Callable[[StageUpdate], None]


def _notify_stage(callback, run_directory, key, *, result=None, review_path=None, ready_outputs=()):
    if callback is not None:
        callback(StageUpdate(
            key, "running" if result is None else result.status, run_directory,
            review_path, "" if result is None else result.confidence, tuple(ready_outputs),
        ))


class WorkflowError(RuntimeError):
    def __init__(self, message, *, candidate=None, diagnostics=None, run_directory=None):
        super().__init__(message)
        self.candidate = candidate
        self.diagnostics = diagnostics
        self.run_directory = run_directory


@dataclass(frozen=True)
class StageOutcome:
    key: str
    title: str
    fixed_name: str
    moving_name: str
    result: RegistrationResult
    diagnostics: dict[str, object] | None = None

    @property
    def transformation(self) -> np.ndarray:
        return np.asarray(self.result.transformation, dtype=float)

    def as_dict(self) -> dict[str, object]:
        quality = self.result.quality
        return {
            "key": self.key,
            "title": self.title,
            "fixed": self.fixed_name,
            "moving": self.moving_name,
            "matrix_convention": "column_vector; fixed_point = matrix @ moving_point",
            "transformation": self.transformation.tolist(),
            "status": self.result.status,
            "confidence": self.result.confidence,
            "metrics": self.result.metrics.as_dict(),
            "quality": quality.as_dict() if quality is not None else None,
            "warnings": list(self.result.warnings),
            "elapsed_seconds": self.result.elapsed_seconds,
            "workflow_diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class StudyOutcome:
    run_directory: Path
    stages: tuple[StageOutcome, ...]
    t_ct: np.ndarray
    t_upper: np.ndarray
    t_delta: np.ndarray
    t_mandible_t0: np.ndarray
    t_mandible_t1: np.ndarray
    output_files: dict[str, Path]

    def as_dict(self) -> dict[str, object]:
        return {
            "run_directory": str(self.run_directory),
            "stages": [stage.as_dict() for stage in self.stages],
            "transforms": {
                "T_CT": self.t_ct.tolist(),
                "T_UPPER": self.t_upper.tolist(),
                "T_DELTA": self.t_delta.tolist(),
                "T_MANDIBLE_T0": self.t_mandible_t0.tolist(),
                "T_MANDIBLE_T1": self.t_mandible_t1.tolist(),
            },
            "output_files": {key: str(path) for key, path in self.output_files.items()},
        }


def _notify(callback: ProgressCallback | None, fraction: float, message: str) -> None:
    if callback is not None:
        callback(max(0.0, min(1.0, float(fraction))), message)


def _scaled_progress(
    callback: ProgressCallback | None,
    *,
    offset: float,
    scale: float,
    prefix: str,
) -> ProgressCallback | None:
    if callback is None:
        return None

    def report(fraction: float, message: str) -> None:
        _notify(callback, offset + scale * float(fraction), f"{prefix}：{message}")

    return report


def _transformed_facts(facts: MeshFacts, mesh: o3d.geometry.TriangleMesh) -> MeshFacts:
    bounds = mesh.get_axis_aligned_bounding_box()
    extent = np.asarray(bounds.get_extent(), dtype=float)
    return replace(
        facts,
        diagonal_mm=float(np.linalg.norm(extent)),
        bounds_min=tuple(float(value) for value in bounds.min_bound),
        bounds_max=tuple(float(value) for value in bounds.max_bound),
    )


def _require_success(stage: StageOutcome, *, allow_warning: bool = False) -> None:
    if stage.result.status == "success" or (allow_warning and stage.result.succeeded):
        return
    details = list(stage.result.warnings)
    if stage.result.quality is not None:
        details.extend(stage.result.quality.reasons)
    suffix = "；".join(dict.fromkeys(details)) or "质量检查未通过"
    raise WorkflowError(f"{stage.title}失败：{suffix}")


def _matrix_payload(name: str, matrix: np.ndarray, meaning: str) -> dict[str, object]:
    return {
        "name": name,
        "meaning": meaning,
        "matrix_convention": "4x4 homogeneous rigid transform for column vectors",
        "units": "millimetres",
        "matrix": np.asarray(matrix, dtype=float).tolist(),
        "rotation_degrees": rotation_degrees(matrix),
        "translation_mm": translation_mm(matrix),
    }


def _transform_disagreement(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    relative = np.asarray(first, dtype=float) @ np.linalg.inv(np.asarray(second, dtype=float))
    return {
        "rotation_degrees": rotation_degrees(relative),
        "translation_mm": translation_mm(relative),
    }


def _require_shared_ct_frame(
    dentition_facts: MeshFacts,
    bone_facts: MeshFacts,
    *,
    bone_name: str = "颌骨",
) -> None:
    report = ct_frame_compatibility(dentition_facts, bone_facts)
    if report["compatible"]:
        return
    dentition_name = Path(dentition_facts.path).name
    raise WorkflowError(
        f"全牙列与{bone_name}不像处于同一原始 CT 坐标系"
        f"（包围盒交叠 {100 * float(report['aabb_overlap_fraction']):.1f}%，"
        f"中心相距 {float(report['center_distance_mm']):.1f} mm）。"
        f"当前全牙列为 {dentition_name}；它可能是已单独配准到口扫的中间文件。"
        f"请改选与{bone_name}同时从 CT 软件导出、未经单独移动的全牙列 STL。"
        f"已停止配准，以免把错误矩阵传给{bone_name}"
    )


def _ct_attempt_is_strong(result: RegistrationResult) -> bool:
    quality = result.quality
    return bool(
        result.succeeded
        and quality is not None
        and quality.target_coverage_ratio >= 0.25
        and quality.normal_consistency_ratio >= 0.70
        and quality.residual_p90_mm is not None
        and quality.residual_p90_mm <= 0.80
    )


def _ct_attempt_score(result: RegistrationResult) -> float:
    quality = result.quality
    if quality is None:
        return float("-inf")
    confidence_bonus = {"高": 0.30, "中": 0.15, "低": 0.0}.get(result.confidence, -0.50)
    p90 = quality.residual_p90_mm if quality.residual_p90_mm is not None else 10.0
    return float(
        2.0 * quality.target_coverage_ratio
        + quality.directed_overlap_ratio
        + quality.normal_consistency_ratio
        - 0.5 * p90
        + confidence_bonus
    )


def _ct_attempt_payload(
    index: int, seed: int, result: RegistrationResult
) -> dict[str, object]:
    quality = result.quality
    score = _ct_attempt_score(result)
    return {
        "attempt": index,
        "random_seed": seed,
        "strong_quality_gate_passed": _ct_attempt_is_strong(result),
        "score": score if np.isfinite(score) else None,
        "status": result.status,
        "confidence": result.confidence,
        "transformation": np.asarray(result.transformation, dtype=float).tolist(),
        "metrics": result.metrics.as_dict(),
        "quality": quality.as_dict() if quality is not None else None,
        "warnings": list(result.warnings),
        "elapsed_seconds": result.elapsed_seconds,
    }


def _register_ct_with_consensus(
    target_mesh: o3d.geometry.TriangleMesh,
    source_mesh: o3d.geometry.TriangleMesh,
    target_facts: MeshFacts,
    source_facts: MeshFacts,
    config: AlignmentConfig,
    run_directory: Path,
    progress: ProgressCallback | None,
    *,
    target_priority_faces: np.ndarray | None = None,
    source_priority_faces: np.ndarray | None = None,
) -> tuple[RegistrationResult, dict[str, object]]:
    """Require two independent CT registrations to agree before propagation."""

    attempts: list[RegistrationResult] = []
    seeds: list[int] = []
    maximum_attempts = 5
    for index in range(maximum_attempts):
        seed = int(config.random_seed) + index
        attempt_config = replace(config, random_seed=seed)
        priority_arguments = {}
        if target_priority_faces is not None:
            priority_arguments["target_priority_faces"] = target_priority_faces
        if source_priority_faces is not None:
            priority_arguments["source_priority_faces"] = source_priority_faces
        result = register_meshes(
            target_mesh,
            source_mesh,
            target_facts,
            source_facts,
            attempt_config,
            _scaled_progress(
                progress,
                offset=0.04 + index * 0.39 / maximum_attempts,
                scale=0.39 / maximum_attempts,
                prefix=f"阶段 1/3 全牙列→下颌口扫.1（一致性运行 {index + 1}/{maximum_attempts}）",
            ),
            **priority_arguments,
        )
        attempts.append(result)
        seeds.append(seed)
        write_json(
            run_directory / "stages" / f"01_T_CT_attempt_{index + 1}.json",
            _ct_attempt_payload(index + 1, seed, result),
        )

        if _ct_attempt_is_strong(result):
            agrees_with_previous = False
            for previous in attempts[:-1]:
                disagreement = _transform_disagreement(previous.transformation, result.transformation)
                if (_ct_attempt_is_strong(previous)
                        and disagreement["rotation_degrees"] <= 2.0
                        and disagreement["translation_mm"] <= 2.0):
                    agrees_with_previous = True
                    break
            if agrees_with_previous:
                break

    pairwise: list[dict[str, object]] = []
    agreeing_pairs: list[tuple[int, int]] = []
    for first in range(len(attempts)):
        for second in range(first + 1, len(attempts)):
            disagreement = _transform_disagreement(
                attempts[first].transformation, attempts[second].transformation
            )
            agrees = bool(
                _ct_attempt_is_strong(attempts[first])
                and _ct_attempt_is_strong(attempts[second])
                and disagreement["rotation_degrees"] <= 2.0
                and disagreement["translation_mm"] <= 2.0
            )
            pairwise.append(
                {
                    "attempts": [first + 1, second + 1],
                    **disagreement,
                    "agrees": agrees,
                }
            )
            if agrees:
                agreeing_pairs.append((first, second))

    diagnostics = {
        "policy": (
            "At least two strong-quality attempts must agree within 2 degrees "
            "and 2 mm before T_CT is propagated."
        ),
        "attempts": [
            _ct_attempt_payload(index + 1, seeds[index], result)
            for index, result in enumerate(attempts)
        ],
        "pairwise_consistency": pairwise,
        "maximum_attempts": maximum_attempts,
        "runtime_threads": {"OpenMP": 1, "BLAS": 1},
        "priority_faces": {
            "target_selected": int(target_priority_faces.sum()) if target_priority_faces is not None else 0,
            "source_selected": int(source_priority_faces.sum()) if source_priority_faces is not None else 0,
        },
    }
    if not agreeing_pairs:
        write_json(run_directory / "stages" / "01_T_CT_consensus.json", diagnostics)
        raise WorkflowError(
            "全牙列配准未形成可靠一致解，已停止向颌骨传递矩阵；可点击红色汇总线检查候选",
            candidate=max(attempts, key=_ct_attempt_score), diagnostics=diagnostics,
        )

    best_pair = max(
        agreeing_pairs,
        key=lambda pair: _ct_attempt_score(attempts[pair[0]]) + _ct_attempt_score(attempts[pair[1]]),
    )
    member_indices = list(best_pair)
    chosen_index = max(member_indices, key=lambda index: _ct_attempt_score(attempts[index]))
    diagnostics["selected_attempt"] = chosen_index + 1
    diagnostics["consensus_members"] = [index + 1 for index in member_indices]
    write_json(run_directory / "stages" / "01_T_CT_consensus.json", diagnostics)
    _notify(progress, 0.43, f"T_CT 一致性检查通过，采用第 {chosen_index + 1} 次结果")
    return attempts[chosen_index], diagnostics


def _run_study_in_directory(
    inputs: StudyInputs,
    run_directory: Path,
    *,
    config: AlignmentConfig | None = None,
    progress: ProgressCallback | None = None,
    stage_changed: StageCallback | None = None,
) -> StudyOutcome:
    """Run the fixed six-STL workflow with an optional CT maxilla/skull mesh.

    Every registration maps the moving/source mesh into the fixed/target mesh.
    The combined CT dentition remains intact. Optional user-selected priority
    faces bias T_CT sampling but do not delete or crop either mesh.
    """

    base_config = config or AlignmentConfig()
    # Stage 1 intentionally compares a combined upper/lower CT dentition with
    # the lower IOS only. A 12% directed-overlap floor accepts this valid
    # asymmetric case while all rigidness, residual and consistency gates stay
    # enabled.
    ct_config = replace(
        base_config,
        partial_registration_enabled=True,
        partial_overlap_threshold=min(base_config.partial_overlap_threshold, 0.12),
    )

    _notify(progress, 0.0, f"已建立运行目录：{run_directory.name}")
    _notify(progress, 0.005, "读取并检查六个配准 STL 与可选上颌骨 STL")
    loaded: dict[str, tuple[o3d.geometry.TriangleMesh, MeshFacts]] = {}
    values = inputs.as_mapping()
    for index, (key, path) in enumerate(values.items(), start=1):
        loaded[key] = load_mesh(path)
        _notify(progress, 0.04 * index / len(values), f"已读取 {path.name}")

    baseline_lower, baseline_lower_facts = loaded["baseline_lower"]
    baseline_upper, baseline_upper_facts = loaded["baseline_upper"]
    ct_dentition, ct_dentition_facts = loaded["ct_dentition"]
    ct_mandible, ct_mandible_facts = loaded["ct_mandible"]
    ct_maxilla_bundle = loaded.get("ct_maxilla")
    followup_upper, followup_upper_facts = loaded["followup_upper"]
    followup_lower, followup_lower_facts = loaded["followup_lower"]

    _require_shared_ct_frame(ct_dentition_facts, ct_mandible_facts)
    if ct_maxilla_bundle is not None:
        _require_shared_ct_frame(
            ct_dentition_facts,
            ct_maxilla_bundle[1],
            bone_name="上颌骨",
        )

    from .registration_selection import load_priority_mask
    target_priority, target_priority_profile = load_priority_mask(
        inputs.baseline_lower, "baseline_lower", len(baseline_lower.triangles)
    )
    source_priority, source_priority_profile = load_priority_mask(
        inputs.ct_dentition, "ct_dentition", len(ct_dentition.triangles)
    )
    if target_priority is not None or source_priority is not None:
        _notify(
            progress,
            0.04,
            "已载入 T_CT 配准选区："
            f"下颌口扫.1 {int(target_priority.sum()) if target_priority is not None else 0:,} 面，"
            f"全牙列 {int(source_priority.sum()) if source_priority is not None else 0:,} 面",
        )

    _notify_stage(stage_changed, run_directory, "T_CT")
    try:
        stage1_result, stage1_diagnostics = _register_ct_with_consensus(
            baseline_lower, ct_dentition, baseline_lower_facts, ct_dentition_facts,
            ct_config, run_directory, progress,
            target_priority_faces=target_priority,
            source_priority_faces=source_priority,
        )
    except WorkflowError as exc:
        if exc.candidate is not None:
            export_stage_review(run_directory, "T_CT", inputs.baseline_lower, ct_dentition,
                                exc.candidate, accepted=False, extra_warnings=(str(exc),))
        raise
    stage1 = StageOutcome(
        "T_CT",
        "全牙列 → 下颌口扫.1",
        "下颌口扫.1",
        "全牙列",
        stage1_result,
        stage1_diagnostics,
    )
    write_json(run_directory / "stages" / "01_T_CT.json", stage1.as_dict())
    review_ct = export_stage_review(run_directory, "T_CT", inputs.baseline_lower, ct_dentition, stage1_result)
    _require_success(stage1, allow_warning=True)

    ct_dentition_t0 = transformed_mesh(ct_dentition, stage1.transformation)
    ct_mandible_t0 = transformed_mesh(ct_mandible, stage1.transformation)
    ct_maxilla_t0 = (
        transformed_mesh(ct_maxilla_bundle[0], stage1.transformation)
        if ct_maxilla_bundle is not None else None
    )
    ready_t0 = ["ct_dentition_t0", "ct_mandible_t0"]
    if ct_maxilla_t0 is not None:
        ready_t0.append("ct_maxilla_t0")
    _notify_stage(stage_changed, run_directory, "T_CT", result=stage1_result,
                  review_path=review_ct, ready_outputs=tuple(ready_t0))

    _notify_stage(stage_changed, run_directory, "T_UPPER")
    stage2_result = register_meshes(
        baseline_upper,
        followup_upper,
        baseline_upper_facts,
        followup_upper_facts,
        base_config,
        _scaled_progress(progress, offset=0.43, scale=0.24, prefix="阶段 2/3 上颌口扫.2→上颌口扫.1"),
    )
    stage2 = StageOutcome(
        "T_UPPER",
        "用上颌统一两次口扫坐标系",
        "上颌口扫.1",
        "上颌口扫.2",
        stage2_result,
    )
    write_json(run_directory / "stages" / "02_T_UPPER.json", stage2.as_dict())
    review_upper = export_stage_review(run_directory, "T_UPPER", inputs.baseline_upper, followup_upper,
                                      stage2_result, accepted=stage2_result.status == "success")
    _require_success(stage2)

    followup_upper_aligned = transformed_mesh(followup_upper, stage2.transformation)
    followup_lower_aligned = transformed_mesh(followup_lower, stage2.transformation)
    followup_lower_aligned_facts = _transformed_facts(
        followup_lower_facts, followup_lower_aligned
    )
    _notify_stage(stage_changed, run_directory, "T_UPPER", result=stage2_result,
                  review_path=review_upper, ready_outputs=("followup_upper_in_t0", "followup_lower_in_t0"))

    _notify_stage(stage_changed, run_directory, "T_DELTA")
    stage3_result = register_meshes(
        followup_lower_aligned,
        baseline_lower,
        followup_lower_aligned_facts,
        baseline_lower_facts,
        base_config,
        _scaled_progress(progress, offset=0.67, scale=0.24, prefix="阶段 3/3 下颌口扫.1→下颌口扫.2"),
    )
    stage3 = StageOutcome(
        "T_DELTA",
        "下颌口扫.1 → 下颌口扫.2（同系）",
        "下颌口扫.2（同系）",
        "下颌口扫.1",
        stage3_result,
    )
    write_json(run_directory / "stages" / "03_T_DELTA.json", stage3.as_dict())
    stage3_target = write_mesh(run_directory / "stage_views" / "T_DELTA" / "target.stl", followup_lower_aligned)
    review_delta = export_stage_review(run_directory, "T_DELTA", stage3_target, baseline_lower,
                                      stage3_result, accepted=stage3_result.status == "success")
    _require_success(stage3)
    _notify_stage(stage_changed, run_directory, "T_DELTA", result=stage3_result, review_path=review_delta)

    t_ct = stage1.transformation
    t_upper = stage2.transformation
    t_delta = stage3.transformation
    t_mandible_t0 = t_ct
    t_mandible_t1 = compose(t_delta, t_ct)

    ct_mandible_t1 = transformed_mesh(ct_mandible, t_mandible_t1)
    ct_dentition_t1 = transformed_mesh(ct_dentition, t_mandible_t1)
    baseline_lower_at_t1 = transformed_mesh(baseline_lower, t_delta)

    _notify(progress, 0.91, "保存网格、矩阵和质量报告")
    mesh_directory = run_directory / "meshes"
    matrix_directory = run_directory / "matrices"

    output_files = {
        "ct_dentition_t0": write_mesh(mesh_directory / "ct_dentition_T0.stl", ct_dentition_t0),
        "ct_dentition_t1": write_mesh(mesh_directory / "ct_dentition_T1.stl", ct_dentition_t1),
        "ct_mandible_t0": write_mesh(mesh_directory / "ct_mandible_T0.stl", ct_mandible_t0),
        "followup_upper_in_t0": write_mesh(mesh_directory / "followup_upper_in_T0.stl", followup_upper_aligned),
        "followup_lower_in_t0": write_mesh(mesh_directory / "followup_lower_in_T0.stl", followup_lower_aligned),
        "baseline_lower_at_t1": write_mesh(mesh_directory / "baseline_lower_at_T1.stl", baseline_lower_at_t1),
        "ct_mandible_t1": write_mesh(mesh_directory / "ct_mandible_T1.stl", ct_mandible_t1),
        "review_T_CT": review_ct,
        "review_T_UPPER": review_upper,
        "review_T_DELTA": review_delta,
    }
    if ct_maxilla_t0 is not None:
        output_files["ct_maxilla_t0"] = write_mesh(
            mesh_directory / "ct_maxilla_T0.stl", ct_maxilla_t0
        )

    matrix_payloads = {
        "T_CT": _matrix_payload("T_CT", t_ct, "CT 原始坐标 → 下颌口扫.1 坐标"),
        "T_UPPER": _matrix_payload("T_UPPER", t_upper, "口扫.2 坐标 → 口扫.1 坐标"),
        "T_DELTA": _matrix_payload("T_DELTA", t_delta, "下颌位姿.1 → 下颌位姿.2（统一上颌坐标系）"),
        "T_MANDIBLE_T0": _matrix_payload("T_MANDIBLE_T0", t_mandible_t0, "CT 原始颌骨 → 颌骨.1"),
        "T_MANDIBLE_T1": _matrix_payload("T_MANDIBLE_T1", t_mandible_t1, "CT 原始颌骨 → 颌骨.2"),
    }
    if ct_maxilla_t0 is not None:
        matrix_payloads["T_MAXILLA_T0"] = _matrix_payload(
            "T_MAXILLA_T0", t_ct, "CT 原始上颌骨 → 上颌骨.1；不生成时点 2 副本"
        )
    for name, payload in matrix_payloads.items():
        write_json(matrix_directory / f"{name}.json", payload)

    stages = (stage1, stage2, stage3)
    manifest = {
        "schema_version": 3,
        "workflow": "mandibular_pose_transfer",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "registration_core_version": registration_core_version,
        "runtime_threads": {"OpenMP": 1, "BLAS": 1},
        "coordinate_reference": "上颌口扫.1 坐标系；下颌位姿由各组数字咬合记录给出",
        "assumptions": [
            "CT 牙列与 CT 下颌骨共享同一原始坐标系",
            "可选上颌骨与 CT 牙列、下颌骨共享同一原始坐标系，且只应用 T_CT 定位到时点 1",
            "两次之间上颌牙冠几何未改变",
            "两次之间下颌牙相对下颌骨没有发生牙移动",
            "所有 STL 坐标单位为毫米，全部变换均为刚性变换",
            "第一阶段默认由算法识别重叠面；若存在人工选区，则作为公共配准接口的重点采样面，不裁剪模型",
        ],
        "inputs": {
            key: {
                "path": str(path),
                "sha256": sha256_file(path),
                "mesh_facts": loaded[key][1].as_dict(),
            }
            for key, path in values.items()
        },
        "algorithm_config": {
            "default_stages": asdict(base_config),
            "ct_partial_overlap_stage": asdict(ct_config),
        },
        "stages": [stage.as_dict() for stage in stages],
        "transforms": matrix_payloads,
        "outputs": {
            key: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for key, path in output_files.items()
        },
    }
    selected_profiles = {
        key: profile
        for key, profile in (
            ("baseline_lower", target_priority_profile),
            ("ct_dentition", source_priority_profile),
        )
        if profile is not None and profile.get("selected_triangle_count", 0) > 0
    }
    if selected_profiles:
        manifest["registration_selections"] = selected_profiles
    # Optional landmarks do not affect registration. Archive the exact selection
    # so the result remains measurable if the original STL is moved later.
    from .condyles import profile_path, read_profile
    selection_path = profile_path(inputs.ct_mandible)
    if selection_path.is_file():
        try:
            manifest["condyle_selection"] = read_profile(
                selection_path, manifest["inputs"]["ct_mandible"]["sha256"]
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            _notify(progress, 0.99, f"髁突选区未归档（配准不受影响）：{exc}")
    project_path = write_json(run_directory / "project.json", manifest)
    output_files["project"] = project_path

    outcome = StudyOutcome(
        run_directory=run_directory,
        stages=stages,
        t_ct=t_ct,
        t_upper=t_upper,
        t_delta=t_delta,
        t_mandible_t0=t_mandible_t0,
        t_mandible_t1=t_mandible_t1,
        output_files=output_files,
    )
    write_json(run_directory / "summary.json", outcome.as_dict())
    _notify(progress, 1.0, f"完成：{run_directory}")
    return outcome


def run_study(
    inputs: StudyInputs,
    output_root: str | Path,
    *,
    config: AlignmentConfig | None = None,
    progress: ProgressCallback | None = None,
    stage_changed: StageCallback | None = None,
) -> StudyOutcome:
    """Create an auditable run directory, then execute the complete workflow."""

    run_directory = timestamped_run_directory(output_root)
    try:
        return _run_study_in_directory(
            inputs,
            run_directory,
            config=config,
            progress=progress,
            stage_changed=stage_changed,
        )
    except Exception as exc:
        failure_path = run_directory / "failure.json"
        try:
            write_json(
                failure_path,
                {
                    "schema_version": 1,
                    "failed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "inputs": inputs.as_dict(),
                    "diagnostic_directory": str(run_directory),
                    "runtime_threads": {"OpenMP": 1, "BLAS": 1},
                },
            )
        except Exception:
            pass
        if isinstance(exc, WorkflowError):
            raise WorkflowError(f"{exc}；诊断记录：{run_directory}", run_directory=run_directory) from exc
        raise WorkflowError(
            f"运行异常：{exc}；已保留诊断记录：{run_directory}", run_directory=run_directory,
        ) from exc
