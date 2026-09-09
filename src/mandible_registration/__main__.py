from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from .models import StudyInputs


_WINDOWED_STANDARD_STREAMS: list[object] = []


def ensure_standard_streams() -> None:
    """Give native libraries writable streams in a windowed executable.

    PyInstaller sets both streams to ``None`` for ``console=False`` builds,
    while Open3D may still emit registration diagnostics through them.
    """
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is not None:
            continue
        stream = open(os.devnull, "w", encoding="utf-8", buffering=1)
        setattr(sys, name, stream)
        _WINDOWED_STANDARD_STREAMS.append(stream)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="下颌位姿配准与 CT 下颌骨变换传递")
    parser.add_argument("--dataset-dir", type=Path, help="按约定中文文件名读取六个必需 STL，并自动读取可选上颌骨 STL")
    parser.add_argument("--output-dir", type=Path, help="结果根目录")
    parser.add_argument("--view-mesh", type=Path, help="单独查看一个 STL")
    parser.add_argument("--view-stage", type=Path, help="使用通用项目查看器打开阶段 results.json")
    parser.add_argument("--view-stage-project", type=Path, help="查看历史项目的某个配准阶段")
    parser.add_argument("--stage-key", choices=("T_CT", "T_UPPER", "T_DELTA"), default="T_CT")
    parser.add_argument("--view-project", type=Path, help="多模型对比及测量：选择结果 project.json")
    parser.add_argument("--select-condyles", type=Path, help="在原始颌骨上选择左右髁突面片")
    parser.add_argument("--condyle-profile", type=Path, help="髁突选区记录 JSON")
    parser.add_argument("--select-registration", type=Path, help="为 T_CT 选择配准重点面片")
    parser.add_argument("--selection-role", choices=("baseline_lower", "ct_dentition"))
    parser.add_argument("--registration-profile", type=Path, help="配准重点选区记录 JSON")
    parser.add_argument("--view-preset", choices=("bones", "ct"), default="bones")
    parser.add_argument("--view-pair", nargs=2, type=Path, metavar=("T0", "T1"), help="叠加查看两个 STL")
    parser.add_argument(
        "--view-ct-fit",
        nargs=2,
        type=Path,
        metavar=("CT_DENTITION", "LOWER_IOS"),
        help="检查全牙列与下颌口扫.1 的重叠",
    )
    return parser


def main() -> int:
    ensure_standard_streams()
    args = build_parser().parse_args()
    if args.select_condyles:
        from .condyles import profile_path
        from .condyle_selection import run_condyle_selection
        run_condyle_selection(args.select_condyles, args.condyle_profile or profile_path(args.select_condyles))
        return 0
    if args.select_registration:
        if args.selection_role is None:
            raise SystemExit("--select-registration 需要同时指定 --selection-role")
        from .registration_selection import profile_path, run_registration_selection
        run_registration_selection(
            args.select_registration,
            args.selection_role,
            args.registration_profile or profile_path(args.select_registration, args.selection_role),
        )
        return 0
    if args.view_stage_project:
        from .stage_review import prepare_project_stage, show_stage_review

        show_stage_review(prepare_project_stage(args.view_stage_project, args.stage_key))
        return 0
    if args.view_stage:
        from .stage_review import show_stage_review

        show_stage_review(args.view_stage)
        return 0
    if args.view_project:
        from .scene_viewer import view_project

        return view_project(args.view_project, preset=args.view_preset)
    if args.view_mesh:
        from .viewer import view_mesh

        view_mesh(args.view_mesh)
        return 0
    if args.view_pair:
        from .viewer import view_pair

        view_pair(*args.view_pair)
        return 0
    if args.view_ct_fit:
        from .viewer import view_ct_fit

        view_ct_fit(*args.view_ct_fit)
        return 0
    if args.dataset_dir:
        from .workflow import WorkflowError, run_study

        inputs = StudyInputs.from_dataset_directory(args.dataset_dir)
        output_dir = args.output_dir or Path.cwd() / "outputs"

        def progress(fraction: float, message: str) -> None:
            print(f"[{fraction:6.1%}] {message}", flush=True)

        try:
            outcome = run_study(inputs, output_dir, progress=progress)
        except WorkflowError as exc:
            print(f"错误：{exc}", file=sys.stderr)
            return 1
        print(f"完成：{outcome.run_directory}")
        for stage in outcome.stages:
            quality = stage.result.quality
            overlap = quality.directed_overlap_ratio if quality is not None else float("nan")
            p90 = quality.residual_p90_mm if quality is not None else float("nan")
            print(
                f"{stage.key}: {stage.result.status}, {stage.result.confidence}, "
                f"定向重叠率={overlap:.3f}, P90={p90:.4f} mm"
            )
        return 0
    if args.output_dir:
        raise SystemExit("--output-dir 需与 --dataset-dir 一起使用")

    from .gui import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
