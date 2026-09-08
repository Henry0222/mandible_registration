from __future__ import annotations

from pathlib import Path

import open3d as o3d

from .core_bridge import ensure_registration_core


ensure_registration_core()
from auto_alignment.integration.core import load_mesh  # noqa: E402


def view_mesh(path: str | Path) -> None:
    mesh, _ = load_mesh(path)
    mesh.paint_uniform_color((0.72, 0.78, 0.86))
    o3d.visualization.draw_geometries(
        [mesh],
        window_name=Path(path).name,
        mesh_show_back_face=True,
    )


def _view_pair(
    first: str | Path,
    second: str | Path,
    *,
    first_color: tuple[float, float, float],
    second_color: tuple[float, float, float],
    title: str,
) -> None:
    first_mesh, _ = load_mesh(first)
    second_mesh, _ = load_mesh(second)
    first_mesh.paint_uniform_color(first_color)
    second_mesh.paint_uniform_color(second_color)
    o3d.visualization.draw_geometries(
        [first_mesh, second_mesh],
        window_name=title,
        mesh_show_back_face=True,
    )


def view_pair(first: str | Path, second: str | Path) -> None:
    _view_pair(
        first,
        second,
        first_color=(0.20, 0.55, 0.90),
        second_color=(0.95, 0.52, 0.16),
        title="下颌骨位姿对比：蓝色 T0 / 橙色 T1",
    )


def view_ct_fit(ct_dentition: str | Path, baseline_lower: str | Path) -> None:
    _view_pair(
        ct_dentition,
        baseline_lower,
        first_color=(0.72, 0.75, 0.80),
        second_color=(0.10, 0.75, 0.32),
        title="T_CT 配准检查：灰色全牙列 / 绿色下颌口扫.1",
    )
