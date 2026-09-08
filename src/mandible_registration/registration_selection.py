"""Optional priority-face selection for the CT-to-lower-IOS registration stage."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from threading import current_thread, main_thread

import numpy as np
from open3d.visualization import gui

from .core_bridge import ensure_registration_core
from .exporter import sha256_file, write_json
from .registration_profiles import ROLE_LABELS, profile_path

ensure_registration_core()

from auto_alignment.integration.review import configure_open3d_font  # noqa: E402
from auto_alignment.integration.selection import (  # noqa: E402
    MultiRegionSelectionViewer,
    SelectionRegionSpec,
    decode_face_ranges,
    encode_face_ranges,
)


REGION_SPECS = (
    SelectionRegionSpec("priority", "配准区", (0.10, 0.72, 0.36)),
)


def build_profile(mesh_path, mesh_sha256, triangle_count, role, masks):
    if role not in ROLE_LABELS:
        raise ValueError("未知配准选区模型")
    combined = np.asarray(masks["priority"])
    if combined.dtype != np.bool_ or combined.shape != (triangle_count,):
        raise ValueError("配准选区面片数与模型不一致")
    return {
        "schema_version": 1,
        "kind": "registration_priority_faces",
        "role": role,
        "role_label": ROLE_LABELS[role],
        "mesh_path": str(Path(mesh_path).resolve()),
        "mesh_sha256": mesh_sha256,
        "mesh_triangle_count": triangle_count,
        "selected_ranges": encode_face_ranges(np.flatnonzero(combined)),
        "selected_triangle_count": int(combined.sum()),
        "sampling_policy": "selected faces are priority input to the public registration API",
        "regions": {
            "priority": {
                "label": "配准区",
                "selected_ranges": encode_face_ranges(np.flatnonzero(combined)),
                "triangle_count": int(combined.sum()),
            }
        },
        "updated_at": datetime.now().astimezone().isoformat(),
    }


def read_profile(path, expected_sha256, expected_triangle_count, expected_role):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("kind") != "registration_priority_faces":
        raise ValueError("配准选区文件格式不受支持")
    if data.get("role") != expected_role:
        raise ValueError("配准选区用途与当前模型不一致")
    if data.get("mesh_sha256") != expected_sha256:
        raise ValueError("配准选区不属于当前 STL，请重新选区")
    if data.get("mesh_triangle_count") != expected_triangle_count:
        raise ValueError("配准选区的基准面数与当前 STL 不一致")
    mask = decode_face_ranges(data.get("selected_ranges"), expected_triangle_count)
    if int(mask.sum()) != data.get("selected_triangle_count"):
        raise ValueError("配准选区面片计数不一致")
    return data, mask


def load_priority_mask(mesh_path, role, triangle_count):
    path = profile_path(mesh_path, role)
    if not path.is_file():
        return None, None
    digest = sha256_file(mesh_path)
    profile, mask = read_profile(path, digest, triangle_count, role)
    return (mask if mask.any() else None), profile


@dataclass(frozen=True)
class RegistrationSelectionContext:
    mesh_path: Path
    role: str
    selection_path: Path
    state_path: Path
    mesh_sha256: str
    triangle_count: int
    initial_masks: dict[str, np.ndarray]
    mesh: object

    @classmethod
    def load(cls, mesh_path, role, selection_path=None):
        if role not in ROLE_LABELS:
            raise ValueError("未知配准选区模型")
        mesh_path = Path(mesh_path).resolve(strict=True)
        selection_path = Path(selection_path or profile_path(mesh_path, role)).resolve()
        state_path = selection_path.with_name(selection_path.stem + ".regions.v1.json")
        signature = (mesh_path.stat().st_size, mesh_path.stat().st_mtime_ns)
        digest = sha256_file(mesh_path)
        from .scene_data import load_mesh
        mesh, _ = load_mesh(mesh_path)
        if (mesh_path.stat().st_size, mesh_path.stat().st_mtime_ns) != signature:
            raise ValueError("读取期间模型文件发生变化")
        triangle_count = len(mesh.triangles)
        initial_masks = {"priority": np.zeros(triangle_count, dtype=bool)}
        if selection_path.is_file():
            _data, combined = read_profile(selection_path, digest, triangle_count, role)
            # Schema v1 profiles created with A/B already contain their union
            # in selected_ranges, so they migrate losslessly to one region.
            initial_masks = {"priority": combined}
        return cls(
            mesh_path, role, selection_path, state_path, digest, triangle_count,
            initial_masks, mesh,
        )

    def profile_from_snapshot(self, snapshot):
        if snapshot.mesh_sha256 != self.mesh_sha256 or snapshot.triangle_count != self.triangle_count:
            raise ValueError("快照不属于当前配准基准网格")
        return build_profile(
            self.mesh_path,
            self.mesh_sha256,
            self.triangle_count,
            self.role,
            snapshot.masks,
        )

    def persist_profile(self, snapshot):
        write_json(self.selection_path, self.profile_from_snapshot(snapshot))

    def persist_complete_profile(self, snapshot):
        profile = self.profile_from_snapshot(snapshot)
        if profile["selected_triangle_count"] <= 0:
            raise ValueError("请至少选择一组用于配准的面片后再保存")
        write_json(self.selection_path, profile)

    def summary(self, snapshot):
        profile = self.profile_from_snapshot(snapshot)
        return "\n".join([
            f"模型：{ROLE_LABELS[self.role]}",
            f"配准区：{profile['selected_triangle_count']:,} 面",
            "所选面片作为 T_CT 重点采样面，不裁剪模型。",
        ])


class RegistrationSelectionViewer(MultiRegionSelectionViewer):
    def __init__(self, mesh_path, role, selection_path=None):
        self.context = RegistrationSelectionContext.load(mesh_path, role, selection_path)
        super().__init__(
            self.context.mesh_path,
            self.context.state_path,
            REGION_SPECS,
            initial_masks=self.context.initial_masks,
            allow_overlap=False,
            on_change=None,
            on_save=self.context.persist_complete_profile,
            summary_provider=self.context.summary,
            preloaded_mesh=self.context.mesh,
            preloaded_mesh_sha256=self.context.mesh_sha256,
        )

    def _build_panel(self):
        super()._build_panel()
        children = self.panel.get_children()
        try:
            selector_index = children.index(self.region_selector)
        except ValueError:
            selector_index = -1
        if selector_index > 0:
            children[selector_index - 1].visible = False
        self.region_selector.visible = False


def run_registration_selection(mesh_path, role, selection_path=None):
    if current_thread() is not main_thread():
        raise RuntimeError("配准选区查看器必须在子进程主线程启动")
    app = gui.Application.instance
    app.initialize()
    configure_open3d_font(
        app,
        extra_text=str(mesh_path) + ROLE_LABELS[role] + "配准区重点采样面",
    )
    viewer = RegistrationSelectionViewer(mesh_path, role, selection_path)
    app.run()
    return viewer.get_snapshot()
