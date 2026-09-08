"""Adapt the public multi-region selector to mandibular condyle profiles."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import current_thread, main_thread

import numpy as np
from open3d.visualization import gui

from .condyles import build_profile, read_profile
from .core_bridge import ensure_registration_core
from .exporter import sha256_file, write_json

ensure_registration_core()

from auto_alignment.integration.review import configure_open3d_font  # noqa: E402
from auto_alignment.integration.selection import (  # noqa: E402
    MultiRegionSelectionViewer,
    SelectionRegionSpec,
    decode_face_ranges,
)


REGION_SPECS = (
    SelectionRegionSpec("left", "左侧髁突", (1.0, 0.2, 0.1)),
    SelectionRegionSpec("right", "右侧髁突", (0.1, 0.3, 1.0)),
)


@dataclass(frozen=True)
class CondyleSelectionContext:
    mesh_path: Path
    selection_path: Path
    state_path: Path
    mesh: object
    mesh_sha256: str
    triangle_count: int
    initial_masks: dict[str, np.ndarray] | None

    @classmethod
    def load(cls, mesh_path, selection_path):
        mesh_path = Path(mesh_path).resolve(strict=True)
        selection_path = Path(selection_path).resolve()
        state_path = selection_path.with_name(selection_path.stem + ".regions.v1.json")
        signature = (mesh_path.stat().st_size, mesh_path.stat().st_mtime_ns)
        digest = sha256_file(mesh_path)
        from .scene_data import load_mesh
        mesh, _ = load_mesh(mesh_path)
        if (mesh_path.stat().st_size, mesh_path.stat().st_mtime_ns) != signature:
            raise ValueError("读取期间模型文件发生变化")
        triangle_count = len(mesh.triangles)
        initial_masks = None
        if selection_path.is_file():
            saved = read_profile(selection_path, digest)
            if saved["mesh_triangle_count"] != triangle_count:
                raise ValueError("髁突选区的基准面数不匹配")
            initial_masks = {}
            for key in ("left", "right"):
                region = saved["regions"].get(key)
                initial_masks[key] = (
                    decode_face_ranges(region["selected_ranges"], triangle_count)
                    if region is not None
                    else np.zeros(triangle_count, dtype=bool)
                )
        return cls(
            mesh_path,
            selection_path,
            state_path,
            mesh,
            digest,
            triangle_count,
            initial_masks,
        )

    def profile_from_snapshot(self, snapshot):
        if (
            snapshot.mesh_sha256 != self.mesh_sha256
            or snapshot.triangle_count != self.triangle_count
        ):
            raise ValueError("快照不属于缓存的 CT 基准网格")
        return build_profile(
            self.mesh,
            self.mesh_path,
            self.mesh_sha256,
            snapshot.masks,
        )

    def persist_profile(self, snapshot):
        write_json(self.selection_path, self.profile_from_snapshot(snapshot))

    def summary(self, snapshot):
        profile = self.profile_from_snapshot(snapshot)
        active = "左侧髁突" if snapshot.active_region == "left" else "右侧髁突"
        rows = [f"当前编辑：{active}"]
        for key, label in (("left", "左侧髁突"), ("right", "右侧髁突")):
            region = profile["regions"].get(key)
            if region is None:
                rows.append(f"{label}：未选择（0 面）")
            else:
                x, y, z = region["center_ct_mm"]
                rows.append(
                    f"{label}：{region['triangle_count']:,} 面；"
                    f"CT 中心 mm：{x:.3f}, {y:.3f}, {z:.3f}"
                )
        return "\n".join(rows)


def selected_condyle_sides(snapshot) -> tuple[str, ...]:
    """Return selected sides in display order; kept GUI-independent for tests."""
    return tuple(key for key in ("left", "right") if np.asarray(snapshot.masks[key]).any())


def needs_single_side_confirmation(snapshot) -> bool:
    return len(selected_condyle_sides(snapshot)) == 1


class CondyleSelectionViewer(MultiRegionSelectionViewer):
    """Compatibility name backed only by the public multi-region API."""

    def __init__(self, mesh_path, selection_path):
        self.context = CondyleSelectionContext.load(mesh_path, selection_path)
        self._single_side_confirmed = False
        self._single_side_dialog = None
        super().__init__(
            self.context.mesh_path,
            self.context.state_path,
            REGION_SPECS,
            initial_masks=self.context.initial_masks,
            allow_overlap=False,
            on_change=None,
            on_save=self.context.persist_profile,
            summary_provider=self.context.summary,
            preloaded_mesh=self.context.mesh,
            preloaded_mesh_sha256=self.context.mesh_sha256,
        )

    def _build_panel(self):
        """Replace the generic region combobox with explicit left/right buttons."""
        super()._build_panel()
        children = self.panel.get_children()
        try:
            selector_index = children.index(self.region_selector)
        except ValueError:
            selector_index = -1
        if selector_index > 0:
            children[selector_index - 1].visible = False
        self.region_selector.visible = False

        em = self.window.theme.font_size
        self.panel.add_child(gui.Label("当前编辑侧（按下按钮切换）"))
        row = gui.Horiz(0.35 * em)
        self.side_buttons = {}
        for spec in REGION_SPECS:
            button = gui.Button(spec.label)
            button.toggleable = True
            button.tooltip = f"切换到{spec.label}选区"
            button.set_on_clicked(lambda key=spec.key: self._select_side(key))
            self.side_buttons[spec.key] = button
            row.add_child(button)
        self.panel.add_child(row)
        self._refresh_side_buttons()

    def _select_side(self, key):
        try:
            self.set_active_region(key)
            self.status_label.text = f"当前编辑：{dict((s.key, s.label) for s in REGION_SPECS)[key]}"
        except Exception as exc:
            self.status_label.text = f"切换失败：{exc}"

    def _refresh_side_buttons(self):
        if not hasattr(self, "side_buttons"):
            return
        active = self.get_snapshot().active_region
        for key, button in self.side_buttons.items():
            button.is_on = key == active

    def _refresh_counts(self):
        super()._refresh_counts()
        self._refresh_side_buttons()

    def save(self):
        result = super().save()
        self.status_label.text = "髁突选区已保存"
        return result

    def _show_single_side_warning(self, snapshot):
        if self._single_side_dialog is not None:
            return
        selected = selected_condyle_sides(snapshot)[0]
        selected_label = "左侧髁突" if selected == "left" else "右侧髁突"
        em = self.window.theme.font_size
        dialog = gui.Dialog("仅选择了单侧髁突")
        layout = gui.Vert(0.6 * em, gui.Margins(em, em, em, em))
        layout.add_child(gui.Label(f"目前只选择了{selected_label}。"))
        layout.add_child(gui.Label("仍可保存；未选择侧不会生成髁突位移数据。"))
        actions = gui.Horiz(0.5 * em)
        cancel = gui.Button("返回补选")
        proceed = gui.Button("仅保存单侧")

        def close_dialog():
            self.window.close_dialog()
            self._single_side_dialog = None

        def save_one_side():
            close_dialog()
            self._single_side_confirmed = True
            self._save_and_close()

        cancel.set_on_clicked(close_dialog)
        proceed.set_on_clicked(save_one_side)
        actions.add_child(cancel)
        actions.add_child(proceed)
        layout.add_child(actions)
        dialog.add_child(layout)
        self._single_side_dialog = dialog
        self.window.show_dialog(dialog)

    def _save_and_close(self):
        if self._closing or self._selection_busy:
            return super()._save_and_close()
        snapshot = self.get_snapshot()
        if needs_single_side_confirmation(snapshot) and not self._single_side_confirmed:
            self._show_single_side_warning(snapshot)
            return
        return super()._save_and_close()


def run_condyle_selection(mesh_path, selection_path):
    if current_thread() is not main_thread():
        raise RuntimeError("髁突查看器必须在子进程主线程启动")
    app = gui.Application.instance
    app.initialize()
    configure_open3d_font(
        app,
        extra_text=str(mesh_path) + "左侧右侧髁突未选择中心 CT mm",
    )
    viewer = CondyleSelectionViewer(mesh_path, selection_path)
    app.run()
    return viewer.get_snapshot()
