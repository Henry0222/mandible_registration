"""Render the actual Qt workflow UI without opening native windows or models."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mandible_registration.gui import MainWindow
from mandible_registration.workflow import StageUpdate
from PySide6.QtCore import QThread, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    app = QApplication(["flow-preview"])
    window = MainWindow()
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.show()
    app.processEvents()
    window.grab().save(str(args.output / "empty.png"))
    window._load_project(args.project)
    app.processEvents()
    window.grab().save(str(args.output / "complete.png"))
    image = QImage(1380, 560, QImage.Format.Format_ARGB32)
    image.fill(QColor("#f4f7fb"))
    painter = QPainter(image)
    window.flow._renderer.render(painter)
    painter.end()
    image.save(str(args.output / "workflow.preview.png"))
    print("Interactive stages:", sorted(window.flow.stage_paths), flush=True)
    # Replay finished quality records as live transitions; no registration or
    # source/result files are modified. This catches regressions hidden by the
    # final-project loader, which restores all statuses at once.
    reviews = dict(window._review_paths)
    states = dict(window.flow.stage_states)
    window._apply_input_paths(window._paths)
    preview_thread = QThread(window)
    window._thread = preview_thread
    for key, outputs in (("T_CT", ("ct_mandible_t0",)),
                         ("T_UPPER", ("followup_lower_in_t0",))):
        window._on_stage_changed(StageUpdate(key, "running", args.project.parent))
        window._on_stage_changed(StageUpdate(key, states[key], args.project.parent, reviews[key], ready_outputs=outputs))
    window._on_stage_changed(StageUpdate("T_DELTA", "running", args.project.parent))
    window._on_progress(75, "阶段 3/3 下颌口扫.1→下颌口扫.2：正在生成多个全局配准候选…")
    app.processEvents()
    window.grab().save(str(args.output / "stage3_running.png"))
    window._thread = None
    preview_thread.deleteLater()
    window._apply_input_paths({key: path for key, path in window._paths.items() if key in ("baseline_lower", "ct_mandible", "followup_upper")})
    app.processEvents()
    assert not window.run_button.isEnabled()
    window.grab().save(str(args.output / "partial.png"))
    window.close()


if __name__ == "__main__":
    main()
