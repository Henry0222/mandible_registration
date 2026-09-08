"""Regression coverage for live stage transitions, not just final project loading."""
import json
from pathlib import Path
from threading import Event
from time import monotonic

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from mandible_registration import gui
from mandible_registration.models import INPUT_SPECS
from mandible_registration.workflow import StageUpdate, WorkflowError


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])
    instance.setQuitOnLastWindowClosed(False)
    return instance


def wait_until(predicate):
    deadline = monotonic() + 5
    while not predicate() and monotonic() < deadline:
        QTest.qWait(10)
    assert predicate()


@pytest.mark.parametrize("ct_status", ["success", "warning"])
def test_worker_preserves_finished_stages_while_third_runs_and_fails(app, tmp_path, monkeypatch, ct_status):
    resume = Event()
    viewer_calls = []
    paths = {}
    for spec in INPUT_SPECS:
        paths[spec.key] = tmp_path / f"{spec.key}.stl"
        paths[spec.key].touch()
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    def study(inputs, output_root, *, progress, stage_changed):
        for key, status, outputs in (
            ("T_CT", ct_status, ("ct_mandible_t0",)),
            ("T_UPPER", "success", ("followup_lower_in_t0",)),
        ):
            stage_changed(StageUpdate(key, "running", run_directory))
            review = run_directory / "stage_views" / key / "results.json"
            review.parent.mkdir(parents=True)
            review.write_text(json.dumps({"registration": {"status": status, "confidence": "高"}}), encoding="utf-8")
            stage_changed(StageUpdate(key, status, run_directory, review, "高", outputs))
        stage_changed(StageUpdate("T_DELTA", "running", run_directory))
        progress(.75, "阶段 3/3 下颌口扫.1→下颌口扫.2：正在生成多个全局配准候选…")
        resume.wait(5)
        # An unexpected algorithm error may have no stage review to read.
        raise WorkflowError("测试阶段中断", run_directory=run_directory)

    monkeypatch.setattr(gui, "run_study", study)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    window = gui.MainWindow()
    monkeypatch.setattr(window, "_viewer_command", lambda *args: viewer_calls.append(args))
    window._apply_input_paths(paths)
    window.output_edit.setText(str(tmp_path))
    window.show()
    app.processEvents()
    try:
        window._start()
        wait_until(lambda: window.progress_bar.value() == 75)
        expected = {"T_CT": ct_status, "T_UPPER": "success", "T_DELTA": "running"}
        assert window.flow.stage_states == expected
        assert {"ct_mandible_t0", "followup_lower_in_t0"} <= window.flow.outputs
        assert not window.view_button.isEnabled()
        assert not window.flow.node_ready(next(node for node in window.flow.nodes if node.key == "comparison"))
        assert not window.sequence_button.isEnabled()
        for key in ("T_CT", "T_UPPER"):
            point = window.flow.widget_point(window.flow.stage_labels[key].center()).toPoint()
            QTest.mouseClick(window.flow, Qt.MouseButton.LeftButton, pos=point)
            assert viewer_calls[-1] == ("--view-stage", str(window._review_paths[key]))
        for value in range(76, 80):
            window._on_progress(value, "阶段 3/3：继续优化")
            assert window.flow.stage_states == expected
            assert set(window.flow.review_paths) == {"T_CT", "T_UPPER"}
        resume.set()
        wait_until(lambda: window._thread is None)
        assert window.flow.stage_states == {**expected, "T_DELTA": "failed"}
        assert {"ct_mandible_t0", "followup_lower_in_t0"} <= window.flow.outputs
        assert set(window.flow.review_paths) == {"T_CT", "T_UPPER"}
        assert "未生成可查看的候选" in window.flow.stage_tooltips["T_DELTA"]
        # A changed input must still invalidate everything from the old run.
        window._apply_input_paths(paths)
        assert not window.flow.stage_states
        assert not window.flow.outputs
        assert not window.flow.review_paths
    finally:
        resume.set()
        wait_until(lambda: window._thread is None)
        window.close()
        window.deleteLater()


def test_failure_retains_review_for_rejected_candidate(app, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    window = gui.MainWindow()
    review = tmp_path / "stage_views" / "T_UPPER" / "results.json"
    review.parent.mkdir(parents=True)
    review.write_text(json.dumps({"registration": {"status": "failed", "confidence": "低"}, "review_only": True}), encoding="utf-8")
    try:
        window._on_stage_changed(StageUpdate("T_CT", "success", tmp_path, ready_outputs=("ct_mandible_t0",)))
        window._on_stage_changed(StageUpdate("T_UPPER", "running", tmp_path))
        window._on_failed("上颌未通过", "测试诊断", tmp_path)
        assert window.flow.stage_states == {"T_CT": "success", "T_UPPER": "failed"}
        assert window.flow.outputs == {"ct_mandible_t0"}
        assert window._review_paths["T_UPPER"] == review
        assert "T_DELTA" not in window.flow.stage_states
    finally:
        window.close()
        window.deleteLater()
