from pathlib import Path

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from mandible_registration.flow_diagram import WorkflowDiagram, STAGE_COLORS
from mandible_registration.gui import MainWindow


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])
    instance.setQuitOnLastWindowClosed(False)
    return instance


def test_only_reviewable_lines_and_their_labels_open_a_stage(app):
    flow = WorkflowDiagram()
    flow.resize(1010, 525)
    clicked = []
    chosen = []
    flow.stage_clicked.connect(clicked.append)
    flow.choose_input.connect(chosen.append)
    flow.show()
    app.processEvents()
    try:
        QTest.mouseClick(flow, Qt.MouseButton.LeftButton, pos=flow.widget_point(flow.stage_paths["T_CT"].pointAtPercent(.5)).toPoint())
        assert not clicked
        states = {"T_CT": "success", "T_UPPER": "warning", "T_DELTA": "failed"}
        flow.set_results(states, {key: Path(key) for key in states})
        for key, path in flow.stage_paths.items():
            QTest.mouseClick(flow, Qt.MouseButton.LeftButton, pos=flow.widget_point(path.pointAtPercent(.5)).toPoint())
            QTest.mouseClick(flow, Qt.MouseButton.LeftButton, pos=flow.widget_point(flow.stage_labels[key].center()).toPoint())
            assert clicked[-2:] == [key, key]
        assert len(set(STAGE_COLORS[key] for key in ("success", "warning", "failed"))) == 3
        node = next(node for node in flow.nodes if node.key == "ct_mandible")
        QTest.mouseClick(flow, Qt.MouseButton.LeftButton, pos=flow.widget_point(node.rect.center()).toPoint())
        assert chosen == ["ct_mandible"]
        flow.import_enabled = False
        QTest.mouseClick(flow, Qt.MouseButton.LeftButton, pos=flow.widget_point(node.rect.center()).toPoint())
        assert chosen == ["ct_mandible"]
    finally:
        flow.close()
        flow.deleteLater()


def test_stage_links_dispatch_the_correct_viewer_mode(app, monkeypatch):
    window = MainWindow()
    calls = []
    monkeypatch.setattr(window, "_viewer_command", lambda *args: calls.append(args))
    try:
        window._review_paths = {"T_CT": Path("results.json"), "T_UPPER": Path("project.json")}
        window._view_stage("T_CT")
        window._view_stage("T_UPPER")
        window._view_stage("T_DELTA")
        assert calls == [("--view-stage", "results.json"),
                         ("--view-stage-project", "project.json", "--stage-key", "T_UPPER")]
    finally:
        window.close()
        window.deleteLater()


def test_main_window_dispatches_both_registration_selection_roles(app, monkeypatch, tmp_path):
    window = MainWindow()
    calls = []
    lower = tmp_path / "lower.stl"
    teeth = tmp_path / "teeth.stl"
    window._paths = {"baseline_lower": lower, "ct_dentition": teeth}
    monkeypatch.setattr(
        "mandible_registration.gui.registration_profile_path",
        lambda mesh, role: tmp_path / f"{role}.json",
    )
    monkeypatch.setattr(
        window,
        "_start_selection_process",
        lambda arguments, job, message: calls.append((arguments, job, message)),
    )
    try:
        window._select_registration_region("baseline_lower")
        window._select_registration_region("ct_dentition")
        assert [call[1] for call in calls] == [
            ("registration", "baseline_lower"),
            ("registration", "ct_dentition"),
        ]
        assert calls[0][0] == [
            "--select-registration", str(lower),
            "--selection-role", "baseline_lower",
            "--registration-profile", str(tmp_path / "baseline_lower.json"),
        ]
        assert calls[1][0] == [
            "--select-registration", str(teeth),
            "--selection-role", "ct_dentition",
            "--registration-profile", str(tmp_path / "ct_dentition.json"),
        ]
    finally:
        window.close()
        window.deleteLater()


def test_svg_states_and_single_connected_workflow(app):
    import xml.etree.ElementTree as ET

    flow = WorkflowDiagram()
    original = flow.asset_path.read_bytes()
    try:
        assert len([node for node in flow.nodes if node.is_input]) == 6
        assert len({node.key for node in flow.nodes}) == len(flow.nodes)
        flow.set_inputs({"baseline_lower": Path("lower.stl")})
        flow.set_results({"T_CT": "success", "T_UPPER": "warning", "T_DELTA": "failed"}, {"T_CT": Path("results.json")})
        elements = {element.get("id"): element for element in ET.fromstring(flow.svg_bytes).iter() if element.get("id")}
        assert elements["ready-baseline_lower"].get("opacity") == "1"
        assert elements["missing-baseline_lower"].get("opacity") == "0"
        assert elements["label-baseline_lower"].get("fill") == "#15191f"
        assert elements["label-followup_lower"].get("opacity") == "0.55"
        assert elements["missing-followup_lower"].get("opacity") == "1"
        for key, status in flow.stage_states.items():
            assert elements[f"stage-{key}"].get("stroke") == STAGE_COLORS[status]
        assert flow.asset_path.read_bytes() == original  # runtime state must not overwrite editor source
    finally:
        flow.deleteLater()


def test_compare_result_node_clicks_only_when_both_bones_exist(app):
    flow = WorkflowDiagram()
    flow.resize(1200, 600)
    flow.show()
    app.processEvents()
    opened = []
    flow.compare_clicked.connect(lambda: opened.append(True))
    node = next(node for node in flow.nodes if node.key == "comparison")
    point = flow.widget_point(node.rect.center()).toPoint()
    try:
        QTest.mouseClick(flow, Qt.MouseButton.LeftButton, pos=point)
        assert not opened
        flow.set_results(outputs=["ct_mandible_t0", "ct_mandible_t1"])
        QTest.mouseClick(flow, Qt.MouseButton.LeftButton, pos=point)
        assert opened == [True]
        assert flow.logical_point(flow.widget_point(node.rect.center())) == node.rect.center()
    finally:
        flow.close()
        flow.deleteLater()


def test_condyle_button_enabled_only_after_bone_import(app):
    flow = WorkflowDiagram()
    flow.resize(1200, 600)
    flow.show()
    app.processEvents()
    clicks = []
    flow.condyle_requested.connect(lambda: clicks.append(True))
    try:
        assert not flow.condyle_button.isEnabled()
        flow.set_inputs({"ct_mandible": Path("bone.stl")})
        assert flow.condyle_button.isEnabled()
        QTest.mouseClick(flow.condyle_button, Qt.MouseButton.LeftButton)
        assert clicks == [True]
        node = next(node for node in flow.nodes if node.key == "ct_mandible")
        assert node.rect.contains(flow.logical_point(flow.condyle_button.geometry().center()))
        flow.set_inputs({})
        assert not flow.condyle_button.isEnabled()
    finally:
        flow.close()
        flow.deleteLater()


def test_registration_selection_buttons_follow_matching_inputs_and_busy_state(app):
    flow = WorkflowDiagram()
    flow.resize(1200, 600)
    flow.show()
    app.processEvents()
    selected = []
    flow.registration_selection_requested.connect(selected.append)
    try:
        assert all(not button.isEnabled() for button in flow.registration_buttons.values())
        flow.set_inputs({"baseline_lower": Path("lower.stl"), "ct_dentition": Path("teeth.stl")})
        for key, button in flow.registration_buttons.items():
            assert button.isEnabled()
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)
            node = next(node for node in flow.nodes if node.key == key)
            assert node.rect.contains(flow.logical_point(button.geometry().center()))
        assert selected == ["baseline_lower", "ct_dentition"]
        flow.set_import_enabled(False)
        assert all(not button.isEnabled() for button in flow.registration_buttons.values())
        flow.set_import_enabled(True)
        assert all(button.isEnabled() for button in flow.registration_buttons.values())
        flow.set_inputs({"baseline_lower": Path("lower.stl")})
        assert flow.registration_buttons["baseline_lower"].isEnabled()
        assert not flow.registration_buttons["ct_dentition"].isEnabled()
    finally:
        flow.close()
        flow.deleteLater()


def test_saved_selection_buttons_turn_green_and_report_existing_profile(app):
    flow = WorkflowDiagram()
    flow.set_inputs({"baseline_lower": Path("lower.stl"), "ct_mandible": Path("bone.stl")})
    try:
        flow.set_selection_statuses({"baseline_lower", "ct_mandible"})
        for button in (flow.registration_buttons["baseline_lower"], flow.condyle_button):
            assert button.text() == "已选"
            assert button.property("selectionSaved") is True
            assert "点击可重新编辑" in button.toolTip()
        flow.set_selection_statuses(set())
        assert flow.registration_buttons["baseline_lower"].text() == "选区"
        assert "选择一个" in flow.registration_buttons["baseline_lower"].toolTip()
    finally:
        flow.close()
        flow.deleteLater()


def test_inkscape_path_without_data_attribute_stays_colored_and_clickable(app, tmp_path):
    import xml.etree.ElementTree as ET
    from PySide6.QtGui import QImage, QPainter

    source = Path(__file__).resolve().parents[1] / "src/mandible_registration/assets/workflow.editable.svg"
    document = ET.parse(source).getroot()
    stage = next(element for element in document.iter() if element.get("id") == "stage-T_CT")
    stage.tag = "{http://www.w3.org/2000/svg}path"
    stage.attrib.pop("points", None)
    stage.attrib.pop("data-stage", None)
    stage.set("d", "M 571.93362,196.1413 H 971.76188")
    stage.set("style", "fill:none;stroke:#c6cdd6;stroke-width:5.36;stroke-linecap:butt")
    path = tmp_path / "editor.svg"
    ET.ElementTree(document).write(path, encoding="utf-8")
    flow = WorkflowDiagram(asset_path=path)
    opened = []
    flow.stage_clicked.connect(opened.append)
    flow.resize(1200, 600)
    flow.show()
    app.processEvents()
    try:
        assert set(flow.stage_paths) == {"T_CT", "T_UPPER", "T_DELTA"}
        flow.set_results({"T_CT": "success"}, {"T_CT": path})
        QTest.mouseClick(flow, Qt.MouseButton.LeftButton, pos=flow.widget_point(QPointF(750, 196)).toPoint())
        assert opened == ["T_CT"]
        image = QImage(1380, 560, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.white)
        painter = QPainter(image)
        flow._renderer.render(painter)
        painter.end()
        assert image.pixelColor(750, 196).name() == STAGE_COLORS["success"]
    finally:
        flow.close()
        flow.deleteLater()


def test_transformed_input_and_stage_use_rendered_hit_coordinates(app, tmp_path):
    import xml.etree.ElementTree as ET

    source = Path(__file__).resolve().parents[1] / "src/mandible_registration/assets/workflow.editable.svg"
    document = ET.parse(source).getroot()
    parent = next(element for element in document.iter() if element.get("id") == "registration-connectors")
    parent.set("transform", "translate(0,-30)")
    path = tmp_path / "translated.svg"
    ET.ElementTree(document).write(path, encoding="utf-8")
    flow = WorkflowDiagram(asset_path=path)
    chosen, opened = [], []
    flow.choose_input.connect(chosen.append)
    flow.stage_clicked.connect(opened.append)
    flow.resize(1010, 525)
    flow.show()
    app.processEvents()
    try:
        flow.set_results({"T_CT": "success"}, {"T_CT": path})
        QTest.mouseClick(flow, Qt.MouseButton.LeftButton, pos=flow.widget_point(QPointF(750, 166)).toPoint())
        assert opened == ["T_CT"]
        for node in flow.nodes:
            if node.is_input:
                QTest.mouseClick(flow, Qt.MouseButton.LeftButton, pos=flow.widget_point(node.rect.center()).toPoint())
                assert chosen[-1] == node.key
        ct = next(node for node in flow.nodes if node.key == "ct_dentition")
        assert ct.rect.center().y() == pytest.approx(196)  # group translate(0,5)
    finally:
        flow.close()
        flow.deleteLater()
