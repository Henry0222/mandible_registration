from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QThread, QUrl, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent, QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from .drop_import import AssignStlDialog, DirectoryDropEdit, validate_assignments
from .flow_diagram import WorkflowDiagram
from .input_dialog import SequentialStlDialog
from .models import INPUT_SPECS, REQUIRED_INPUT_KEYS, StudyInputs
from .registration_profiles import (
    ROLE_LABELS,
    condyle_profile_path,
    profile_path as registration_profile_path,
    saved_selection_keys,
)
from .theme import apply_light_theme
from . import __version__

if TYPE_CHECKING:
    from .workflow import StageUpdate, StudyOutcome

# Kept as an injectable test/embedding hook without importing the heavy
# registration stack during main-window startup.
run_study = None


def _read_stage_reviews(path):
    from .stage_review import read_stage_reviews
    return read_stage_reviews(path)


def _application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _child_process_command(arguments, *, frozen=None):
    is_frozen = getattr(sys, "frozen", False) if frozen is None else bool(frozen)
    prefix = [] if is_frozen else ["-m", "mandible_registration"]
    return sys.executable, [*prefix, *arguments]


class WorkflowWorker(QObject):
    progress = Signal(int, str)
    stage_changed = Signal(object)
    completed = Signal(object)
    failed = Signal(str, str, object)

    def __init__(self, inputs: StudyInputs, output_root: Path):
        super().__init__()
        self.inputs, self.output_root = inputs, output_root

    @Slot()
    def run(self):
        try:
            study_runner = run_study
            if study_runner is None:
                from .workflow import run_study as study_runner
            outcome = study_runner(
                self.inputs, self.output_root,
                progress=lambda fraction, message: self.progress.emit(round(fraction * 100), message),
                stage_changed=self.stage_changed.emit,
            )
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc(), getattr(exc, "run_directory", None))
            return
        self.completed.emit(outcome)


class MainWindow(QMainWindow):
    def __init__(self):
        apply_light_theme()
        super().__init__()
        self.setWindowTitle(f"下颌位姿配准 v{__version__}")
        self.resize(1250, 760)
        self.setMinimumSize(1050, 680)
        self._paths: dict[str, Path] = {}
        self._result: StudyOutcome | None = None
        self._thread = None
        self._worker = None
        self._last_run_directory: Path | None = None
        self._project_path: Path | None = None
        self._review_paths: dict[str, Path] = {}
        self._selection_process = None
        self._selection_job = None
        self._measurement_windows = {}
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(12)
        toolbar = QHBoxLayout()
        title = QLabel("下颌位姿配准")
        title.setStyleSheet("font-size: 23px; font-weight: 700; color: #12345b;")
        toolbar.addWidget(title)
        toolbar.addStretch(1)
        self.input_count = QLabel()
        toolbar.addWidget(self.input_count)
        self.sequence_button = QPushButton("导入 / 补充 STL")
        self.sequence_button.clicked.connect(self._choose_inputs)
        toolbar.addWidget(self.sequence_button)
        layout.addLayout(toolbar)

        self.flow = WorkflowDiagram()
        self.flow.choose_input.connect(self._choose_input_key)
        self.flow.clear_input.connect(self._clear_input)
        self.flow.files_dropped.connect(self._drop_flow_inputs)
        self.flow.stage_clicked.connect(self._view_stage)
        self.flow.compare_clicked.connect(self._view_result)
        self.flow.condyle_requested.connect(self._select_condyles)
        self.flow.registration_selection_requested.connect(self._select_registration_region)
        layout.addWidget(self.flow, 1)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("结果位置"))
        self.output_edit = DirectoryDropEdit(str(_application_root() / "outputs"))
        self.output_edit.setAcceptDrops(True)
        self.output_edit.setToolTip("可以直接拖入文件夹；拖入文件时使用其所在目录。")
        self.output_button = QPushButton("选择目录")
        self.output_button.clicked.connect(self._choose_output_directory)
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(self.output_button)
        layout.addLayout(output_row)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("开始配准")
        self.run_button.setStyleSheet("QPushButton:enabled {background:#1769aa; color:white; font-weight:600; padding:9px 24px;} QPushButton:disabled {background:#e5eaf1; color:#65758b; padding:9px 24px;}")
        self.run_button.clicked.connect(self._start)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.progress_bar, 1)
        layout.addLayout(run_row)

        actions = QHBoxLayout()
        self.details_button = QPushButton("运行详情")
        self.details_button.setCheckable(True)
        actions.addWidget(self.details_button)
        self.open_button = QPushButton("打开结果目录")
        self.open_button.clicked.connect(self._open_result_directory)
        actions.addWidget(self.open_button)
        self.existing_view_button = QPushButton("打开已有结果")
        self.existing_view_button.clicked.connect(self._view_existing)
        actions.addWidget(self.existing_view_button)
        actions.addStretch(1)
        self.view_button = QPushButton("颌骨对比 / 测量")
        self.view_button.clicked.connect(self._view_result)
        actions.addWidget(self.view_button)
        layout.addLayout(actions)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        self.log.setMaximumHeight(110)
        self.log.hide()
        self.details_button.toggled.connect(self.log.setVisible)
        layout.addWidget(self.log)
        self._refresh_sequence()

    def _refresh_sequence(self):
        idle = self._thread is None and self._selection_process is None
        ready_count = len(REQUIRED_INPUT_KEYS & self._paths.keys())
        all_ready = REQUIRED_INPUT_KEYS <= self._paths.keys()
        optional_text = "上颌骨已导入" if "ct_maxilla" in self._paths else "上颌骨可选"
        self.input_count.setText(f"配准输入 {ready_count} / {len(REQUIRED_INPUT_KEYS)} · {optional_text}")
        self.run_button.setEnabled(all_ready and idle)
        self.sequence_button.setEnabled(idle)
        self.flow.set_import_enabled(idle)
        self.output_button.setEnabled(idle)
        self.output_edit.setEnabled(idle)
        self.existing_view_button.setEnabled(idle)
        self.view_button.setEnabled(self._project_path is not None)
        self.open_button.setEnabled(self._last_run_directory is not None)
        if not all_ready and idle:
            self.progress_bar.setFormat(
                f"配准输入 {ready_count}/{len(REQUIRED_INPUT_KEYS)}，补齐后可开始配准"
            )

    def _choose_input_key(self, key):
        self._choose_input(next(index for index, spec in enumerate(INPUT_SPECS) if spec.key == key))

    def _choose_input(self, row):
        if self._thread is not None:
            return
        spec = INPUT_SPECS[row]
        initial = str(self._paths[spec.key].parent) if spec.key in self._paths else str(next(reversed(self._paths.values())).parent) if self._paths else ""
        filename, _ = QFileDialog.getOpenFileName(self, f"选择 {spec.title}", initial, "STL 三角网格 (*.stl *.STL)")
        if filename:
            self._apply_input_paths({**self._paths, spec.key: Path(filename).resolve()})

    def _clear_input(self, key):
        if self._thread is None:
            self._apply_input_paths({name: path for name, path in self._paths.items() if name != key})

    def _apply_input_paths(self, paths):
        try:
            paths = validate_assignments(paths)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "文件校验未通过", str(exc))
            return
        self._paths = dict(paths)
        self._result = None
        self._project_path = None
        self._last_run_directory = None
        self._review_paths = {}
        self.flow.set_inputs(paths)
        self.flow.set_selection_statuses(saved_selection_keys(paths))
        self.flow.set_results()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(
            "六项配准输入已齐；上颌骨已导入" if "ct_maxilla" in paths
            else "六项配准输入已齐；上颌骨可稍后补充"
        )
        self._refresh_sequence()

    def _drop_flow_inputs(self, key, paths):
        row = next((index for index, spec in enumerate(INPUT_SPECS) if spec.key == key), -1)
        self._drop_inputs(row, paths)

    def _drop_inputs(self, row, paths):
        if self._thread is not None:
            return
        if len(paths) == 1 and 0 <= row < len(INPUT_SPECS):
            self._apply_input_paths({**self._paths, INPUT_SPECS[row].key: paths[0]})
            return
        dialog = AssignStlDialog(paths, self._paths, self)
        try:
            if dialog.exec() == dialog.DialogCode.Accepted and dialog.paths is not None:
                self._apply_input_paths(dialog.paths)
        finally:
            dialog.deleteLater()

    def _choose_inputs(self):
        if self._thread is not None:
            return
        initial = str(next(reversed(self._paths.values())).parent) if self._paths else ""
        dialog = SequentialStlDialog(self, directory=initial, initial_paths=self._paths)
        try:
            if dialog.exec() == QFileDialog.DialogCode.Accepted and dialog.paths is not None:
                self._apply_input_paths(dialog.paths)
        finally:
            dialog.deleteLater()

    # Compatibility for older callers and UI tests.
    def _choose_six_inputs(self):
        self._choose_inputs()

    def _choose_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "选择结果根目录", self.output_edit.text().strip())
        if directory:
            self.output_edit.setText(directory)

    def _start(self):
        if self._thread is not None or self._selection_process is not None:
            return
        output_text = self.output_edit.text().strip()
        if not output_text:
            QMessageBox.warning(self, "无法开始", "请选择结果目录。")
            return
        try:
            inputs = StudyInputs.from_mapping(self._paths)
            output_root = Path(output_text).expanduser().resolve()
        except Exception as exc:
            QMessageBox.warning(self, "无法开始", str(exc))
            return
        self._result = None
        self._project_path = None
        self._last_run_directory = None
        self._review_paths = {}
        self.flow.set_results()
        self.log.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("启动中…")
        self._thread = QThread(self)
        self._worker = WorkflowWorker(inputs, output_root)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.stage_changed.connect(self._on_stage_changed)
        self._worker.completed.connect(self._on_completed)
        self._worker.failed.connect(self._on_failed)
        self._worker.completed.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.completed.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread_finished)
        self._thread.start()
        self._refresh_sequence()

    @Slot(int, str)
    def _on_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{value}%　{message}")
        self.log.appendPlainText(message)

    @Slot(object)
    def _on_stage_changed(self, update: StageUpdate):
        # A progress message must never erase earlier quality results or links.
        states = {**self.flow.stage_states, update.key: update.status}
        tips = dict(self.flow.stage_tooltips)
        self._last_run_directory = update.run_directory
        if update.review_path is not None:
            self._review_paths[update.key] = update.review_path
            tips[update.key] = f"{update.key} · {update.status} · {update.confidence}\n点击查看配准彩虹图"
        else:
            tips[update.key] = "此阶段正在配准；已完成阶段仍可点击查看彩虹图。"
        self.flow.set_results(states, self._review_paths, self.flow.outputs | set(update.ready_outputs), tips)
        self._refresh_sequence()

    @Slot(object)
    def _on_completed(self, outcome):
        self._result = outcome
        self._last_run_directory = outcome.run_directory
        self._project_path = outcome.output_files["project"]
        self._review_paths, states, tips = _read_stage_reviews(outcome.run_directory)
        self.flow.set_results(states, self._review_paths, outcome.output_files.keys(), tips)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("配准完成 · 点击彩色粗线查看各阶段彩虹图")
        self.log.appendPlainText(f"结果目录：{outcome.run_directory}")
        self._refresh_sequence()

    @Slot(str, str, object)
    def _on_failed(self, message, details, run_directory=None):
        self.log.appendPlainText(details)
        states = dict(self.flow.stage_states)
        tips = dict(self.flow.stage_tooltips)
        active = [key for key, status in states.items() if status == "running"]
        if run_directory is not None:
            self._last_run_directory = Path(run_directory)
            try:
                reviews, recorded_states, recorded_tips = _read_stage_reviews(run_directory)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.log.appendPlainText(f"阶段报告读取失败：{exc}")
            else:
                self._review_paths.update(reviews)
                states.update(recorded_states)
                tips.update(recorded_tips)
        for key in active:
            states[key] = "failed"
            tips[key] = message + ("\n点击查看候选彩虹图" if key in self._review_paths else "\n未生成可查看的候选模型。")
        self.flow.set_results(states, self._review_paths, self.flow.outputs, tips)
        has_failed_review = any(states.get(key) == "failed" for key in self._review_paths)
        self.progress_bar.setFormat("配准未完成 · 点击红色粗线检查候选" if has_failed_review else "配准未完成 · 已保留已完成阶段，请查看运行详情")
        self._refresh_sequence()
        QMessageBox.warning(self, "配准未完成", message)

    @Slot()
    def _thread_finished(self):
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._worker = None
        self._refresh_sequence()

    def _viewer_command(self, *arguments):
        environment = os.environ.copy()
        if not getattr(sys, "frozen", False):
            source_root = str(Path(__file__).resolve().parents[1])
            current = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = source_root + (os.pathsep + current if current else "")
        executable, child_arguments = _child_process_command(arguments)
        subprocess.Popen(
            [executable, *child_arguments],
            env=environment,
            cwd=str(_application_root()),
        )

    def _view_stage(self, key):
        if key in self._review_paths:
            path = self._review_paths[key]
            if path.name == "project.json":
                self._viewer_command("--view-stage-project", str(path), "--stage-key", key)
            else:
                self._viewer_command("--view-stage", str(path))

    def _view_result(self):
        if self._project_path is not None:
            key = str(self._project_path.resolve())
            window = self._measurement_windows.get(key)
            if window is None:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                QApplication.processEvents()
                try:
                    from .scene_viewer import SceneViewer
                    # Parentless top-level window: independent taskbar entry,
                    # focus and minimization, while retaining shared-process cache.
                    window = SceneViewer(reusable=True)
                    self._measurement_windows[key] = window
                    window.show()
                    window.vtk_widget.Initialize()
                    window.load_project(self._project_path)
                finally:
                    QApplication.restoreOverrideCursor()
            else:
                window.show()
                window.raise_()
                window.activateWindow()
                window.reload_condyles()

    def _select_condyles(self):
        if self._thread is not None or self._selection_process is not None or "ct_mandible" not in self._paths:
            return
        mesh = self._paths["ct_mandible"]
        self._start_selection_process(
            ["--select-condyles", str(mesh), "--condyle-profile", str(condyle_profile_path(mesh))],
            ("condyle", "ct_mandible"),
            "正在打开髁突选区窗口…",
        )

    def _select_registration_region(self, key):
        if (
            self._thread is not None
            or self._selection_process is not None
            or key not in {"baseline_lower", "ct_dentition"}
            or key not in self._paths
        ):
            return
        mesh = self._paths[key]
        self._start_selection_process(
            [
                "--select-registration", str(mesh),
                "--selection-role", key,
                "--registration-profile", str(registration_profile_path(mesh, key)),
            ],
            ("registration", key),
            f"正在打开{ROLE_LABELS[key]}的配准选区窗口…",
        )

    def _start_selection_process(self, arguments, job, message):
        process = QProcess(self)
        environment = QProcessEnvironment.systemEnvironment()
        if not getattr(sys, "frozen", False):
            source = str(Path(__file__).resolve().parents[1])
            environment.insert("PYTHONPATH", source + os.pathsep + environment.value("PYTHONPATH", ""))
        process.setProcessEnvironment(environment)
        process.setWorkingDirectory(str(_application_root()))
        executable, child_arguments = _child_process_command(arguments)
        process.setProgram(executable)
        process.setArguments(child_arguments)
        process.finished.connect(self._selection_finished)
        process.errorOccurred.connect(self._selection_error)
        self._selection_process = process
        self._selection_job = job
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(message)
        self._refresh_sequence()
        self.log.appendPlainText(message)
        process.start()

    def _selection_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self._selection_finished(-1)

    def _selection_finished(self, exit_code, exit_status=None):
        process = self._selection_process
        if process is None:
            return
        job = self._selection_job
        self._selection_process = None
        self._selection_job = None
        if exit_code != 0:
            details = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
            self.log.appendPlainText(details)
            QMessageBox.warning(self, "选区窗口未正常关闭", "请查看运行详情；已经保存的选区仍保留。")
        elif job and job[0] == "condyle":
            self.log.appendPlainText("髁突选区窗口已关闭；主界面已刷新实际保存的选区状态。")
            for window in self._measurement_windows.values():
                window.reload_condyles()
        elif job:
            self.log.appendPlainText(f"{ROLE_LABELS[job[1]]}选区窗口已关闭；主界面已刷新实际保存状态。")
        process.deleteLater()
        self.flow.set_selection_statuses(saved_selection_keys(self._paths))
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("选区窗口已关闭")
        self._refresh_sequence()

    def _view_existing(self):
        filename, _ = QFileDialog.getOpenFileName(self, "选择 project.json", self.output_edit.text(), "配准项目 (project.json)")
        if not filename:
            return
        try:
            self._load_project(Path(filename))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, "无法打开结果", str(exc))

    def _load_project(self, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("workflow") != "mandibular_pose_transfer":
            raise ValueError("请选择本软件保存的 project.json。")
        paths = {}
        for spec in INPUT_SPECS:
            record = data["inputs"].get(spec.key)
            if record:
                candidate = Path(record["path"])
                if not candidate.is_absolute():
                    candidate = path.parent / candidate
                if candidate.is_file():
                    paths[spec.key] = candidate.resolve()
        reviews, review_states, tips = _read_stage_reviews(path.parent)
        states = {stage["key"]: stage["status"] for stage in data.get("stages", [])}
        states.update(review_states)
        for key in states:
            reviews.setdefault(key, path)
        self._paths = paths
        self._result = None
        self._project_path = path
        self._last_run_directory = path.parent
        self._review_paths = reviews
        self.flow.set_inputs(paths)
        self.flow.set_selection_statuses(saved_selection_keys(paths))
        self.flow.set_results(states, reviews, data.get("outputs", {}).keys(), tips)
        self.progress_bar.setValue(100)
        self._refresh_sequence()
        self.progress_bar.setFormat("已打开结果 · 可查看颌骨或阶段彩虹图")

    def _open_result_directory(self):
        if self._last_run_directory is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_run_directory)))

    def closeEvent(self, event: QCloseEvent):
        if self._selection_process is not None:
            QMessageBox.information(self, "选区窗口仍在运行", "请先保存并关闭当前选区窗口。")
            event.ignore()
            return
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "配准仍在运行", "请等待运行完成后再关闭窗口。")
            event.ignore()
            return
        for window in self._measurement_windows.values():
            window._reusable = False
            if not window.isVisible():
                if window.has_section_measurements() or window._thread is not None:
                    window.show()  # Keep unexported work alive independently.
                else:
                    window.close()
        super().closeEvent(event)


def main():
    application = QApplication.instance() or QApplication(sys.argv)
    application.setApplicationName("下颌位姿配准")
    application.setWindowIcon(
        QIcon(str(Path(__file__).resolve().parent / "assets" / "app_icon.ico"))
    )
    window = MainWindow()
    window.show()
    return application.exec()
