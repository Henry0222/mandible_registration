from pathlib import Path
import sys
import os
import subprocess

from mandible_registration import __version__
from mandible_registration.__main__ import ensure_standard_streams
from mandible_registration import gui


def test_release_version_is_1_1_0():
    assert __version__ == "1.1.0"


def test_main_gui_import_does_not_initialize_section_or_vtk_stack():
    source = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": str(source)}
    result = subprocess.run([sys.executable, "-c", (
        "import sys; import mandible_registration.gui; "
        "assert 'mandible_registration.section_viewer' not in sys.modules; "
        "assert 'mandible_registration.section_geometry' not in sys.modules; "
        "assert 'mandible_registration.scene_viewer' not in sys.modules; "
        "assert not any(k.startswith('vtkmodules') for k in sys.modules)"
    )], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_windowed_executable_installs_writable_standard_streams(monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    ensure_standard_streams()

    assert sys.stdout is not None
    assert sys.stderr is not None
    assert sys.stdout.write("Open3D diagnostic\n") > 0
    assert sys.stderr.write("Open3D warning\n") > 0
    sys.stdout.flush()
    sys.stderr.flush()


def test_frozen_child_process_reuses_executable_without_python_module_flag(monkeypatch):
    monkeypatch.setattr(gui.sys, "executable", r"C:\Release\MandibleRegistration-v1.1.0.exe")
    executable, arguments = gui._child_process_command(
        ["--select-registration", "model.stl"], frozen=True
    )
    assert executable.endswith("MandibleRegistration-v1.1.0.exe")
    assert arguments == ["--select-registration", "model.stl"]


def test_source_child_process_keeps_module_entrypoint(monkeypatch):
    monkeypatch.setattr(gui.sys, "executable", r"C:\Python\python.exe")
    executable, arguments = gui._child_process_command(["--view-stage", "result.json"], frozen=False)
    assert Path(executable).name == "python.exe"
    assert arguments == ["-m", "mandible_registration", "--view-stage", "result.json"]
