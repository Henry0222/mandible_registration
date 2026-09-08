from pathlib import Path
import sys

from mandible_registration import __version__
from mandible_registration.__main__ import ensure_standard_streams
from mandible_registration import gui


def test_release_version_is_1_0_0():
    assert __version__ == "1.0.0"


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
    monkeypatch.setattr(gui.sys, "executable", r"C:\Release\MandibleRegistration-v1.0.0.exe")
    executable, arguments = gui._child_process_command(
        ["--select-registration", "model.stl"], frozen=True
    )
    assert executable.endswith("MandibleRegistration-v1.0.0.exe")
    assert arguments == ["--select-registration", "model.stl"]


def test_source_child_process_keeps_module_entrypoint(monkeypatch):
    monkeypatch.setattr(gui.sys, "executable", r"C:\Python\python.exe")
    executable, arguments = gui._child_process_command(["--view-stage", "result.json"], frozen=False)
    assert Path(executable).name == "python.exe"
    assert arguments == ["-m", "mandible_registration", "--view-stage", "result.json"]
