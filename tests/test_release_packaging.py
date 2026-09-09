"""Catch source/installed-wheel drift before distributing a frozen build."""
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "release_packaging", Path(__file__).resolve().parents[1] / "scripts" / "package_windows_release.py"
)
packaging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packaging)


@pytest.mark.parametrize("bundled_content", [None, b"old workflow"])
def test_packaging_rejects_missing_or_stale_workflow(tmp_path, bundled_content):
    source = tmp_path / "src/mandible_registration/assets"
    source.mkdir(parents=True)
    (source / "workflow.maxilla.editable.svg").write_bytes(b"current workflow")
    release = tmp_path / "dist/release"
    if bundled_content is not None:
        bundled = release / "_internal/mandible_registration/assets"
        bundled.mkdir(parents=True)
        (bundled / "workflow.maxilla.editable.svg").write_bytes(bundled_content)
    with pytest.raises(RuntimeError, match="workflow.maxilla.editable.svg"):
        packaging.validate_application_assets(tmp_path, release)


def test_packaging_accepts_exact_source_assets(tmp_path):
    source = tmp_path / "src/mandible_registration/assets"
    source.mkdir(parents=True)
    release = tmp_path / "dist/release"
    bundled = release / "_internal/mandible_registration/assets"
    bundled.mkdir(parents=True)
    for name in ("workflow.editable.svg", "workflow.maxilla.editable.svg", "app_icon.ico"):
        content = f"current {name}".encode()
        (source / name).write_bytes(content)
        (bundled / name).write_bytes(content)
    packaging.validate_application_assets(tmp_path, release)
