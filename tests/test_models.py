from pathlib import Path

import pytest

from mandible_registration.models import INPUT_SPECS, StudyInputs


def test_dataset_directory_uses_the_six_expected_names(tmp_path: Path) -> None:
    for spec in INPUT_SPECS:
        (tmp_path / spec.expected_filename).write_bytes(b"solid empty\nendsolid empty\n")

    inputs = StudyInputs.from_dataset_directory(tmp_path)

    assert inputs.baseline_lower.name == "第一次口扫下颌.stl"
    assert inputs.followup_lower.name == "第二次口扫下颌.stl"


def test_missing_dataset_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="下颌口扫.1"):
        StudyInputs.from_dataset_directory(tmp_path)


def test_same_stl_cannot_fill_multiple_slots(tmp_path: Path) -> None:
    mesh = tmp_path / "one.stl"
    mesh.write_bytes(b"solid empty\nendsolid empty\n")
    with pytest.raises(ValueError, match="不同"):
        StudyInputs.from_mapping({spec.key: mesh for spec in INPUT_SPECS})
