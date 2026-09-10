from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class InputSpec:
    key: str
    sequence: int
    title: str
    expected_filename: str
    description: str
    required: bool = True


INPUT_SPECS: tuple[InputSpec, ...] = (
    InputSpec("baseline_lower", 1, "下颌口扫.1", "第一次口扫下颌.stl", "全牙列配准基准"),
    InputSpec("baseline_upper", 2, "上颌口扫.1", "第一次口扫上颌.stl", "固定坐标系"),
    InputSpec("ct_dentition", 3, "全牙列", "牙列CT.stl", "CT 牙列"),
    InputSpec("ct_mandible", 4, "颌骨", "下颌骨CT.stl", "跟随全牙列变换"),
    InputSpec("ct_maxilla", 5, "上颌骨", "上颌骨CT.stl", "可选；仅跟随颌骨.1定位", False),
    InputSpec("followup_upper", 6, "上颌口扫.2", "第二次口扫上颌.stl", "统一两组口扫坐标"),
    InputSpec("followup_lower", 7, "下颌口扫.2", "第二次口扫下颌.stl", "计算下颌位移"),
)

REQUIRED_INPUT_SPECS = tuple(spec for spec in INPUT_SPECS if spec.required)
REQUIRED_INPUT_KEYS = frozenset(spec.key for spec in REQUIRED_INPUT_SPECS)


@dataclass(frozen=True)
class StudyInputs:
    baseline_lower: Path
    baseline_upper: Path
    ct_dentition: Path
    ct_mandible: Path
    followup_upper: Path
    followup_lower: Path
    ct_maxilla: Path | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, str | Path]) -> "StudyInputs":
        missing = [spec.title for spec in REQUIRED_INPUT_SPECS if not values.get(spec.key)]
        if missing:
            raise ValueError("尚未选择：" + "、".join(missing))
        paths = {
            spec.key: Path(values[spec.key]).expanduser().resolve()
            for spec in INPUT_SPECS
            if values.get(spec.key)
        }
        for spec in INPUT_SPECS:
            if spec.key not in paths:
                continue
            path = paths[spec.key]
            if not path.is_file():
                raise FileNotFoundError(f"找不到{spec.title}：{path}")
            if path.suffix.lower() != ".stl":
                raise ValueError(f"{spec.title}必须是 STL 文件：{path.name}")
        if len(set(paths.values())) != len(paths):
            raise ValueError("各输入角色必须分别选择不同的 STL 文件。")
        return cls(
            **{spec.key: paths[spec.key] for spec in REQUIRED_INPUT_SPECS},
            ct_maxilla=paths.get("ct_maxilla"),
        )

    @classmethod
    def from_dataset_directory(cls, directory: str | Path) -> "StudyInputs":
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"找不到测试数据目录：{root}")
        paths = {
            spec.key: root / spec.expected_filename
            for spec in REQUIRED_INPUT_SPECS
        }
        for spec in INPUT_SPECS:
            candidate = root / spec.expected_filename
            if not spec.required and candidate.is_file():
                paths[spec.key] = candidate
        return cls.from_mapping(paths)

    def as_mapping(self) -> dict[str, Path]:
        return {
            spec.key: value
            for spec in INPUT_SPECS
            if (value := getattr(self, spec.key)) is not None
        }

    def as_dict(self) -> dict[str, str]:
        return {key: str(path) for key, path in self.as_mapping().items()}
