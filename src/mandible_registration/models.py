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


INPUT_SPECS: tuple[InputSpec, ...] = (
    InputSpec("baseline_lower", 1, "下颌口扫.1", "第一次口扫下颌.stl", "全牙列配准基准"),
    InputSpec("baseline_upper", 2, "上颌口扫.1", "第一次口扫上颌.stl", "固定坐标系"),
    InputSpec("ct_dentition", 3, "全牙列", "牙列CT.stl", "CT 牙列"),
    InputSpec("ct_mandible", 4, "颌骨", "下颌骨CT.stl", "跟随全牙列变换"),
    InputSpec("followup_upper", 5, "上颌口扫.2", "第二次口扫上颌.stl", "统一两组口扫坐标"),
    InputSpec("followup_lower", 6, "下颌口扫.2", "第二次口扫下颌.stl", "计算下颌位移"),
)


@dataclass(frozen=True)
class StudyInputs:
    baseline_lower: Path
    baseline_upper: Path
    ct_dentition: Path
    ct_mandible: Path
    followup_upper: Path
    followup_lower: Path

    @classmethod
    def from_mapping(cls, values: Mapping[str, str | Path]) -> "StudyInputs":
        missing = [spec.title for spec in INPUT_SPECS if not values.get(spec.key)]
        if missing:
            raise ValueError("尚未选择：" + "、".join(missing))
        paths = {spec.key: Path(values[spec.key]).expanduser().resolve() for spec in INPUT_SPECS}
        for spec in INPUT_SPECS:
            path = paths[spec.key]
            if not path.is_file():
                raise FileNotFoundError(f"找不到{spec.title}：{path}")
            if path.suffix.lower() != ".stl":
                raise ValueError(f"{spec.title}必须是 STL 文件：{path.name}")
        if len(set(paths.values())) != len(paths):
            raise ValueError("六个输入必须分别选择不同的 STL 文件。")
        return cls(**paths)

    @classmethod
    def from_dataset_directory(cls, directory: str | Path) -> "StudyInputs":
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"找不到测试数据目录：{root}")
        return cls.from_mapping(
            {spec.key: root / spec.expected_filename for spec in INPUT_SPECS}
        )

    def as_mapping(self) -> dict[str, Path]:
        return {spec.key: getattr(self, spec.key) for spec in INPUT_SPECS}

    def as_dict(self) -> dict[str, str]:
        return {key: str(path) for key, path in self.as_mapping().items()}
