import json

import numpy as np

from mandible_registration.exporter import write_json


def test_json_writer_replaces_nonfinite_values_with_null(tmp_path) -> None:
    path = write_json(
        tmp_path / "report.json",
        {"nan": float("nan"), "inf": np.float64(float("inf")), "ok": np.eye(2)},
    )

    text = path.read_text("utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text)["nan"] is None
