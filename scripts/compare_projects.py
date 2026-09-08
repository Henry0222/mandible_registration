"""Compare two workflow projects without treating pose magnitude as pose error."""
from pathlib import Path
import argparse
import json
import math

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    for key in ("T_CT", "T_UPPER", "T_DELTA", "T_MANDIBLE_T0", "T_MANDIBLE_T1"):
        expected = np.asarray(reference["transforms"][key]["matrix"], dtype=float)
        actual = np.asarray(candidate["transforms"][key]["matrix"], dtype=float)
        relative = actual @ np.linalg.inv(expected)
        angle = math.degrees(
            math.acos(float(np.clip((np.trace(relative[:3, :3]) - 1) / 2, -1, 1)))
        )
        translation = float(np.linalg.norm(relative[:3, 3]))
        maximum = float(np.max(np.abs(actual - expected)))
        print(
            f"{key}: relative_rotation_deg={angle:.9f}, "
            f"relative_translation_mm={translation:.9f}, max_abs_matrix={maximum:.9g}"
        )
    t_ct = np.asarray(candidate["transforms"]["T_CT"]["matrix"], dtype=float)
    t_delta = np.asarray(candidate["transforms"]["T_DELTA"]["matrix"], dtype=float)
    t1 = np.asarray(candidate["transforms"]["T_MANDIBLE_T1"]["matrix"], dtype=float)
    print(f"candidate_chain_max_abs={np.max(np.abs(t1 - t_delta @ t_ct)):.9g}")


if __name__ == "__main__":
    main()
