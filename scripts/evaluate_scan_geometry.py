"""Read-only geometric equivalence check; does not estimate clinical accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from mandible_registration.scene_data import load_scene, load_mesh
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    args = parser.parse_args()
    data = json.loads(args.project.read_text(encoding="utf-8"))
    keys = ("baseline_lower", "followup_lower", "baseline_upper", "followup_upper")
    scene = load_scene(args.project, keys=(keys[0], keys[2]))  # validates model hashes before reading
    models = {model.key: model for model in scene.models}
    raw_followups = {}
    for key in (keys[1], keys[3]):
        record = data["inputs"][key]
        path = Path(record["path"])
        if not path.is_absolute():
            path = args.project.parent / path
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Input content changed: {key}")
        raw_followups[key], _ = load_mesh(path)
    upper = np.asarray(data["transforms"]["T_UPPER"]["matrix"])
    delta = scene.delta
    findings = {}
    for name, first, second, transform in (
        ("lower", keys[0], keys[1], delta), ("upper", keys[2], keys[3], np.eye(4)),
    ):
        a = np.asarray(models[first].mesh.vertices)
        b = np.asarray(raw_followups[second].vertices)
        b = b @ upper[:3, :3].T + upper[:3, 3]
        a = a @ transform[:3, :3].T + transform[:3, 3]
        cloud_a = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(a))
        cloud_b = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(b))
        distances = np.concatenate((np.asarray(cloud_a.compute_point_cloud_distance(cloud_b)),
                                    np.asarray(cloud_b.compute_point_cloud_distance(cloud_a))))
        findings[name] = {
            "definition": "bidirectional nearest-vertex distances after the saved rigid transforms; not target registration error",
            "vertices": [len(a), len(b)],
            "triangles": [len(models[first].mesh.triangles), len(raw_followups[second].triangles)],
            "median_mm": float(np.median(distances)), "p95_mm": float(np.percentile(distances, 95)),
            "max_mm": float(distances.max()),
        }
    print(json.dumps({"project": str(args.project.resolve()), "checks": findings,
                      "limitation": "Surface equivalence does not prove scan provenance, bite accuracy, or condylar accuracy."}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
