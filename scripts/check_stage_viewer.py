"""Open a stage adapter through the public general-registration review API."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_alignment.integration.review import load_review_manifest, run_registration_review


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path, nargs="?", help="保留旧命令兼容；查看器标注仍写在结果目录。")
    args = parser.parse_args()
    load_review_manifest(args.result)
    run_registration_review(args.result)


if __name__ == "__main__":
    main()
