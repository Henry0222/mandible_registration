"""Reproduce CT candidate stability in isolated single-threaded OpenMP runs."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mandible_registration import workflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    inputs = workflow.StudyInputs.from_dataset_directory(args.dataset)
    target, tf = workflow.load_mesh(inputs.baseline_lower)
    source, sf = workflow.load_mesh(inputs.ct_dentition)
    config = replace(workflow.AlignmentConfig(), partial_overlap_threshold=0.12)
    for repeat in range(args.repeats):
        directory = workflow.timestamped_run_directory(args.output)
        try:
            result, diagnostics = workflow._register_ct_with_consensus(target, source, tf, sf, config, directory, None)
            print(repeat, "PASS", diagnostics["selected_attempt"], result.transformation.tolist(), flush=True)
        except workflow.WorkflowError as exc:
            print(repeat, "FAIL", str(exc), directory, flush=True)


if __name__ == "__main__":
    main()
