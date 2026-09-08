"""Mandibular pose registration workflow."""

import os

# Open3D's parallel RANSAC consumes a process-global RNG. Seed alone does not
# make its candidates reproducible. Set these before loading Open3D; only this
# application process is affected, not Windows or the sibling application.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

__version__ = "1.0.0"
