"""Refuses to start unless the job sees no GPU or exactly one GPU from gpu_allowlist.txt.

On the course server GPUs 0-3 are split into MIG instances that belong to other users, and
CUDA numbers devices differently from nvidia-smi (those MIG instances get CUDA ids 4-7).
So a job must name its GPU by UUID. The check runs before CUDA is initialised.
Machines without gpu_allowlist.txt are not affected."""
import os
import sys
from pathlib import Path

ALLOWLIST = Path(__file__).with_name("gpu_allowlist.txt")


def check():
    if not ALLOWLIST.exists():
        return
    allowed = {ln.split()[0] for ln in ALLOWLIST.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis == "" or vis in allowed:
        return
    sys.exit(f"gpu_guard: CUDA_VISIBLE_DEVICES={vis!r} is not allowed here. "
             f"Use CUDA_VISIBLE_DEVICES='' for CPU or one UUID from {ALLOWLIST.name}.")
