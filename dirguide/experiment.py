"""Headless equal-budget experiments and parallel reference rendering.

Everything here is deterministic given a seed and safe to call in worker
processes.  Runs compare:

* fixed lighting: pure BSDF sampling vs true-mixture direction guiding;
* light-move protocol: identical trained table at move time, then
  stale / decay / rebuild handling, plus a BSDF baseline.
"""

from __future__ import annotations

import os

# Limit BLAS threads *before* importing numpy so many worker processes do
# not oversubscribe the machine.
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .imaging import exposure_from_reference
from .scene import build_scene
from .session import RenderSession
from .tracer import trace_spp

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
IMAGE_DIR = os.path.join(RESULTS_DIR, "images")
VERSION_DIR = os.path.join(RESULTS_DIR, "versions")

PLACEMENT_A = {
    "window": "A:left_window",
    "uniform": "A:ceiling_front",
}
PLACEMENT_B = {
    "window": "B:right_window",
    "uniform": "B:ceiling_back",
}


# ----------------------------------------------------------- reference jobs
def reference_chunk_worker(params: tuple[Any, ...]) -> np.ndarray:
    kind, placement, width, height, chunk_spp, seed, max_bounces = params
    scn, cam = build_scene(kind)
    scn.set_light_placement(placement)
    rng = np.random.default_rng(seed)
    acc = np.zeros((height, width, 3), dtype=np.float64)
    for _ in range(chunk_spp):
        res = trace_spp(
            scn, cam, None, rng, width, height,
            lam=0.0, train=False, max_bounces=max_bounces,
        )
        img = np.nan_to_num(res["image"], nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(img, 0.0, 1e6, out=img)
        acc += img
    return acc


def render_reference(
    kind: str,
    placement: str,
    *,
    width: int,
    height: int,
    total_spp: int,
    workers: int,
    chunk_spp: int = 32,
    seed_base: int = 90000,
    max_bounces: int = 10,
) -> np.ndarray:
    """Parallel unbiased BSDF+NEE reference (chunks sum linearly)."""
    chunks = max(1, total_spp // chunk_spp)
    params = [
        (kind, placement, width, height, chunk_spp,
         seed_base + 7919 * i + hash(placement) % 100000, max_bounces)
        for i in range(chunks)
    ]
    t0 = time.perf_counter()
    if workers <= 1:
        results = [reference_chunk_worker(p) for p in params]
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(processes=workers) as pool:
            results = pool.map(reference_chunk_worker, params)
    total = sum(results) / float(chunks * chunk_spp)
    print(f"  ref {kind} {placement}: {chunks*chunk_spp} spp in "
          f"{time.perf_counter()-t0:.1f}s ({workers} workers)", flush=True)
    return total


# ------------------------------------------------------------- study jobs
@dataclass
class FixedJob:
    kind: str
    strategy: str
    spp: int
    seed: int
    width: int
    height: int
    placement: str | None = None
    guide_mix: float = 0.6
    eta: float = 0.2
    half_life: float = 6.0
    use_nee: bool = True


def fixed_run_worker(job: FixedJob) -> dict[str, Any]:
    s = RenderSession(
        kind=job.kind, width=job.width, height=job.height, seed=job.seed,
        strategy=job.strategy, guide_mix=job.guide_mix, eta=job.eta,
        half_life=job.half_life, use_nee=job.use_nee,
    )
    if job.placement:
        s.scene.set_light_placement(job.placement)
    ref_path = os.path.join(METRICS_DIR, f"ref_{job.kind}_{_tag(job.placement)}.npy")
    if os.path.exists(ref_path):
        s.reference = np.load(ref_path)
        s.exposure = exposure_from_reference(s.reference)
    for _ in range(job.spp):
        s.step_spp()
    return {
        "kind": job.kind,
        "strategy": job.strategy,
        "seed": job.seed,
        "placement": job.placement or s.scene.active_placement,
        "history": s.history,
        "hdr": s.image_hdr(),
        "exposure": s.exposure,
        "guide_W": s.guide.W.copy(),
        "guide_version": s.guide.version,
        "n_deposits": s.guide.n_deposits,
        "elapsed": s.elapsed,
    }


@dataclass
class MoveJob:
    kind: str
    policy: str           # stale / decay / rebuild / bsdf
    train_spp: int
    test_spp: int
    seed: int
    width: int
    height: int
    guide_mix: float = 0.6
    eta: float = 0.2
    half_life: float = 6.0
    move_retain: float = 0.25


def move_run_worker(job: MoveJob) -> dict[str, Any]:
    s = RenderSession(
        kind=job.kind, width=job.width, height=job.height, seed=job.seed,
        strategy="bsdf" if job.policy == "bsdf" else "guide",
        guide_mix=job.guide_mix, eta=job.eta, half_life=job.half_life,
        move_retain=job.move_retain,
    )
    ref_a = os.path.join(METRICS_DIR, f"ref_{job.kind}_{_tag(PLACEMENT_A[job.kind])}.npy")
    ref_b = os.path.join(METRICS_DIR, f"ref_{job.kind}_{_tag(PLACEMENT_B[job.kind])}.npy")
    if os.path.exists(ref_a):
        s.reference = np.load(ref_a)
    # Phase A: identical online guide training for all policies.
    for _ in range(job.train_spp):
        s.step_spp()
    phase_a_last = dict(s.history[-1])
    w_at_move = s.guide.W.copy()

    # Move light and apply policy.
    if job.policy != "bsdf":
        s.set_strategy(job.policy)
    if os.path.exists(ref_b):
        s.reference = np.load(ref_b)
        s.exposure = exposure_from_reference(s.reference)
    msg = s.move_light(PLACEMENT_B[job.kind])
    b_history: list[dict[str, Any]] = []
    for _ in range(job.test_spp):
        s.step_spp()
        b_history.append(dict(s.history[-1]))
    return {
        "kind": job.kind,
        "policy": job.policy,
        "seed": job.seed,
        "move_message": msg,
        "phase_a_last": phase_a_last,
        "history_b": b_history,
        "hdr_b": s.image_hdr(),
        "exposure": s.exposure,
        "w_at_move": w_at_move,
        "w_final": s.guide.W.copy(),
        "guide_version_final": s.guide.version,
        "guide_spp": s.guide.spp,
        "n_deposits": s.guide.n_deposits,
        "elapsed": s.elapsed,
    }


def _tag(placement: str | None) -> str:
    if not placement:
        return "A"
    return placement.split(":")[0]


def aggregate_curves(runs: list[dict[str, Any]], key: str = "history") -> dict[str, Any]:
    """Average per-spp metrics over seeds."""
    curves: dict[str, list[float]] = {}
    n = min(len(r[key]) for r in runs)
    metric_keys = [k for k in runs[0][key][0].keys()
                   if isinstance(runs[0][key][0][k], (int, float))
                   and k not in ("spp", "elapsed", "step_ms",
                                 "used_guide_version", "guide_version",
                                 "light_generation", "deposits")]
    for k in metric_keys:
        series = np.array([[pt[k] for pt in r[key][:n]] for r in runs], dtype=float)
        curves[k + "_mean"] = series.mean(axis=0).tolist()
        curves[k + "_std"] = series.std(axis=0).tolist()
    curves["spp"] = [runs[0][key][i]["spp"] for i in range(n)]
    return curves
