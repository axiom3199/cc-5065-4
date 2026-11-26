"""Tests for the direction guide: exact pdfs, coverage, update timing."""

import math

import numpy as np
import pytest

from dirguide.guiding import FOUR_PI, DirectionGuide, GridSpec
from dirguide import verify


def test_pdf_normalizes_after_learning():
    res = verify.check_guide_normalization()
    assert res["ok"], res


def test_uniform_floor_keeps_unobserved_directions_supported():
    g = verify._make_spiky_guide(eta=0.25)
    centers = g.smap.bin_center_dirs
    pdf = g.pdf(np.zeros(g.smap.K, dtype=int), centers)
    # Every exact bin direction is at least the analytic uniform floor.
    assert pdf.min() >= 0.25 / FOUR_PI - 1e-12
    # A never-observed far-away cell is exactly uniform sphere.
    pdf_empty = g.pdf(np.full(g.smap.K, 4), centers)
    assert np.allclose(pdf_empty, 1.0 / FOUR_PI)


def test_true_mixture_pdf_matches_eval_and_all_bins_visited():
    res = verify.check_mixture_support()
    assert res["pdf_eval_max_abs_diff"] < 1e-10
    assert res["hemisphere_support_ok"]
    assert res["bin_coverage_fraction"] > 0.99


def test_known_integral_through_mixture_estimator():
    res = verify.check_known_integral()
    assert res["ok"], res


def test_sampling_returns_exact_pdf():
    g = DirectionGuide(GridSpec((-2, 0, -2), (2, 3, 2), (4, 3, 4)))
    rng = np.random.default_rng(0)
    points = np.array([[0.1, 0.2, 0.3]] * 64)
    normals = np.array([[0.0, 1.0, 0.0]] * 64)
    out = g.sample_mixture(points, normals, rng, guide_mix=0.5)
    p2, _, _ = g.mixture_pdf(out["cells"], normals, out["dirs"], guide_mix=0.5)
    assert np.allclose(out["pdf"], p2)
    assert np.all(out["pdf"] > 0.0)


def test_decay_and_rebuild_change_versions_and_mass():
    g = DirectionGuide(GridSpec((-1, -1, -1), (1, 1, 1), (2, 2, 2)))
    ids = np.array([0 * g.smap.K + 5, 1 * g.smap.K + 9])
    w = np.array([10.0, 7.0])
    v0 = g.version
    g.commit(ids, w)
    assert g.version == v0 + 1
    assert g.W[0, 5] == 10.0
    total = g.W.sum()
    g.on_light_moved("decay", move_retain=0.25)
    assert math.isclose(g.W.sum(), total * 0.25, rel_tol=1e-12)
    g.on_light_moved("rebuild")
    assert g.W.sum() == 0.0
    # floor still defines a valid normalized distribution
    centers = g.smap.bin_center_dirs
    pdf = g.pdf(np.zeros(g.smap.K, dtype=int), centers)
    assert np.isfinite(pdf).all()


def test_snapshot_roundtrip(tmp_path):
    g = DirectionGuide(GridSpec((-1, -1, -1), (1, 1, 1), (2, 2, 2)))
    g.commit(np.array([3]), np.array([4.0]))
    path = g.save_version(str(tmp_path), label="pytest")
    g2 = DirectionGuide.from_snapshot(__import__("json").load(open(path)))
    assert np.allclose(g.W, g2.W)
    assert g2.version == g.version


def test_training_never_enters_image_accumulator():
    """Huge guide weights must not be added as energy to the image."""
    from dirguide.session import RenderSession

    s = RenderSession("window", width=48, height=32, seed=3, strategy="guide")
    s.step_spp()
    before = s.accum.copy()
    # Committing arbitrary training data changes only guide weights.
    big_ids = np.arange(0, 20)
    s.guide.commit(big_ids, np.full(20, 1e12))
    assert np.array_equal(before, s.accum)
    # The next estimate stays finite despite extreme proposal concentration.
    s.step_spp()
    assert np.isfinite(s.accum).all()


def test_deposit_weight_needs_pdf_correction():
    """Weighted-bin identity: depositing t/p recovers ∫_bin t under p~proposal.

    Regression guard for the online training credit: storing fr*cos*L
    directly (instead of the MC contribution fr*cos*L/p) biases the learned
    histogram by the current non-uniform proposal density.
    """
    smap = DirectionGuide(GridSpec((-1, -1, -1), (1, 1, 1), (1, 1, 1))).smap
    K = smap.K
    rng = np.random.default_rng(7)
    # Sparse target T_b and a deliberately different non-uniform proposal P_b.
    T = rng.random(K) * 0.2
    T[17] = 50.0
    T[120] = 30.0
    P = rng.random(K) + 0.05
    P /= float((P * smap.bin_domega).sum())  # density integrates to 1
    prop_mass = P * smap.bin_domega
    prop_cdf = np.cumsum(prop_mass); prop_cdf[-1] = 1.0

    n = 400_000
    bins = np.searchsorted(prop_cdf, rng.random(n)).astype(np.int64)
    p_dir = P[bins]
    correct = T[bins] / p_dir          # MC contribution t / p
    uncorrected = T[bins]              # the old p*(t/p) = t credit

    def q_of(weights):
        hist = np.bincount(bins, weights=weights, minlength=K)
        q = hist / max(hist.sum(), 1e-300)
        return q

    q_true = T * smap.bin_domega
    q_true /= q_true.sum()
    q_ok = q_of(correct)
    q_bad = q_of(uncorrected)
    l1_ok = float(np.abs(q_ok - q_true).sum())
    l1_bad = float(np.abs(q_bad - q_true).sum())
    assert l1_ok < 0.05, l1_ok
    assert l1_ok < 0.25 * l1_bad, (l1_ok, l1_bad)


def test_tracer_runs_and_deposits_credit():
    from dirguide.scene import build_scene, default_grid_spec
    from dirguide.tracer import trace_spp

    scn, cam = build_scene("window")
    g = DirectionGuide(default_grid_spec("window"))
    rng = np.random.default_rng(0)
    res = trace_spp(scn, cam, g, rng, 40, 26, lam=0.6, train=True)
    assert res["image"].shape == (26, 40, 3)
    assert np.isfinite(res["image"]).all()
    assert res["deposit_ids"].size > 0
    assert res["deposit_weights"].sum() > 0


def test_guided_and_baseline_means_agree_statistically():
    """Both estimators target the same radiance; means must be close."""
    from dirguide.session import RenderSession

    means = {}
    for strat in ("bsdf", "guide"):
        s = RenderSession("uniform", width=56, height=40, seed=2024, strategy=strat)
        for _ in range(96):
            s.step_spp()
        means[strat] = float(s.image_hdr().mean())
    rel = abs(means["guide"] - means["bsdf"]) / max(means["bsdf"], 1e-6)
    assert rel < 0.08, means
