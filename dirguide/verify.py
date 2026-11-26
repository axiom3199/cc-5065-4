"""Coverage and correctness certificates for the guiding sampler.

These checks back the claim that low-probability directions remain covered
and that the sampler evaluates the *true* mixture probability.  They are
used both by pytest and by the experiment script, which writes the results
to outputs/metrics/coverage_checks.json.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .guiding import FOUR_PI, DirectionGuide, GridSpec


def _make_spiky_guide(eta: float = 0.2, lam: float = 0.7) -> DirectionGuide:
    g = DirectionGuide(GridSpec((-1, -1, -1), (1, 1, 1), (2, 2, 2)),
                       theta_bins=12, phi_bins=24, eta=eta, guide_mix=lam)
    # Put almost all learned mass in one obscure bin of one cell.
    g.W[0, 137] = 100000.0
    g.W[0, 12] = 5000.0
    g._rebuild_cdf()
    return g


def check_guide_normalization() -> dict[str, Any]:
    g = _make_spiky_guide()
    centers = g.smap.bin_center_dirs
    worst = 0.0
    for cell in (0, 3, 7):
        pdf = g.pdf(np.full(g.smap.K, cell), centers)
        integral = float(np.sum(pdf * g.smap.bin_domega))
        worst = max(worst, abs(integral - 1.0))
    ok = worst < 2e-2  # bin-center quadrature tolerance
    return {"name": "guide_pdf_normalizes", "ok": ok,
            "max_abs_error": worst, "tolerance": 2e-2}


def check_floor_support() -> dict[str, Any]:
    eta = 0.2
    g = _make_spiky_guide(eta=eta)
    centers = g.smap.bin_center_dirs
    pdf = g.pdf(np.zeros(g.smap.K, dtype=int), centers)
    floor = eta / FOUR_PI
    min_pdf = float(pdf.min())
    ok = min_pdf >= floor - 1e-12
    return {"name": "uniform_floor_every_direction", "ok": ok,
            "min_guide_pdf": min_pdf, "guaranteed_floor": floor}


def check_mixture_support() -> dict[str, Any]:
    lam = 0.7
    g = _make_spiky_guide(lam=lam)
    rng = np.random.default_rng(0)
    n = 200000
    points = np.zeros((n, 3))
    normals = np.broadcast_to(np.array([0.0, 1.0, 0.0]), (n, 3)).copy()
    samp = g.sample_mixture(points, normals, rng, guide_mix=lam)
    d, p = samp["dirs"], samp["pdf"]
    p_eval, pg, pb = g.mixture_pdf(samp["cells"], normals, d, guide_mix=lam)
    pdf_match = float(np.max(np.abs(p - p_eval)))
    # On the hemisphere the BSDF component alone guarantees support.
    hemi = d[:, 1] > 0.0
    bound = (1.0 - lam) * np.maximum(d[hemi, 1], 0.0) / math.pi
    support_ok = bool(np.all(p[hemi] + 1e-12 >= bound))
    # Every bin must be visited despite the spiky learned table.
    bins = g.smap.bin_ids(d)
    visited = np.unique(bins)
    coverage = visited.size / g.smap.K
    return {
        "name": "true_mixture_pdf_and_coverage",
        "ok": pdf_match < 1e-10 and support_ok and coverage > 0.99,
        "pdf_eval_max_abs_diff": pdf_match,
        "hemisphere_support_ok": support_ok,
        "bin_coverage_fraction": coverage,
        "bins_visited": int(visited.size),
        "bins_total": int(g.smap.K),
    }


def check_known_integral() -> dict[str, Any]:
    """Integrate f(w) = 0.3 + max(w.n,0)^2 using the true mixture pdf."""
    lam = 0.65
    g = _make_spiky_guide(eta=0.15, lam=lam)
    rng = np.random.default_rng(123)
    n = 400000
    points = np.zeros((n, 3))
    normal = np.array([0.3, 0.8, -0.5])
    normal /= np.linalg.norm(normal)
    normals = np.broadcast_to(normal, (n, 3)).copy()
    samp = g.sample_mixture(points, normals, rng, guide_mix=lam)
    d, p = samp["dirs"], samp["pdf"]
    cos = np.maximum(d @ normal, 0.0)
    f = 0.3 + cos**2
    estimate = float(np.mean(f / p))
    analytic = 0.3 * FOUR_PI + 2.0 * math.pi / 3.0
    rel_err = abs(estimate - analytic) / analytic
    ok = rel_err < 0.02
    return {"name": "mixture_estimator_known_integral", "ok": ok,
            "estimate": estimate, "analytic": analytic,
            "relative_error": rel_err}


def check_bsdf_only_integral() -> dict[str, Any]:
    """Lam=0 path must reduce exactly to cosine pdf and integrate cos."""
    g = _make_spiky_guide()
    rng = np.random.default_rng(7)
    n = 200000
    normals = np.broadcast_to(np.array([0.0, 0.0, 1.0]), (n, 3)).copy()
    d, p_b = g.cosine_hemisphere(normals, rng)
    cos = np.maximum(d[:, 2], 0.0)
    est = float(np.mean(cos / p_b))
    rel = abs(est - math.pi) / math.pi
    return {"name": "cosine_pdf_sanity", "ok": rel < 0.02,
            "estimate": est, "analytic": math.pi, "relative_error": rel}


def run_all_checks() -> dict[str, Any]:
    checks = [
        check_guide_normalization(),
        check_floor_support(),
        check_mixture_support(),
        check_known_integral(),
        check_bsdf_only_integral(),
    ]
    return {
        "all_ok": all(c["ok"] for c in checks),
        "checks": checks,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_all_checks(), indent=2, ensure_ascii=False))
