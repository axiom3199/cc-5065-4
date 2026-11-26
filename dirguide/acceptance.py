"""Statistical-correctness acceptance for the learned-proposal mixture.

This module is the direction-guiding acceptance set.  It builds an analytic
small scene -- one shading point (or a tiny image of tilted shading points)
lit by spherical-cap lights -- where the reflected radiance has a closed
form, independently derived:

    L_o = rho * L_e * sin^2(alpha) * max(n . w0, 0)

because the integral of cos over a cap of half-angle ``alpha`` centered on
``w0`` is exactly ``(n . w0) * pi * sin^2(alpha)`` whenever the whole cap
lies above the horizon.

On top of that reference the suite checks, with mean confidence intervals
derived from independent trials (never from the sampler's own bookkeeping):

* probability normalization of the learned/uniform/BSDF mixture and a
  chi-square distribution test of the sampler against the exact bin masses;
* an image test: per-pixel means against the analytic image;
* a mid-sampling fixture that turns a small light on halfway, sends the old
  direction table back to the sampler together with the samples that
  generated it, and then checks energy over multiple independent intervals;
* the zero-support negative example (exploration branch eta disabled): the
  new light's direction is completely missed, and the fixture must demand a
  proposal rebuild instead of accepting a stable underestimate as
  convergence; the first anomalous light-source component is saved;
* fixed-budget independent trials with separate bias and variance
  assertions, cross-checked against an independent unguided estimator;
* training-size invariance (no systematic darkening), injected wrong
  densities (stale / scaled pdf) that must be caught, and the separation of
  evaluation data from learning records.

Every mean-CI entry carries the trial/interval means it was computed from
(``evidence``) so a reviewer can recompute the interval; the acceptance set
never reports numbers that did not come from an actual run.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

import numpy as np

from .guiding import FOUR_PI, DirectionGuide, GridSpec, luminance

SCHEMA = "dirguide.acceptance.v1"
ANOMALY_SCHEMA = "dirguide.acceptance.anomaly.v1"

# Two-sided 95% t critical values by degrees of freedom (df>=30 -> normal).
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t_critical_95(df: int) -> float:
    if df < 1:
        return float("inf")
    if df >= 30:
        return 1.960
    return _T95[int(df)]


def mean_ci(samples: np.ndarray) -> dict[str, Any]:
    """95% t-interval of the mean of independent samples.

    Returns the interval together with the raw evidence so any reported CI
    can be recomputed from the execution record.
    """
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if n == 0:
        raise ValueError("mean_ci needs at least one sample")
    mean = float(x.mean())
    if n == 1:
        return {"mean": mean, "half_width": float("inf"), "lo": -float("inf"),
                "hi": float("inf"), "n": 1, "std": 0.0, "confidence": 0.95,
                "evidence": [mean]}
    std = float(x.std(ddof=1))
    hw = t_critical_95(n - 1) * std / math.sqrt(n)
    return {"mean": mean, "half_width": hw, "lo": mean - hw, "hi": mean + hw,
            "n": n, "std": std, "confidence": 0.95,
            "evidence": [float(v) for v in x]}


def ci_covers(ci: dict[str, Any], reference: float) -> bool:
    return bool(ci["lo"] <= reference <= ci["hi"])


# --------------------------------------------------------------------------
# Analytic small scene
# --------------------------------------------------------------------------
def _unit(v: tuple[float, float, float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


class CapLight:
    """Spherical-cap directional light: constant radiance inside the cap."""

    def __init__(
        self,
        name: str,
        direction: tuple[float, float, float],
        alpha: float,
        emission: tuple[float, float, float],
    ) -> None:
        self.name = name
        self.w0 = _unit(direction)
        self.alpha = float(alpha)
        self.le = np.asarray(emission, dtype=np.float64)

    @property
    def solid_angle(self) -> float:
        return 2.0 * math.pi * (1.0 - math.cos(self.alpha))

    @property
    def sin2_alpha(self) -> float:
        s = math.sin(self.alpha)
        return s * s

    def radiance(self, dirs: np.ndarray) -> np.ndarray:
        cos = dirs @ self.w0
        on = cos >= math.cos(self.alpha)
        return self.le[None, :] * on[:, None]

    def fully_above_horizon(self, normal: np.ndarray) -> bool:
        n = normal / np.linalg.norm(normal)
        cos_min = (n @ self.w0) * math.cos(self.alpha) - (
            np.linalg.norm(n - (n @ self.w0) * self.w0) * math.sin(self.alpha)
        )
        return bool(cos_min > 0.0)

    def analytic_exitance(self, normal: np.ndarray, albedo: np.ndarray) -> np.ndarray:
        """Closed-form reflected radiance for a fully-visible cap."""
        n = normal / np.linalg.norm(normal)
        return albedo * self.le * self.sin2_alpha * max(float(n @ self.w0), 0.0)


# Light A: present from the start of training.
LIGHT_A = CapLight("A_old", (0.45, 1.0, 0.20), 0.35, (6.0, 5.0, 4.0))
# Light B: the small light switched on mid-sampling.
LIGHT_B = CapLight("B_new", (-0.55, 1.0, -0.35), 0.18, (9.0, 7.0, 5.0))

NORMAL_UP = np.array([0.0, 1.0, 0.0])
ALBEDO = np.array([0.70, 0.60, 0.50])


class DirectionalFixture:
    """One shading point lit by cap lights; the guide proposes directions.

    Training draws commit importance-correct deposits to the guide (the
    learning record); evaluation draws never touch the guide and are stored
    separately (the evaluation data), so the two never mix.
    """

    def __init__(
        self,
        normal: np.ndarray = NORMAL_UP,
        albedo: np.ndarray = ALBEDO,
        *,
        eta: float = 0.15,
        lam: float = 0.6,
        seed: int = 0,
        theta_bins: int = 24,
        phi_bins: int = 48,
        corrupt: str | None = None,
        corrupt_delta: float = 0.5,
    ) -> None:
        self.normal = normal / np.linalg.norm(normal)
        self.albedo = np.asarray(albedo, dtype=np.float64)
        self.lam = float(lam)
        self.guide = DirectionGuide(
            GridSpec((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (1, 1, 1)),
            theta_bins=theta_bins, phi_bins=phi_bins, eta=eta, guide_mix=lam,
        )
        self.rng = np.random.default_rng(seed)
        self.lights: list[CapLight] = []
        self.active: list[bool] = []
        self.corrupt = corrupt
        self.corrupt_delta = float(corrupt_delta)
        self._frozen: DirectionGuide | None = None
        # Learning record vs evaluation data: strictly separated.
        self.train_deposits: list[tuple[np.ndarray, np.ndarray]] = []
        self.eval_records: list[dict[str, Any]] = []

    # ------------------------------------------------------------- lights
    def add_light(self, light: CapLight, active: bool = True) -> None:
        if not light.fully_above_horizon(self.normal):
            raise ValueError(f"cap of {light.name} crosses the horizon")
        self.lights.append(light)
        self.active.append(bool(active))

    def set_light_on(self, name: str, on: bool = True) -> None:
        for i, l in enumerate(self.lights):
            if l.name == name:
                self.active[i] = bool(on)
                return
        raise KeyError(name)

    # ------------------------------------------------------------ analytic
    def radiance(self, dirs: np.ndarray, only: str | None = None) -> np.ndarray:
        out = np.zeros((dirs.shape[0], 3), dtype=np.float64)
        for l, on in zip(self.lights, self.active):
            if not on or (only is not None and l.name != only):
                continue
            out += l.radiance(dirs)
        return out

    def analytic_mean(self, normal: np.ndarray | None = None,
                      only: str | None = None) -> np.ndarray:
        n = self.normal if normal is None else normal / np.linalg.norm(normal)
        out = np.zeros(3, dtype=np.float64)
        for l, on in zip(self.lights, self.active):
            if not on or (only is not None and l.name != only):
                continue
            out += l.analytic_exitance(n, self.albedo)
        return out

    def analytic_luminance(self, only: str | None = None) -> float:
        return float(luminance(self.analytic_mean(only=only)))

    # ------------------------------------------------------------- density
    def freeze_density_snapshot(self) -> None:
        """Remember the current table; later evals may (mis)use its density."""
        self._frozen = DirectionGuide.from_snapshot(self.guide.snapshot())

    def _eval_pdf(self, samp: dict[str, np.ndarray], normals: np.ndarray) -> np.ndarray:
        if self.corrupt == "scale_pdf":
            return samp["pdf"] * (1.0 + self.corrupt_delta)
        if self.corrupt == "stale_pdf" and self._frozen is not None:
            pdf, _, _ = self._frozen.mixture_pdf(
                samp["cells"], normals, samp["dirs"], guide_mix=self.lam
            )
            return pdf
        return samp["pdf"]

    # -------------------------------------------------------------- draws
    def draw(
        self,
        n: int,
        *,
        train: bool,
        normal: np.ndarray | None = None,
        label: str = "",
    ) -> dict[str, Any]:
        """One batch of mixture samples; returns the MC estimate of L_o."""
        nrm = self.normal if normal is None else normal / np.linalg.norm(normal)
        normals = np.broadcast_to(nrm, (n, 3)).copy()
        points = np.zeros((n, 3), dtype=np.float64)
        used_version = self.guide.version
        samp = self.guide.sample_mixture(points, normals, self.rng,
                                         guide_mix=self.lam)
        dirs = samp["dirs"]
        pdf = self._eval_pdf(samp, normals)
        cos = np.clip(dirs @ nrm, 0.0, 1.0)
        fr = self.albedo / math.pi
        inc = self.radiance(dirs)
        contrib = fr[None, :] * cos[:, None] * inc / np.maximum(
            pdf, 1e-300)[:, None]
        est = contrib.mean(axis=0)
        se = contrib.std(axis=0, ddof=1) / math.sqrt(n)
        rec: dict[str, Any] = {
            "label": label,
            "n": int(n),
            "train": bool(train),
            "used_version": int(used_version),
            "mean": est,
            "se": se,
            "lum_mean": float(luminance(est)),
            "lum_se": float(luminance(contrib).std(ddof=1) / math.sqrt(n)),
        }
        if train:
            w = luminance(contrib)
            good = (w > 1e-10) & np.isfinite(w)
            flat = samp["cells"][good] * self.guide.smap.K + samp["bins"][good]
            self.guide.commit(flat, w[good])
            self.train_deposits.append((flat.copy(), w[good].copy()))
            rec["deposits"] = int(flat.size)
        else:
            comps: dict[str, Any] = {}
            for l, on in zip(self.lights, self.active):
                if not on:
                    continue
                c = fr[None, :] * cos[:, None] * l.radiance(dirs) / np.maximum(
                    pdf, 1e-300)[:, None]
                comps[l.name] = {
                    "lum_mean": float(luminance(c).mean()),
                    "lum_se": float(luminance(c).std(ddof=1) / math.sqrt(n)),
                }
            rec["components"] = comps
            self.eval_records.append(rec)
        return rec

    def train_round(self, n: int, rounds: int = 1) -> None:
        for _ in range(rounds):
            if n > 0:
                self.draw(n, train=True)

    def eval_interval(self, n: int, label: str = "") -> dict[str, Any]:
        return self.draw(n, train=False, label=label)

    # ------------------------------------------------- table + provenance
    def export_table_with_samples(self) -> dict[str, Any]:
        """The old direction table together with the samples that built it."""
        return {
            "snapshot": self.guide.snapshot(),
            "deposits": [(ids.copy(), w.copy())
                         for ids, w in self.train_deposits],
        }

    def continue_with(self, exported: dict[str, Any]) -> bool:
        """Receive an old table plus its generating samples into this sampler.

        Replays the recorded deposits into a fresh guide and verifies they
        regenerate the exported table exactly; the live guide is then
        replaced by the exported table and the learning record continues.
        """
        replay = DirectionGuide.from_snapshot(exported["snapshot"])
        replay.W.fill(0.0)
        replay._rebuild_cdf()
        for ids, w in exported["deposits"]:
            replay.commit(ids, w)
        expected = np.asarray(exported["snapshot"]["W"], dtype=np.float64)
        provenance_ok = bool(np.allclose(replay.W, expected, rtol=0, atol=1e-9))
        self.guide = DirectionGuide.from_snapshot(exported["snapshot"])
        self.train_deposits = [(ids.copy(), w.copy())
                               for ids, w in exported["deposits"]]
        return provenance_ok

    def light_support(self, light: CapLight) -> dict[str, Any]:
        """Guide pdf over the bins whose centers fall inside the light cap."""
        centers = self.guide.smap.bin_center_dirs
        inside = (centers @ light.w0) >= math.cos(light.alpha)
        if not inside.any():
            return {"bins": 0, "pdf_min": 0.0, "pdf_max": 0.0, "mass": 0.0}
        cells = np.zeros(int(inside.sum()), dtype=np.int64)
        pdf = self.guide.pdf(cells, centers[inside])
        domega = self.guide.smap.bin_domega[
            self.guide.smap.bin_ids(centers[inside])]
        return {
            "bins": int(inside.sum()),
            "pdf_min": float(pdf.min()),
            "pdf_max": float(pdf.max()),
            "mass": float((pdf * domega).sum()),
        }


# --------------------------------------------------------------------------
# Mid-sampling light-on fixture
# --------------------------------------------------------------------------
def run_mid_light(
    *,
    eta: float = 0.15,
    lam: float = 0.6,
    seed: int = 2101,
    train_rounds: int = 3,
    train_n: int = 20000,
    eval_intervals: int = 6,
    eval_n: int = 20000,
    continued_train_n: int = 4000,
    artifact_dir: str | None = None,
) -> dict[str, Any]:
    """Turn a small light on mid-sampling and check energy over intervals.

    Phase 1 trains the guide on light A only.  The old table and the samples
    that generated it are then handed to the phase-2 sampler, light B is
    switched on, and energy is checked over multiple independent evaluation
    intervals while training continues on the old table.
    """
    phase1 = DirectionalFixture(eta=eta, lam=lam, seed=seed)
    phase1.add_light(LIGHT_A, active=True)
    phase1.add_light(LIGHT_B, active=False)
    phase1.train_round(train_n, rounds=train_rounds)
    exported = phase1.export_table_with_samples()
    support = phase1.light_support(LIGHT_B)

    # Mid-sampling: the small light turns on; the old table together with
    # the samples that generated it goes back into the sampler.
    phase2 = DirectionalFixture(eta=eta, lam=lam, seed=seed + 1)
    phase2.add_light(LIGHT_A, active=True)
    phase2.add_light(LIGHT_B, active=True)
    provenance_ok = phase2.continue_with(exported)

    reference = phase2.analytic_luminance()
    ref_b = phase2.analytic_luminance(only=LIGHT_B.name)

    intervals: list[dict[str, Any]] = []
    first_anomaly: dict[str, Any] | None = None
    for k in range(eval_intervals):
        if continued_train_n:
            phase2.train_round(continued_train_n)
        rec = phase2.eval_interval(eval_n, label=f"interval{k}")
        comp = rec["components"][LIGHT_B.name]
        hw = 1.96 * comp["lum_se"]
        anomalous = not (comp["lum_mean"] - hw <= ref_b <= comp["lum_mean"] + hw)
        intervals.append({
            "label": rec["label"],
            "n": rec["n"],
            "lum_mean": rec["lum_mean"],
            "lum_se": rec["lum_se"],
            "component_mean": comp["lum_mean"],
            "component_se": comp["lum_se"],
            "used_version": rec["used_version"],
        })
        if anomalous and first_anomaly is None:
            first_anomaly = {
                "light": LIGHT_B.name,
                "interval": k,
                "expected_luminance": ref_b,
                "observed_luminance": comp["lum_mean"],
                "observed_se": comp["lum_se"],
                "support_pdf_min": support["pdf_min"],
                "support_mass": support["mass"],
                "guide_version": rec["used_version"],
            }

    means = np.array([it["lum_mean"] for it in intervals])
    grand = mean_ci(means)
    covers = ci_covers(grand, reference)
    zero_support = support["pdf_min"] <= 0.0 or support["mass"] <= 0.0
    spread = float(means.max() - means.min())
    stable = spread / max(reference, 1e-9) < 0.03
    missing_fraction = (reference - grand["mean"]) / max(reference, 1e-9)

    verdict = "rebuild_required" if (zero_support or not covers) else "accept"
    converged = bool(covers and not zero_support)

    result: dict[str, Any] = {
        "name": "mid_light_on",
        "eta": eta, "lam": lam,
        "reference_luminance": reference,
        "new_light_reference": ref_b,
        "support": support,
        "zero_support": bool(zero_support),
        "provenance_ok": bool(provenance_ok),
        "intervals": intervals,
        "grand_ci": grand,
        "covers_reference": bool(covers),
        "stable_across_intervals": bool(stable),
        "stable_but_underestimated": bool(stable and not covers),
        "missing_energy_fraction": float(missing_fraction),
        "converged": converged,
        "verdict": verdict,
        "first_anomalous_component": first_anomaly,
        "eval_records_separate": (
            len(phase2.eval_records) == eval_intervals
            and all(not r["train"] for r in phase2.eval_records)
        ),
        "train_record_continues": (
            len(phase2.train_deposits)
            == len(exported["deposits"]) + eval_intervals * (1 if continued_train_n else 0)
        ),
    }

    if artifact_dir and first_anomaly is not None:
        os.makedirs(artifact_dir, exist_ok=True)
        payload = {
            "schema": ANOMALY_SCHEMA,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "misuse": "stale_density_after_mid_sampling_light_on",
            "verdict": verdict,
            "first_anomalous_component": first_anomaly,
            "reference_luminance": reference,
            "grand_ci": {k: v for k, v in grand.items() if k != "evidence"},
        }
        path = os.path.join(artifact_dir, "first_anomalous_component.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        result["anomaly_artifact"] = path
    return result


# --------------------------------------------------------------------------
# Distribution-level checks
# --------------------------------------------------------------------------
def mixture_normalization_check(
    *, eta: float = 0.15, lam: float = 0.6, seed: int = 41,
    train_n: int = 20000, quadrature_n: int = 200000,
) -> dict[str, Any]:
    """The exact mixture pdf must integrate to 1 over the full sphere."""
    fix = DirectionalFixture(eta=eta, lam=lam, seed=seed)
    fix.add_light(LIGHT_A)
    fix.add_light(LIGHT_B)
    fix.train_round(train_n)
    g = fix.guide
    rng = np.random.default_rng(seed + 7)
    dirs, _ = g.smap.sample_uniform_sphere(quadrature_n, rng)
    cells = np.zeros(quadrature_n, dtype=np.int64)
    normals = np.broadcast_to(NORMAL_UP, (quadrature_n, 3)).copy()
    p, pg, pb = g.mixture_pdf(cells, normals, dirs, guide_mix=lam)
    est = float(p.mean() * FOUR_PI)
    se = float(p.std(ddof=1) / math.sqrt(quadrature_n) * FOUR_PI)
    ci = {"mean": est, "lo": est - 1.96 * se, "hi": est + 1.96 * se,
          "n": quadrature_n, "reference": 1.0}
    ok = ci["lo"] <= 1.0 <= ci["hi"]
    return {"name": "mixture_pdf_normalizes", "ok": bool(ok),
            "mean_ci": ci, "guide_min_pdf": float(pg.min()),
            "bsdf_integral": float(pb.mean() * FOUR_PI)}


def sampling_distribution_check(
    *, eta: float = 0.15, lam: float = 0.6, seed: int = 97,
    train_n: int = 20000, n: int = 300000,
) -> dict[str, Any]:
    """Chi-square test: sampled bins vs exact mixture bin masses."""
    fix = DirectionalFixture(eta=eta, lam=lam, seed=seed)
    fix.add_light(LIGHT_A)
    fix.add_light(LIGHT_B)
    fix.train_round(train_n)
    g = fix.guide
    normals = np.broadcast_to(NORMAL_UP, (n, 3)).copy()
    samp = g.sample_mixture(np.zeros((n, 3)), normals, fix.rng, guide_mix=lam)
    observed = np.bincount(samp["bins"], minlength=g.smap.K).astype(np.float64)

    dist = g.cell_distribution(0)
    g_mass = np.asarray(dist["g_mass"], dtype=np.float64)
    # Exact BSDF (cosine) pdf mass per bin for normal +Y; the pdf is
    # max(cos,0)/pi, zero on the lower hemisphere:
    #   integral_bin cos/pi dW = dphi * (sin^2(th_hi) - sin^2(th_lo)) / (2*pi)
    # with theta clipped at the horizon pi/2.
    smap = g.smap
    dphi = smap.phi_edges[1:] - smap.phi_edges[:-1]
    sin2 = np.sin(np.minimum(smap.theta_edges, math.pi / 2.0)) ** 2
    cos_mass = (sin2[1:] - sin2[:-1])[:, None] * dphi[None, :] / (2.0 * math.pi)
    cos_mass = cos_mass.reshape(-1)
    expected = lam * g_mass + (1.0 - lam) * cos_mass
    total = float(expected.sum())
    chi2 = float(np.sum((observed - expected * n) ** 2 /
                        np.maximum(expected * n, 1e-9)))
    df = g.smap.K - 1
    lo = df - 4.0 * math.sqrt(2.0 * df)
    hi = df + 4.0 * math.sqrt(2.0 * df)
    ok = (abs(total - 1.0) < 1e-9) and (lo <= chi2 <= hi)
    return {"name": "sampling_distribution_chi2", "ok": bool(ok),
            "chi2": chi2, "df": int(df), "window": [lo, hi],
            "expected_mass_total": total, "n": n}


def image_sweep_check(
    *,
    width: int = 24,
    height: int = 16,
    n_per_pixel: int = 2500,
    train_n: int = 20000,
    eta: float = 0.15,
    lam: float = 0.6,
    seed: int = 555,
    tilt_x: float = 0.55,
    tilt_z: float = 0.35,
    artifact_path: str | None = None,
) -> dict[str, Any]:
    """Image test: a small image of tilted shading points vs the analytic image."""
    fix = DirectionalFixture(eta=eta, lam=lam, seed=seed)
    fix.add_light(LIGHT_A)
    fix.add_light(LIGHT_B)
    fix.train_round(train_n)

    est = np.zeros((height, width, 3))
    se = np.zeros((height, width, 3))
    ref = np.zeros((height, width, 3))
    xs = np.linspace(-tilt_x, tilt_x, width)
    zs = np.linspace(-tilt_z, tilt_z, height)
    for iy in range(height):
        for ix in range(width):
            nrm = np.array([xs[ix], 1.0, zs[iy]], dtype=np.float64)
            nrm /= np.linalg.norm(nrm)
            rec = fix.draw(n_per_pixel, train=False, normal=nrm,
                           label=f"px{ix}_{iy}")
            est[iy, ix] = rec["mean"]
            se[iy, ix] = rec["se"]
            ref[iy, ix] = fix.analytic_mean(normal=nrm)
    # Per-pixel z-scores of luminance.
    est_l = luminance(est)
    ref_l = luminance(ref)
    se_l = luminance(se)
    z = np.abs(est_l - ref_l) / np.maximum(se_l, 1e-12)
    max_z = float(z.max())
    frac_within_5 = float((z <= 5.0).mean())
    # Mean-image CI from per-pixel means (independent pixels).
    pix_means = est_l.reshape(-1)
    mean_img = float(pix_means.mean())
    se_img = float(np.sqrt(np.sum(se_l.reshape(-1) ** 2)) / pix_means.size)
    ref_img = float(ref_l.mean())
    ci = {"mean": mean_img, "lo": mean_img - 1.96 * se_img,
          "hi": mean_img + 1.96 * se_img, "n": int(pix_means.size),
          "reference": ref_img}
    ok = (frac_within_5 >= 0.99) and (ci["lo"] <= ref_img <= ci["hi"])

    if artifact_path:
        from .charts import save_panel
        from .imaging import to_uint8, tonemap
        e_img = to_uint8(tonemap(est, 1.0))
        r_img = to_uint8(tonemap(ref, 1.0))
        err = np.clip(np.abs(est_l - ref_l) / np.maximum(ref_l, 1e-6) * 4.0, 0, 1)
        err_rgb = to_uint8(np.stack([err, 1.0 - err, np.zeros_like(err)], -1))
        save_panel(artifact_path, [
            (r_img, "analytic reference"),
            (e_img, "mixture estimate"),
            (err_rgb, "relative error x4"),
        ], ncols=3)

    return {"name": "image_sweep_means", "ok": bool(ok),
            "width": width, "height": height, "n_per_pixel": n_per_pixel,
            "max_abs_z": max_z, "frac_pixels_within_5sigma": frac_within_5,
            "mean_image_ci": ci, "artifact": artifact_path}


# --------------------------------------------------------------------------
# Bias / variance study over independent trials
# --------------------------------------------------------------------------
def bias_variance_study(
    *,
    n_trials: int = 10,
    train_n: int = 8000,
    eval_n: int = 12000,
    eta: float = 0.15,
    lam: float = 0.6,
    seed0: int = 9000,
    corrupt: str | None = None,
) -> dict[str, Any]:
    """Fixed-budget independent trials with separate bias/variance assertions."""
    reference = None
    trial_means: list[float] = []
    trial_ses: list[float] = []
    for r in range(n_trials):
        fix = DirectionalFixture(eta=eta, lam=lam, seed=seed0 + 101 * r,
                                 corrupt=corrupt)
        fix.add_light(LIGHT_A)
        fix.add_light(LIGHT_B)
        if corrupt == "stale_pdf":
            # Misuse of an old density: the estimator divides by the pdf of
            # the table as it was *before* training.
            fix.freeze_density_snapshot()
        fix.train_round(train_n)
        rec = fix.eval_interval(eval_n, label=f"trial{r}")
        if reference is None:
            reference = fix.analytic_luminance()
        trial_means.append(rec["lum_mean"])
        trial_ses.append(rec["lum_se"])
    assert reference is not None
    means = np.asarray(trial_means)
    grand = mean_ci(means)
    bias_ok = ci_covers(grand, reference)
    emp_var = float(means.var(ddof=1)) if n_trials > 1 else 0.0
    pred_var = float(np.mean(np.asarray(trial_ses) ** 2))
    var_ratio = emp_var / max(pred_var, 1e-300)
    var_ok = bool(0.4 <= var_ratio <= 2.2)
    return {
        "name": "bias_variance",
        "corrupt": corrupt,
        "n_trials": n_trials,
        "train_n": train_n,
        "eval_n": eval_n,
        "reference_luminance": reference,
        "mean_ci": grand,
        "bias_ok": bool(bias_ok),
        "empirical_var": emp_var,
        "predicted_var": pred_var,
        "variance_ratio": var_ratio,
        "variance_ok": var_ok,
        "ok": bool(bias_ok and var_ok),
    }


def training_size_invariance(
    *,
    train_sizes: tuple[int, ...] = (0, 1500, 15000),
    n_trials: int = 6,
    eval_n: int = 10000,
    eta: float = 0.15,
    lam: float = 0.6,
    seed0: int = 12000,
) -> dict[str, Any]:
    """Changing the training sample size must not systematically darken."""
    studies = []
    refs = []
    for i, tn in enumerate(train_sizes):
        st = bias_variance_study(n_trials=n_trials, train_n=tn,
                                 eval_n=eval_n, eta=eta, lam=lam,
                                 seed0=seed0 + 1000 * i)
        studies.append(st)
        refs.append(st["reference_luminance"])
    ref = refs[0]
    all_cover = all(ci_covers(st["mean_ci"], ref) for st in studies)
    max_rel_dev = max(abs(st["mean_ci"]["mean"] - ref) / ref for st in studies)
    ok = all_cover and max_rel_dev < 0.05
    return {"name": "training_size_invariance", "ok": bool(ok),
            "train_sizes": list(train_sizes), "reference_luminance": ref,
            "all_cis_cover": bool(all_cover),
            "max_relative_deviation": float(max_rel_dev),
            "studies": studies}


def unguided_cross_check(
    *,
    n_rounds: int = 6,
    train_n: int = 8000,
    eval_n: int = 12000,
    eta: float = 0.15,
    lam: float = 0.6,
    seed0: int = 20000,
) -> dict[str, Any]:
    """Multi-round interval check against an independent unguided estimator."""
    reference = None
    diffs: list[float] = []
    g_vars: list[float] = []
    u_vars: list[float] = []
    for r in range(n_rounds):
        guided = DirectionalFixture(eta=eta, lam=lam, seed=seed0 + 31 * r)
        guided.add_light(LIGHT_A)
        guided.add_light(LIGHT_B)
        guided.train_round(train_n)
        rec_g = guided.eval_interval(eval_n, label=f"guided{r}")
        plain = DirectionalFixture(eta=eta, lam=0.0, seed=seed0 + 1000 + 31 * r)
        plain.add_light(LIGHT_A)
        plain.add_light(LIGHT_B)
        rec_u = plain.eval_interval(eval_n, label=f"unguided{r}")
        if reference is None:
            reference = guided.analytic_luminance()
        diffs.append(rec_g["lum_mean"] - rec_u["lum_mean"])
        g_vars.append(rec_g["lum_se"] ** 2)
        u_vars.append(rec_u["lum_se"] ** 2)
    assert reference is not None
    diff_ci = mean_ci(np.asarray(diffs))
    agrees = ci_covers(diff_ci, 0.0)
    var_ratio = float(np.mean(g_vars) / max(np.mean(u_vars), 1e-300))
    ok = bool(agrees and var_ratio < 1.0)
    return {"name": "unguided_cross_check", "ok": ok,
            "n_rounds": n_rounds, "reference_luminance": reference,
            "difference_ci": diff_ci, "agrees_with_unguided": bool(agrees),
            "guided_over_unguided_var": var_ratio}


# --------------------------------------------------------------------------
# Full suite
# --------------------------------------------------------------------------
def run_quick_suite(
    artifact_dir: str | None = None,
    *,
    quick: bool = True,
) -> dict[str, Any]:
    """Run the acceptance set and return a JSON-serializable report."""
    t0 = time.perf_counter()
    if quick:
        dist_n, quad_n = 150000, 100000
        img_kw = dict(width=20, height=12, n_per_pixel=1500, train_n=8000)
        mid_kw = dict(train_rounds=2, train_n=10000, eval_intervals=5,
                      eval_n=10000, continued_train_n=2000)
        bv_kw = dict(n_trials=6, train_n=4000, eval_n=8000)
        inv_kw = dict(train_sizes=(0, 1500, 12000), n_trials=5, eval_n=6000)
        ug_kw = dict(n_rounds=5, train_n=4000, eval_n=8000)
    else:
        dist_n, quad_n = 300000, 200000
        img_kw = dict(width=24, height=16, n_per_pixel=2500, train_n=20000)
        mid_kw = dict(train_rounds=3, train_n=20000, eval_intervals=6,
                      eval_n=20000, continued_train_n=4000)
        bv_kw = dict(n_trials=10, train_n=8000, eval_n=12000)
        inv_kw = dict(train_sizes=(0, 1500, 15000), n_trials=6, eval_n=10000)
        ug_kw = dict(n_rounds=6, train_n=8000, eval_n=12000)

    checks: list[dict[str, Any]] = []
    checks.append(mixture_normalization_check(quadrature_n=quad_n))
    checks.append(sampling_distribution_check(n=dist_n))
    img_art = (os.path.join(artifact_dir, "image_sweep.png")
               if artifact_dir else None)
    checks.append(image_sweep_check(artifact_path=img_art, **img_kw))

    mid_ok = run_mid_light(artifact_dir=artifact_dir, **mid_kw)
    checks.append({
        "name": "mid_light_energy_recovered",
        "ok": bool(mid_ok["verdict"] == "accept"
                   and mid_ok["provenance_ok"]
                   and mid_ok["eval_records_separate"]
                   and mid_ok["train_record_continues"]),
        "verdict": mid_ok["verdict"],
        "grand_ci": mid_ok["grand_ci"],
        "reference_luminance": mid_ok["reference_luminance"],
        "support": mid_ok["support"],
        "missing_energy_fraction": mid_ok["missing_energy_fraction"],
        "detail": mid_ok,
    })

    mid_zero = run_mid_light(eta=0.0, lam=1.0, artifact_dir=artifact_dir,
                             **mid_kw)
    checks.append({
        "name": "zero_support_requires_rebuild",
        "ok": bool(mid_zero["verdict"] == "rebuild_required"
                   and not mid_zero["converged"]
                   and mid_zero["stable_but_underestimated"]
                   and mid_zero["missing_energy_fraction"] > 0.15
                   and mid_zero["first_anomalous_component"] is not None
                   and mid_zero["first_anomalous_component"]["light"] == "B_new"),
        "verdict": mid_zero["verdict"],
        "converged": mid_zero["converged"],
        "stable_but_underestimated": mid_zero["stable_but_underestimated"],
        "missing_energy_fraction": mid_zero["missing_energy_fraction"],
        "first_anomalous_component": mid_zero["first_anomalous_component"],
        "anomaly_artifact": mid_zero.get("anomaly_artifact"),
        "detail": mid_zero,
    })

    bv = bias_variance_study(**bv_kw)
    bv["name"] = "bias_variance_clean"
    checks.append(bv)

    bv_stale = bias_variance_study(corrupt="stale_pdf", **bv_kw)
    checks.append({
        "name": "injected_stale_density_caught",
        "ok": not bv_stale["bias_ok"],
        "mean_ci": bv_stale["mean_ci"],
        "reference_luminance": bv_stale["reference_luminance"],
        "bias_ok": bv_stale["bias_ok"],
        "note": "corrupted run must NOT pass the bias assertion",
    })
    bv_scaled = bias_variance_study(corrupt="scale_pdf", **bv_kw)
    checks.append({
        "name": "injected_scaled_density_caught",
        "ok": not bv_scaled["bias_ok"],
        "mean_ci": bv_scaled["mean_ci"],
        "reference_luminance": bv_scaled["reference_luminance"],
        "bias_ok": bv_scaled["bias_ok"],
        "note": "corrupted run must NOT pass the bias assertion",
    })

    checks.append(training_size_invariance(**inv_kw))
    checks.append(unguided_cross_check(**ug_kw))

    all_ok = all(bool(c.get("ok")) for c in checks)
    return {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "quick": quick,
        "all_ok": bool(all_ok),
        "checks": checks,
    }


if __name__ == "__main__":
    out = run_quick_suite(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "outputs", "acceptance"))
    print(json.dumps({k: v for k, v in out.items() if k != "checks"},
                     ensure_ascii=False, indent=2))
    for c in out["checks"]:
        print(f"  [{'OK' if c.get('ok') else 'FAIL'}] {c['name']}")
