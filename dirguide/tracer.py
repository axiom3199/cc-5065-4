"""Vectorized indirect-direction-guided path tracer.

One call traces one sample per pixel as a batch of rays.  The estimator at
each diffuse vertex is the true mixture pdf

    p(w) = lam * g(w) + (1-lam) * p_bsdf(w)

combined with next-event estimation through the balance heuristic.

Training credit is reconstructed backwards from per-vertex records.
For sampled continuation factor f_b = fr*cos / p_b and suffix radiance L,
the importance-correct bin weight is the Monte Carlo contribution

    w_b = luminance(f_b * L) = luminance(fr*cos * L / p_b).

Depositing fr*cos*L directly would be wrong under non-uniform proposals:
samples are already distributed ~p, so an un-divided weight makes the
learned bin mass proportional to integral p*fr*cos*L and double-counts the
current proposal.  With the 1/p_b factor already inside f_b, the weighted
bin mass estimates the true target integral integral_bin fr*cos*L.  These
weights land only in the guide table; image radiance is exactly the
reconstructed path estimator.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .guiding import DirectionGuide, luminance
from .scene import Camera, Scene

OFFSET = 1.2e-3
PI = math.pi


def trace_spp(
    scene: Scene,
    camera: Camera,
    guide: DirectionGuide | None,
    rng: np.random.Generator,
    width: int,
    height: int,
    *,
    lam: float = 0.6,
    train: bool = True,
    max_bounces: int = 8,
    rr_start: int = 3,
    nee: bool = True,
) -> dict[str, Any]:
    n = width * height
    o, d = camera.generate_rays(width, height, rng)

    albedo_table = np.array([m.albedo for m in scene.materials])
    emit_table = np.array([m.emission for m in scene.materials])

    rec_p = np.zeros((n, max_bounces), dtype=np.float64)
    rec_f = np.zeros((n, max_bounces, 3), dtype=np.float64)
    rec_a = np.zeros((n, max_bounces, 3), dtype=np.float64)
    rec_cell = np.full((n, max_bounces), -1, dtype=np.int64)
    rec_bin = np.full((n, max_bounces), -1, dtype=np.int64)

    beta = np.ones((n, 3), dtype=np.float64)
    active = np.ones(n, dtype=bool)
    n_trace_rays = n
    n_shadow_rays = 0

    light = scene.light_rect
    light_area = light.area if light is not None else 1.0

    for b in range(max_bounces):
        if not active.any():
            break
        hit = scene.intersect(o, d)
        n_trace_rays += int(active.sum())
        valid_hit = active & hit["hit"]
        active = valid_hit
        if not active.any():
            break

        p_hit = hit["p"]
        n_hit = hit["n"]
        mat = np.maximum(hit["mat"], 0)
        emit_rgb = emit_table[mat]
        is_emitter = active & (emit_rgb.max(axis=1) > 0.0)

        # ---- emission hit, MIS against light-area sampling --------------
        if is_emitter.any():
            idx_l = np.flatnonzero(is_emitter)
            if b == 0 or not nee:
                # Camera-primary emission, or no light-sampling strategy in
                # use: there is no competing NEE pdf to balance against.
                w_mis = np.ones(idx_l.size)
            else:
                p_prev = rec_p[idx_l, b - 1]
                r2 = np.sum((p_hit[idx_l] - o[idx_l]) ** 2, axis=1)
                cos_l = np.abs(d[idx_l] @ light.normal)  # type: ignore[union-attr]
                p_light = r2 / light_area / np.maximum(cos_l, 1e-6)
                w_mis = p_prev / (p_light + p_prev + 1e-300)
            rec_a[idx_l, b, :] = (
                emit_rgb[idx_l] * w_mis[:, None]
            )
            active[idx_l] = False

        # ---- diffuse shading + NEE + continuation ------------------------
        shade = active
        if shade.any():
            idx = np.flatnonzero(shade)
            x = p_hit[idx]
            ns = n_hit[idx]
            alb = albedo_table[mat[idx]]

            # Next-event estimation of the area light.
            if nee and light is not None:
                ls = scene.sample_light_point(idx.size, rng)
                y = ls["p"]
                wi = y - x
                r2 = np.sum(wi * wi, axis=1)
                dist = np.sqrt(r2)
                wi /= np.maximum(dist[:, None], 1e-12)
                cos_x = np.sum(ns * wi, axis=1)
                cos_l = np.abs(np.sum(ls["n"] * (-wi), axis=1))
                p_light = scene.light_solid_angle_pdf(
                    ls["pdf_area"], r2, cos_l
                )
                shadow_o = x + ns * OFFSET
                n_shadow_rays += idx.size
                visible = ~scene.occluded(shadow_o, wi, dist)
                valid_l = visible & (cos_x > 1e-5) & (cos_l > 1e-5)

                if guide is not None and lam > 0.0:
                    cells_nee = guide.locate(x)
                    p_mix, _, _ = guide.mixture_pdf(
                        cells_nee, ns, wi, guide_mix=lam
                    )
                else:
                    p_mix = np.clip(cos_x, 0.0, None) / PI
                w_nee = p_light / (p_light + p_mix + 1e-300)
                fr = alb / PI
                term = (
                    fr
                    * np.maximum(cos_x, 0.0)[:, None]
                    * ls["emission"]
                    / np.maximum(p_light, 1e-300)[:, None]
                    * w_nee[:, None]
                )
                term *= valid_l[:, None]
                rec_a[idx, b, :] += term

            # Sample indirect continuation from the true mixture.
            if guide is not None and lam > 0.0:
                samp = guide.sample_mixture(x, ns, rng, guide_mix=lam)
                new_d = samp["dirs"]
                p_mix = samp["pdf"]
                cells = samp["cells"]
                bins = samp["bins"]
            else:
                new_d, p_bsdf = _cosine_hemisphere(ns, rng)
                p_mix = p_bsdf
                cells = (
                    guide.locate(x)
                    if guide is not None
                    else np.zeros(idx.size, dtype=np.int64)
                )
                bins = (
                    guide.smap.bin_ids(new_d)
                    if guide is not None
                    else np.zeros(idx.size, dtype=np.int64)
                )

            cos_out = np.sum(ns * new_d, axis=1)
            good = cos_out > 1e-6
            q = alb / PI * np.maximum(cos_out, 0.0)[:, None]
            f = q / np.maximum(p_mix[:, None], 1e-300)
            f *= good[:, None]

            rec_p[idx, b] = p_mix
            rec_f[idx, b, :] = f
            rec_cell[idx, b] = cells
            rec_bin[idx, b] = bins

            beta[idx] *= f
            cont = good.copy()

            # Russian roulette (unbiased via survival correction).
            if b >= rr_start and good.any():
                pr = np.clip(np.max(beta[idx], axis=1), 0.1, 1.0)
                rr_ok = rng.random(idx.size) < pr
                survive = good & rr_ok
                corr = np.where(survive, 1.0 / np.maximum(pr, 1e-3), 0.0)
                f = f * corr[:, None]
                rec_f[idx, b, :] = f
                beta[idx] *= corr[:, None]
                cont = survive

            # Advance surviving rays; others terminate this bounce.
            o[idx] = x + ns * OFFSET
            d[idx] = new_d
            active[idx] = cont

    # ---- backward reconstruction of radiance and guide credit -----------
    radiance = np.zeros((n, 3), dtype=np.float64)
    dep_c: list[np.ndarray] = []
    dep_b: list[np.ndarray] = []
    dep_w: list[np.ndarray] = []

    for b in range(max_bounces - 1, -1, -1):
        cont_rad = rec_f[:, b, :] * radiance
        if train and guide is not None:
            # Importance-corrected MC contribution t/p of this direction.
            w_credit = luminance(cont_rad)
            good_w = (rec_cell[:, b] >= 0) & (w_credit > 1e-10) & np.isfinite(w_credit)
            if good_w.any():
                dep_c.append(rec_cell[good_w, b])
                dep_b.append(rec_bin[good_w, b])
                dep_w.append(w_credit[good_w])
        radiance = rec_a[:, b, :] + cont_rad

    if dep_c:
        dep_c_arr = np.concatenate(dep_c)
        dep_b_arr = np.concatenate(dep_b)
        dep_w_arr = np.concatenate(dep_w)
        flat_ids = dep_c_arr * guide.smap.K + dep_b_arr  # type: ignore[union-attr]
    else:
        flat_ids = np.zeros(0, dtype=np.int64)
        dep_w_arr = np.zeros(0, dtype=np.float64)

    img = radiance.reshape(height, width, 3)
    stats = {
        "primary_rays": n,
        "trace_rays": n_trace_rays,
        "shadow_rays": n_shadow_rays,
        "deposits": int(dep_w_arr.size),
        "deposit_weight_sum": float(dep_w_arr.sum()) if dep_w_arr.size else 0.0,
    }
    return {
        "image": img,
        "deposit_ids": flat_ids,
        "deposit_weights": dep_w_arr,
        "stats": stats,
    }


def _cosine_hemisphere(
    normals: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Local cosine sample oriented to arbitrary normals (fallback path)."""
    count = normals.shape[0]
    n = normals
    r1 = rng.random(count)
    r2 = rng.random(count)
    r = np.sqrt(r1)
    phi = 2.0 * PI * r2
    local = np.stack(
        [r * np.cos(phi), np.sqrt(np.maximum(0.0, 1.0 - r1)), r * np.sin(phi)],
        axis=1,
    )
    up = np.where(
        np.abs(n[:, 1:2]) < 0.9,
        np.broadcast_to(np.array([0.0, 1.0, 0.0]), n.shape),
        np.broadcast_to(np.array([1.0, 0.0, 0.0]), n.shape),
    )
    t = np.cross(up, n)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    bt = np.cross(n, t)
    d = local[:, 0:1] * t + local[:, 1:2] * n + local[:, 2:3] * bt
    cos = np.clip((d * n).sum(axis=1), 0.0, 1.0)
    return d, cos / PI
