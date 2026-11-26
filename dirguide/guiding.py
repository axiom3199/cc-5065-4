"""Spatial direction-guiding structure.

The guide is a coarse 3D grid of spatial regions.  Every cell owns a
piecewise-constant distribution over the full sphere, parameterized as a
longitude/latitude histogram.

Correctness contract
---------------------
* Training observations only update the weight table ``W``; they are never
  accumulated as radiance/energy into any image.
* :meth:`DirectionGuide.pdf` is the single source of truth for the guide
  density ``g``.  :meth:`DirectionGuide.sample_mixture` draws from the true
  mixture ``p = lam*g + (1-lam)*p_bsdf`` and returns that exact pdf, so the
  integrator can divide by the real mixture probability.
* A uniform floor of mass ``eta`` guarantees ``g >= eta/(4*pi)`` in every
  direction, including bins never observed.  The BSDF component of the
  outer mixture independently guarantees support on the local hemisphere.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

TWO_PI = 2.0 * math.pi
FOUR_PI = 4.0 * math.pi


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luminance, works on (..., 3) arrays."""
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


@dataclass(frozen=True)
class GridSpec:
    bmin: tuple[float, float, float]
    bmax: tuple[float, float, float]
    dims: tuple[int, int, int]

    @property
    def cells(self) -> int:
        return self.dims[0] * self.dims[1] * self.dims[2]

    @property
    def cell_size(self) -> np.ndarray:
        lo = np.asarray(self.bmin, dtype=np.float64)
        hi = np.asarray(self.bmax, dtype=np.float64)
        d = np.asarray(self.dims, dtype=np.float64)
        return (hi - lo) / d


class SphericalMap:
    """Longitude/latitude bin mapping (polar angle from +Y)."""

    def __init__(self, theta_bins: int = 12, phi_bins: int = 24) -> None:
        self.T = int(theta_bins)
        self.P = int(phi_bins)
        self.K = self.T * self.P
        self.theta_edges = np.linspace(0.0, math.pi, self.T + 1)
        self.phi_edges = np.linspace(0.0, TWO_PI, self.P + 1)
        # Solid angle of every theta strip: dOmega = dphi * (cos_lo - cos_hi)
        cos_lo = np.cos(self.theta_edges[:-1])
        cos_hi = np.cos(self.theta_edges[1:])
        self.theta_domega = TWO_PI / self.P * (cos_lo - cos_hi)  # (T,)
        self.bin_domega = np.repeat(self.theta_domega, self.P)  # (K,)
        # Bin center directions (K, 3)
        tc = 0.5 * (self.theta_edges[:-1] + self.theta_edges[1:])
        pc = 0.5 * (self.phi_edges[:-1] + self.phi_edges[1:])
        tt, pp = np.meshgrid(tc, pc, indexing="ij")
        st = np.sin(tt)
        centers = np.stack(
            [st * np.cos(pp), np.cos(tt), st * np.sin(pp)], axis=-1
        ).reshape(-1, 3)
        self.bin_center_dirs = centers
        # CDF over bins proportional to solid angle (uniform sphere pick)
        area_mass = self.bin_domega / FOUR_PI
        self.area_cdf = np.cumsum(area_mass)
        self.area_cdf[-1] = 1.0

    def bin_ids(self, dirs: np.ndarray) -> np.ndarray:
        """Map (N,3) unit directions to flat bin ids."""
        d = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
        theta = np.arccos(np.clip(d[:, 1], -1.0, 1.0))
        phi = np.arctan2(d[:, 2], d[:, 0])
        phi = np.where(phi < 0.0, phi + TWO_PI, phi)
        it = np.clip((theta / math.pi * self.T).astype(np.int64), 0, self.T - 1)
        ip = np.clip((phi / TWO_PI * self.P).astype(np.int64), 0, self.P - 1)
        return it * self.P + ip

    def sample_uniform_sphere(
        self, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return directions and their (uniform) pdf = 1/4pi."""
        z = rng.random(n) * 2.0 - 1.0
        phi = rng.random(n) * TWO_PI
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        dirs = np.stack([r * np.cos(phi), z, r * np.sin(phi)], axis=1)
        pdf = np.full(n, 1.0 / FOUR_PI)
        return dirs, pdf

    def sample_in_bins(
        self, bin_ids: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Uniform-solid-angle sample inside each requested bin."""
        flat = np.asarray(bin_ids, dtype=np.int64)
        it = flat // self.P
        ip = flat % self.P
        u1 = rng.random(flat.shape)
        u2 = rng.random(flat.shape)
        cos_lo = np.cos(self.theta_edges[it])
        cos_hi = np.cos(self.theta_edges[it + 1])
        # cos(theta) uniform between cos_hi and cos_lo for equal area
        cos_t = cos_hi + u1 * (cos_lo - cos_hi)
        cos_t = np.clip(cos_t, -1.0, 1.0)
        sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t * cos_t))
        phi = self.phi_edges[ip] + u2 * (
            self.phi_edges[ip + 1] - self.phi_edges[ip]
        )
        return np.stack(
            [sin_t * np.cos(phi), cos_t, sin_t * np.sin(phi)], axis=1
        )


class DirectionGuide:
    """Grid of per-cell spherical direction histograms."""

    SCHEMA = "dirguide.distribution.v1"

    def __init__(
        self,
        grid: GridSpec,
        theta_bins: int = 12,
        phi_bins: int = 24,
        eta: float = 0.2,
        guide_mix: float = 0.6,
    ) -> None:
        self.grid = grid
        self.smap = SphericalMap(theta_bins, phi_bins)
        self.eta = float(np.clip(eta, 0.0, 0.999))
        self.guide_mix = float(np.clip(guide_mix, 0.0, 1.0))
        # Raw observed contribution weights, never radiance caches.
        self.W = np.zeros((grid.cells, self.smap.K), dtype=np.float64)
        self.cell_cdf = np.broadcast_to(
            self.smap.area_cdf, (grid.cells, self.smap.K)
        ).copy()
        self.version = 0
        self.spp = 0
        self.light_generation = 0
        self.created_at = time.time()
        self.n_deposits = 0
        self.last_decay_factor = 1.0

    # ---------------------------------------------------------------- grid
    def locate(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        lo = np.asarray(self.grid.bmin, dtype=np.float64)
        hi = np.asarray(self.grid.bmax, dtype=np.float64)
        dims = np.asarray(self.grid.dims, dtype=np.int64)
        u = (p - lo) / (hi - lo)
        ix = np.clip((u[:, 0] * dims[0]).astype(np.int64), 0, dims[0] - 1)
        iy = np.clip((u[:, 1] * dims[1]).astype(np.int64), 0, dims[1] - 1)
        iz = np.clip((u[:, 2] * dims[2]).astype(np.int64), 0, dims[2] - 1)
        return ix * dims[1] * dims[2] + iy * dims[2] + iz

    def cell_ijk(self, cell_id: int) -> tuple[int, int, int]:
        nx, ny, nz = self.grid.dims
        iz = cell_id % nz
        iy = (cell_id // nz) % ny
        ix = cell_id // (ny * nz)
        return int(ix), int(iy), int(iz)

    def cell_center(self, cell_id: int) -> np.ndarray:
        ix, iy, iz = self.cell_ijk(cell_id)
        lo = np.asarray(self.grid.bmin)
        cs = self.grid.cell_size
        return lo + (np.array([ix, iy, iz]) + 0.5) * cs

    # ------------------------------------------------------------- updates
    def _rebuild_cdf(self) -> None:
        wsum = self.W.sum(axis=1)
        nonempty = wsum > 0.0
        cdf = np.empty_like(self.W)
        if nonempty.any():
            ne = np.flatnonzero(nonempty)
            c = np.cumsum(self.W[ne], axis=1)
            c /= c[:, -1:]
            cdf[ne] = c
        empty = ~nonempty
        if empty.any():
            cdf[empty] = self.smap.area_cdf
        self.cell_cdf = cdf

    def reset(self, *, bump_generation: bool = True) -> None:
        """Rebuild: drop every learned observation (uniform floor remains)."""
        self.W.fill(0.0)
        self._rebuild_cdf()
        self.version += 1
        if bump_generation:
            self.light_generation += 1

    def on_light_moved(self, policy: str, move_retain: float = 0.25) -> str:
        """Apply the requested staleness policy after a light move.

        Returns a human-readable description of what happened.
        """
        self.light_generation += 1
        if policy == "rebuild":
            self.W.fill(0.0)
            self._rebuild_cdf()
            self.version += 1
            return "rebuild: 已清空全部方向表，从均匀底重新学习"
        if policy == "decay":
            f = float(np.clip(move_retain, 0.0, 1.0))
            self.W *= f
            self._rebuild_cdf()
            self.version += 1
            self.last_decay_factor = f
            return f"decay: 光源移动，旧表权重一次性衰减并保留 {f:.0%}"
        # stale / guide / bsdf: keep the table exactly as-is
        self.version += 1
        return "stale: 保留旧方向表（可能误导新光路）"

    def commit(
        self,
        flat_ids: np.ndarray,
        weights: np.ndarray,
        *,
        decay_factor: float = 1.0,
    ) -> int:
        """Atomically apply per-spp decay then add new observations.

        ``flat_ids`` are ``cell*K + bin`` indices.  Training data lands only
        in ``W``; nothing here touches image/radiance accumulators.
        """
        if decay_factor < 0.999999:
            self.W *= decay_factor
            self.last_decay_factor = decay_factor
        if flat_ids.size:
            sums = np.bincount(
                flat_ids, weights=weights, minlength=self.W.size
            )
            self.W += sums.reshape(self.W.shape)
            self.n_deposits += int(flat_ids.size)
        self._rebuild_cdf()
        self.spp += 1
        self.version += 1
        return self.version

    # ------------------------------------------------------------ pdf/eval
    def _learned_density(
        self, cell_ids: np.ndarray, bin_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (learned density l, Wsum) for (cell, bin) queries."""
        c = np.asarray(cell_ids, dtype=np.int64)
        b = np.asarray(bin_ids, dtype=np.int64)
        w_bin = self.W[c, b]
        w_sum = self.W[c].sum(axis=1)
        domega = self.smap.bin_domega[b]
        learned = np.where(
            w_sum[:, None] > 0.0,
            w_bin[:, None] / np.maximum(w_sum[:, None], 1e-300) / domega[:, None],
            1.0 / FOUR_PI,
        ).reshape(-1)
        return learned, w_sum

    def pdf(self, points_or_cells: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """Guide density g(omega) in solid-angle measure."""
        arr = np.asarray(points_or_cells)
        if arr.ndim == 2:
            cells = self.locate(arr)
        else:
            cells = arr.astype(np.int64)
        d = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
        bins = self.smap.bin_ids(d)
        learned, _ = self._learned_density(cells, bins)
        return (1.0 - self.eta) * learned + self.eta / FOUR_PI

    def cell_distribution(self, cell_id: int) -> dict[str, Any]:
        """Inspectable distribution summary for one region."""
        w = self.W[int(cell_id)]
        wsum = float(w.sum())
        if wsum > 0.0:
            learned_mass = w / wsum
            learned_density = learned_mass / self.smap.bin_domega
        else:
            learned_mass = np.zeros_like(w)
            learned_density = np.full_like(w, 1.0 / FOUR_PI)
        g_density = (1.0 - self.eta) * learned_density + self.eta / FOUR_PI
        # Actual probability mass of g inside each bin (density * dOmega)
        g_mass = g_density * self.smap.bin_domega
        entropy = float(
            -np.sum(np.where(g_mass > 0, g_mass * np.log(g_mass + 1e-300), 0.0))
        )
        entropy_norm = entropy / math.log(self.smap.K)
        kl_uniform = float(
            np.sum(
                np.where(
                    g_mass > 0,
                    g_mass * np.log(g_mass * self.smap.K + 1e-300),
                    0.0,
                )
            )
        )
        return {
            "cell": int(cell_id),
            "ijk": self.cell_ijk(int(cell_id)),
            "center": [float(v) for v in self.cell_center(int(cell_id))],
            "w_sum": wsum,
            "version": self.version,
            "light_generation": self.light_generation,
            "eta": self.eta,
            "entropy": entropy,
            "entropy_norm": entropy_norm,
            "kl_uniform": kl_uniform,
            "min_pdf": float(g_density.min()),
            "floor_pdf": self.eta / FOUR_PI,
            "raw_weights": w.tolist(),
            "learned_mass": learned_mass.tolist(),
            "g_mass": g_mass.tolist(),
            "theta_bins": self.smap.T,
            "phi_bins": self.smap.P,
            "bin_domega": self.smap.bin_domega.tolist(),
            "bin_centers": self.smap.bin_center_dirs.tolist(),
        }

    # ----------------------------------------------------------- sampling
    def sample_guide(
        self, cells: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample g exactly; return directions and evaluated g pdf."""
        cells = np.asarray(cells, dtype=np.int64)
        n = cells.shape[0]
        use_uniform = rng.random(n) < self.eta
        dirs = np.zeros((n, 3), dtype=np.float64)

        # uniform-floor component
        if use_uniform.any():
            idx = np.flatnonzero(use_uniform)
            u = rng.random(idx.shape)
            area_bin = np.searchsorted(self.smap.area_cdf, u).astype(np.int64)
            area_bin = np.clip(area_bin, 0, self.smap.K - 1)
            dirs[idx] = self.smap.sample_in_bins(area_bin, rng)

        # learned component (empty cells use the uniform-sphere CDF)
        learn_idx = np.flatnonzero(~use_uniform)
        if learn_idx.size:
            cc = cells[learn_idx]
            cdf = self.cell_cdf[cc]
            u = rng.random((learn_idx.size, 1))
            bins = (cdf >= u).argmax(axis=1)
            dirs[learn_idx] = self.smap.sample_in_bins(bins, rng)

        pdf = self.pdf(cells, dirs)
        return dirs, pdf

    def cosine_hemisphere(
        self, normals: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cosine-weighted hemisphere directions, pdf = cos/pi.

        Returns world directions and pdf (0 for back-facing, never here).
        """
        n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        count = n.shape[0]
        r1 = rng.random(count)
        r2 = rng.random(count)
        r = np.sqrt(r1)
        phi = TWO_PI * r2
        local = np.stack(
            [r * np.cos(phi), np.sqrt(np.maximum(0.0, 1.0 - r1)), r * np.sin(phi)],
            axis=1,
        )
        # Build tangent frame per normal.
        up = np.where(
            np.abs(n[:, 1:2]) < 0.9,
            np.broadcast_to(np.array([0.0, 1.0, 0.0]), n.shape),
            np.broadcast_to(np.array([1.0, 0.0, 0.0]), n.shape),
        )
        t = np.cross(up, n)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        b = np.cross(n, t)
        dirs = (
            local[:, 0:1] * t
            + local[:, 1:2] * n
            + local[:, 2:3] * b
        )
        cos = np.clip((dirs * n).sum(axis=1), 0.0, 1.0)
        pdf = cos / math.pi
        return dirs, pdf

    def mixture_pdf(
        self,
        cells: np.ndarray,
        normals: np.ndarray,
        dirs: np.ndarray,
        *,
        guide_mix: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """True mixture probability and its two components."""
        lam = self.guide_mix if guide_mix is None else float(guide_mix)
        g = self.pdf(cells, dirs)
        n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        d = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
        cos = np.clip((n * d).sum(axis=1), 0.0, 1.0)
        p_b = cos / math.pi
        p = lam * g + (1.0 - lam) * p_b
        return p, g, p_b

    def sample_mixture(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        rng: np.random.Generator,
        *,
        guide_mix: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Draw from p = lam*g + (1-lam)*cosine and return exact pdfs."""
        lam = self.guide_mix if guide_mix is None else float(guide_mix)
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        cells = self.locate(p)
        count = p.shape[0]
        choose_guide = rng.random(count) < lam

        g_dirs, _ = self.sample_guide(cells, rng)
        b_dirs, _ = self.cosine_hemisphere(n, rng)
        dirs = np.where(choose_guide[:, None], g_dirs, b_dirs)

        pdf, g_pdf, b_pdf = self.mixture_pdf(
            cells, n, dirs, guide_mix=lam
        )
        bins = self.smap.bin_ids(dirs)
        return {
            "dirs": dirs,
            "pdf": pdf,
            "g_pdf": g_pdf,
            "bsdf_pdf": b_pdf,
            "cells": cells,
            "bins": bins,
            "choose_guide": choose_guide,
        }

    # ------------------------------------------------------- serialization
    def snapshot(self, label: str = "") -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "label": label,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": self.version,
            "spp": self.spp,
            "light_generation": self.light_generation,
            "eta": self.eta,
            "guide_mix": self.guide_mix,
            "theta_bins": self.smap.T,
            "phi_bins": self.smap.P,
            "grid": {
                "bmin": list(self.grid.bmin),
                "bmax": list(self.grid.bmax),
                "dims": list(self.grid.dims),
            },
            "n_deposits": self.n_deposits,
            "W": self.W.tolist(),
        }

    def save_version(self, directory: str, label: str = "") -> str:
        os.makedirs(directory, exist_ok=True)
        fname = f"guide_v{self.version:05d}_gen{self.light_generation}_{int(time.time()*1000)}.json"
        path = os.path.join(directory, fname)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.snapshot(label), fh)
        return path

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> "DirectionGuide":
        g = data["grid"]
        guide = cls(
            GridSpec(tuple(g["bmin"]), tuple(g["bmax"]), tuple(g["dims"])),  # type: ignore[arg-type]
            theta_bins=data.get("theta_bins", 12),
            phi_bins=data.get("phi_bins", 24),
            eta=data.get("eta", 0.2),
            guide_mix=data.get("guide_mix", 0.6),
        )
        guide.W = np.asarray(data["W"], dtype=np.float64).reshape(guide.W.shape)
        guide._rebuild_cdf()
        guide.version = int(data.get("version", 0))
        guide.spp = int(data.get("spp", 0))
        guide.light_generation = int(data.get("light_generation", 0))
        guide.n_deposits = int(data.get("n_deposits", 0))
        return guide
