"""Static figures for the delivered image set (matplotlib Agg)."""

from __future__ import annotations

import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Prefer a CJK-capable font if the environment happens to have one.
for cand in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
             "Microsoft YaHei", "PingFang SC", "Arial Unicode MS"):
    try:
        from matplotlib import font_manager
        if any(cand in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = cand
            break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False


def save_panel(path: str, tiles: list[tuple[np.ndarray, str]], ncols: int = 3,
               figsize_each: float = 3.4) -> str:
    n = len(tiles)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_each * ncols, figsize_each * 0.78 * nrows),
        dpi=130,
    )
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, (img, title) in zip(axes, tiles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes[len(tiles):]:
        ax.axis("off")
    fig.tight_layout(pad=0.6)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_convergence_chart(
    path: str,
    title: str,
    series: dict[str, dict[str, Any]],
    x: list[int],
) -> str:
    """series[name] = {'rmse_mean': [...], 'rmse_std': [...], 'rel_mean': [...]}"""
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), dpi=130)
    colors = {"BSDF only": "#8a94a6", "Guided mixture": "#2f81f7",
              "guide": "#2f81f7", "bsdf": "#8a94a6"}
    for name, data in series.items():
        c = colors.get(name)
        m = np.asarray(data["rmse_mean"])
        s = np.asarray(data.get("rmse_std", np.zeros_like(m)))
        ax[0].plot(x[: len(m)], m, label=name, color=c, lw=2)
        ax[0].fill_between(x[: len(m)], np.maximum(m - s, 0), m + s,
                           color=c, alpha=0.15)
        if "rel_mean" in data:
            rm = np.asarray(data["rel_mean"])
            ax[1].plot(x[: len(rm)], rm, label=name, color=c, lw=2)
    ax[0].set_title("Display RMSE vs spp")
    ax[0].set_xlabel("spp"); ax[0].set_ylabel("RMSE"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[1].set_yscale("log")
    ax[1].set_title("HDR relative MSE vs spp")
    ax[1].set_xlabel("spp"); ax[1].set_ylabel("relMSE (log)"); ax[1].grid(alpha=.3, which="both")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_move_chart(
    path: str,
    title: str,
    curves: dict[str, dict[str, list[float]]],
) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), dpi=130)
    palette = {"stale": "#e0564b", "decay": "#f0b350",
               "rebuild": "#46d19a", "bsdf": "#8a94a6"}
    labels = {"stale": "keep stale table", "decay": "exponential decay",
              "rebuild": "clear & rebuild", "bsdf": "BSDF only"}
    for pol, c in curves.items():
        x = c["spp"]
        m = np.asarray(c["rmse_display_mean"])
        s = np.asarray(c.get("rmse_display_std", np.zeros_like(m)))
        ax[0].plot(x, m, lw=2, color=palette.get(pol), label=labels.get(pol, pol))
        ax[0].fill_between(x, np.maximum(m - s, 0), m + s,
                           color=palette.get(pol), alpha=.15)
        rm = np.asarray(c["rel_mse_mean"])
        ax[1].plot(x, rm, lw=2, color=palette.get(pol), label=labels.get(pol, pol))
    ax[0].set_title("Post-move display RMSE")
    ax[0].set_xlabel("spp after light move"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[1].set_yscale("log")
    ax[1].set_title("Post-move relMSE (log)")
    ax[1].set_xlabel("spp after light move"); ax[1].grid(alpha=.3, which="both")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_distribution_figure(path: str, dist: dict[str, Any], title: str) -> str:
    T, P = dist["theta_bins"], dist["phi_bins"]
    mass = np.asarray(dist["g_mass"], float).reshape(T, P)
    domega = np.asarray(dist["bin_domega"], float)
    dens = mass / domega.reshape(T, P)
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=140)
    theta_edges = np.linspace(0, np.pi, T + 1)
    phi_edges = np.linspace(0, 2 * np.pi, P + 1)
    logd = np.log10(np.maximum(dens, dist["floor_pdf"]))
    im = ax.pcolormesh(phi_edges, theta_edges, logd, cmap="inferno", shading="flat")
    ax.set_ylim(np.pi, 0)
    ax.set_xlabel("phi (azimuth)")
    ax.set_ylabel("theta (polar from +Y)")
    ax.set_title(title, fontsize=11)
    n = dist.get("normal")
    if n:
        # Sweep the great circle n·w=0 through an orthonormal basis; this stays
        # correct for horizontal normals where theta(phi) degenerates.
        nn = np.asarray(n, dtype=float)
        nn = nn / max(np.linalg.norm(nn), 1e-12)
        helper = np.array([0.0, 1.0, 0.0]) if abs(nn[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(nn, helper)
        u /= max(np.linalg.norm(u), 1e-12)
        v = np.cross(nn, u)
        psi = np.linspace(0.0, 2.0 * np.pi, 481)
        w = (np.cos(psi)[:, None] * u[None, :]
             + np.sin(psi)[:, None] * v[None, :])
        th = np.arccos(np.clip(w[:, 1], -1.0, 1.0))
        ph = np.mod(np.arctan2(w[:, 2], w[:, 0]), 2.0 * np.pi)
        # Break the line at phi wrap / pole jumps.
        jump = np.abs(np.diff(ph)) > np.pi / 2
        ph_plot = ph.astype(float).copy()
        th_plot = th.astype(float).copy()
        ph_plot[1:][jump] = np.nan
        th_plot[1:][jump] = np.nan
        ax.plot(ph_plot, th_plot, color="white", lw=1.8, label="hemisphere horizon")
        thN = np.arccos(np.clip(n[1], -1, 1))
        phN = np.arctan2(n[2], n[0]) % (2 * np.pi)
        ax.plot([phN], [thN], marker="*", color="#7dffd8", ms=14,
                linestyle="none", label="shading normal")
        ax.legend(loc="upper right", fontsize=8)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("log10 guide density g [sr^-1]")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
