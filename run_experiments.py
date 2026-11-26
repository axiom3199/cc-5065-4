#!/usr/bin/env python3
"""Equal-budget experiments: mean / error, light moves, coverage, figures.

Examples
--------
python3 run_experiments.py --quick      # fast smoke of the whole pipeline
python3 run_experiments.py              # full delivered study (minutes, parallel)
"""

from __future__ import annotations

import argparse
import json
import os
import time

# Must precede numpy-heavy imports in the parent too.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from dirguide.charts import (
    save_convergence_chart,
    save_distribution_figure,
    save_move_chart,
    save_panel,
)
from dirguide.experiment import (
    IMAGE_DIR,
    METRICS_DIR,
    PLACEMENT_A,
    PLACEMENT_B,
    VERSION_DIR,
    FixedJob,
    MoveJob,
    _tag,
    aggregate_curves,
    fixed_run_worker,
    move_run_worker,
    render_reference,
)
from dirguide.guiding import DirectionGuide
from dirguide.imaging import (
    error_heatmap,
    exposure_from_reference,
    save_png,
    to_uint8,
    tonemap,
)
from dirguide.scene import default_grid_spec
from dirguide.session import RenderSession
from dirguide.verify import run_all_checks

SCENE_LABEL = {"window": "小窗口集中光束（含门洞，间接光路径狭窄）",
               "uniform": "较均匀室内（宽大顶灯）"}
# Matplotlib on the render host may lack CJK fonts, so baked figure titles
# stay ASCII; the web UI uses the Chinese SCENE_LABEL via browser fonts.
SCENE_TITLE_EN = {
    "window": "Concentrated beam through small windows",
    "uniform": "Even indoor room (broad ceiling light)",
}
PROBE_PX = {"window": (132, 122), "uniform": (128, 112)}


def _web(path: str) -> str:
    rel = os.path.relpath(path, os.path.dirname(__file__) + "/outputs")
    return "/outputs/" + rel.replace(os.sep, "/")


def _pick_probe_pixel(ps: RenderSession) -> tuple[int, int]:
    """Choose a visible surface pixel whose cell has a learned, non-flat map."""
    w, h = ps.width, ps.height
    best = None
    for py in range(int(h * 0.42), h - 4, 6):
        for px in range(8, w - 8, 8):
            o, d = ps.camera.ray_for_pixel(px, py, w, h)
            hit = ps.scene.intersect(o[None, :], d[None, :])
            if not bool(hit["hit"][0]):
                continue
            cid = int(ps.guide.locate(hit["p"][0][None, :])[0])
            wsum = float(ps.guide.W[cid].sum())
            if wsum <= 1e-6:
                continue
            summary = ps.guide.cell_distribution(cid)
            score = wsum * (0.4 + max(0.0, summary["kl_uniform"]))
            if best is None or score > best[0]:
                best = (score, px, py)
    if best is not None:
        return int(best[1]), int(best[2])
    return PROBE_PX.get(ps.kind, (w // 2, int(h * 0.62)))


def _pool_map(jobs, worker, workers: int):
    if workers <= 1:
        return [worker(j) for j in jobs]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(processes=workers) as pool:
        return pool.map(worker, jobs)


def ensure_reference(kind: str, placement: str, args) -> np.ndarray:
    tag = _tag(placement)
    npy = os.path.join(METRICS_DIR, f"ref_{kind}_{tag}.npy")
    meta = os.path.join(METRICS_DIR, f"ref_{kind}_{tag}.json")
    need = True
    if os.path.exists(npy) and not args.force_ref:
        try:
            info = json.load(open(meta))
            need = info.get("spp", 0) < args.ref_spp * 0.9
        except Exception:
            need = True
    if need:
        ref = render_reference(
            kind, placement, width=args.width, height=args.height,
            total_spp=args.ref_spp, workers=args.workers,
            chunk_spp=min(32, args.ref_spp),
        )
        np.save(npy, ref)
        json.dump({"spp": args.ref_spp, "placement": placement}, open(meta, "w"))
    else:
        ref = np.load(npy)
        if ref.shape[:2] != (args.height, args.width):
            ref = render_reference(
                kind, placement, width=args.width, height=args.height,
                total_spp=args.ref_spp, workers=args.workers,
                chunk_spp=min(32, args.ref_spp))
            np.save(npy, ref)
    exp = exposure_from_reference(ref)
    save_png(os.path.join(IMAGE_DIR, f"ref_{kind}_{tag}.png"), ref, exp)
    return ref


def fixed_study(kind: str, ref: np.ndarray, args) -> dict:
    print(f"[fixed] {kind}: bsdf vs guide x{len(args.seeds)} seeds, {args.fixed_spp} spp")
    jobs = []
    for seed in args.seeds:
        for strat in ("bsdf", "guide"):
            jobs.append(FixedJob(
                kind=kind, strategy=strat, spp=args.fixed_spp, seed=seed,
                width=args.width, height=args.height,
                placement=PLACEMENT_A[kind], guide_mix=args.lam, eta=args.eta))
    results = _pool_map(jobs, fixed_run_worker, args.workers)
    by = {"bsdf": [], "guide": []}
    for r in results:
        by[r["strategy"]].append(r)

    series, rows = {}, []
    for strat, runs in by.items():
        curves = aggregate_curves(runs, "history")
        n = len(curves["spp"])
        series["BSDF only" if strat == "bsdf" else "Guided mixture"] = {
            "rmse_mean": curves["rmse_display_mean"],
            "rmse_std": curves["rmse_display_std"],
            "rel_mean": curves["rel_mse_mean"],
        }
        last = -1
        row = {"strategy": strat,
               "label": "纯 BSDF 采样" if strat == "bsdf" else "方向引导混合"}
        for ck in (8, 16, 32, 48, 64):
            idx = ck - 1
            if idx < n:
                row[f"rmse_{ck}"] = round(curves["rmse_display_mean"][idx], 5)
        row["rmse_final"] = round(curves["rmse_display_mean"][last], 5)
        row["rmse_std"] = round(curves["rmse_display_std"][last], 5)
        if "rmse_roi_mean" in curves:
            row["rmse_hard_roi"] = round(curves["rmse_roi_mean"][last], 5)
        row["rel_mse_final"] = float(f"{curves['rel_mse_mean'][last]:.3e}")
        row["mean_luma"] = round(curves["mean_luma_mean"][last], 4)
        rows.append(row)

    bsdf0, guide0 = by["bsdf"][0], by["guide"][0]
    exp = exposure_from_reference(ref)
    b_img = to_uint8(tonemap(bsdf0["hdr"], exp))
    g_img = to_uint8(tonemap(guide0["hdr"], exp))
    ref_img = to_uint8(tonemap(ref, exp))
    b_err = error_heatmap(bsdf0["hdr"], ref, exp)
    g_err = error_heatmap(guide0["hdr"], ref, exp)
    panel = os.path.join(IMAGE_DIR, f"fixed_{kind}_panel.png")
    save_panel(panel, [
        (ref_img, f"reference ({args.ref_spp} spp)"),
        (b_img, f"BSDF {args.fixed_spp} spp"),
        (g_img, f"guided {args.fixed_spp} spp"),
        (b_err, "BSDF |error| x4"),
        (g_err, "guided |error| x4"),
    ], ncols=3)
    chart = os.path.join(IMAGE_DIR, f"fixed_{kind}_convergence.png")
    save_convergence_chart(
        chart, f"{SCENE_TITLE_EN[kind]} · equal {args.fixed_spp} spp budget",
        series, aggregate_curves(by["guide"], "history")["spp"])

    # Distribution figure at an informative visible probe pixel.
    ps = RenderSession(kind, width=args.width, height=args.height)
    ps.guide.W = guide0["guide_W"].copy()
    ps.guide._rebuild_cdf()
    px, py = _pick_probe_pixel(ps)
    dist = ps.probe(px, py)
    print(f"[fixed] {kind}: probe pixel=({px},{py}) ijk={dist.get('ijk')}")
    dist_fig = os.path.join(IMAGE_DIR, f"dist_{kind}_probe.png")
    if dist.get("hit"):
        save_distribution_figure(
            dist_fig, dist,
            f"{kind}: guide density at region {dist['ijk']} (pixel {px},{py})")
        # Persist an inspectable version from the experiment too.
        ps.guide.version = int(guide0.get("guide_version", args.fixed_spp))
        ps.guide.spp = int(args.fixed_spp)
        ps.guide.n_deposits = int(guide0.get("n_deposits", 0))
        ps.guide.save_version(VERSION_DIR, label=f"exp_{kind}_guided_final")
        dist_fig_web = _web(dist_fig)
    else:
        dist_fig_web = None

    ablation = _nee_off_ablation(kind, ref, args)

    return {
        "kind": kind, "label": SCENE_LABEL[kind],
        "budget_spp": args.fixed_spp, "seeds": len(args.seeds),
        "rows": rows, "panel": _web(panel), "chart": _web(chart),
        "dist_fig": dist_fig_web, "ablation": ablation,
    }


def _nee_off_ablation(kind: str, ref: np.ndarray, args) -> dict | None:
    """NEE-off pair: directional sampling is the only way to hit the light.

    This isolates the importance-sampling change and typically shows a much
    larger guided win than the default NEE+MIS estimator, where next-event
    estimation already removes most direct-light variance.
    """
    abl_spp = int(getattr(args, "abl_spp", 0) or 0)
    if abl_spp <= 0:
        return None
    abl_lam = float(getattr(args, "abl_lam", 0.6))
    jobs = [
        FixedJob(kind=kind, strategy="bsdf", spp=abl_spp, seed=args.seeds[0],
                 width=args.width, height=args.height,
                 placement=PLACEMENT_A[kind], use_nee=False),
        FixedJob(kind=kind, strategy="guide", spp=abl_spp, seed=args.seeds[0],
                 width=args.width, height=args.height,
                 placement=PLACEMENT_A[kind], guide_mix=abl_lam, eta=args.eta,
                 use_nee=False),
    ]
    res = _pool_map(jobs, fixed_run_worker, min(args.workers, 2))
    by = {s: [r for r in res if r["strategy"] == s] for s in ("bsdf", "guide")}
    labels = {"bsdf": "纯 BSDF 采样 (NEE关)", "guide": "方向引导混合 (NEE关)"}
    series, ab_rows = {}, []
    curves_g = None
    for strat in ("bsdf", "guide"):
        curves = aggregate_curves(by[strat], "history")
        if strat == "guide":
            curves_g = curves
        series["BSDF only" if strat == "bsdf" else "Guided mixture"] = {
            "rmse_mean": curves["rmse_display_mean"],
            "rmse_std": curves["rmse_display_std"],
            "rel_mean": curves["rel_mse_mean"],
        }
        ab_rows.append({
            "strategy": strat, "label": labels[strat],
            "rmse_final": round(curves["rmse_display_mean"][-1], 5),
            "rmse_hard_roi": (round(curves["rmse_roi_mean"][-1], 5)
                              if "rmse_roi_mean" in curves else None),
            "rel_mse_final": float(f"{curves['rel_mse_mean'][-1]:.3e}"),
        })
    chart = os.path.join(IMAGE_DIR, f"ablation_{kind}_neeoff.png")
    save_convergence_chart(
        chart,
        f"{SCENE_TITLE_EN[kind]} · NEE off, directional sampling only",
        series, curves_g["spp"])
    b_rmse = ab_rows[0]["rmse_final"]
    g_rmse = ab_rows[1]["rmse_final"]
    reduction = round(100.0 * (b_rmse - g_rmse) / max(b_rmse, 1e-9), 1)
    print(f"[ablation] {kind} NEE-off {abl_spp}spp: bsdf={b_rmse} "
          f"guide={g_rmse} rmse_reduction={reduction}%")
    return {
        "spp": abl_spp, "lam": abl_lam, "rows": ab_rows,
        "rmse_reduction_pct": reduction, "chart": _web(chart),
    }


def move_study(kind: str, ref_b: np.ndarray, args) -> dict:
    print(f"[move] {kind}: stale/decay/rebuild x{len(args.seeds)} seeds")
    policies = ["bsdf", "stale", "decay", "rebuild"]
    jobs = []
    for seed in args.seeds:
        for pol in policies:
            jobs.append(MoveJob(
                kind=kind, policy=pol, train_spp=args.train_spp,
                test_spp=args.test_spp, seed=seed,
                width=args.width, height=args.height,
                guide_mix=args.lam, eta=args.eta,
                half_life=args.half_life, move_retain=args.move_retain))
    results = _pool_map(jobs, move_run_worker, args.workers)
    by: dict[str, list] = {p: [] for p in policies}
    for r in results:
        by[r["policy"]].append(r)

    curves, rows = {}, []
    for pol, runs in by.items():
        c = aggregate_curves(runs, "history_b")
        curves[pol] = c
        n = len(c["spp"])
        row = {"policy": pol}
        for ck in (8, 12, 16, 24, 32, 48):
            idx = ck - 1
            if idx < n:
                row[f"rmse_{ck}"] = round(c["rmse_display_mean"][idx], 5)
        row["rmse_final"] = round(c["rmse_display_mean"][-1], 5)
        row["rmse_std"] = round(c["rmse_display_std"][-1], 5)
        row["rel_mse_final"] = float(f"{c['rel_mse_mean'][-1]:.3e}")
        row["mean_luma"] = round(c["mean_luma_mean"][-1], 4)
        rows.append(row)

    exp = exposure_from_reference(ref_b)
    ref_img = to_uint8(tonemap(ref_b, exp))
    tiles = [(ref_img, f"reference B ({args.ref_spp} spp)")]
    for pol in ("stale", "decay", "rebuild", "bsdf"):
        rep = by[pol][0]
        img = to_uint8(tonemap(rep["hdr_b"], exp))
        tiles.append((img, f"{pol} {args.test_spp}spp"))
    for pol in ("stale", "decay", "rebuild", "bsdf"):
        rep = by[pol][0]
        tiles.append((error_heatmap(rep["hdr_b"], ref_b, exp), f"{pol} |err| x4"))
        # save one inspectable final distribution version per policy
        g = DirectionGuide(default_grid_spec(kind), eta=args.eta)
        g.W = rep["w_final"].copy()
        g._rebuild_cdf()
        g.light_generation = 1
        g.version = int(rep.get("guide_version_final", args.train_spp + args.test_spp))
        g.spp = int(rep.get("guide_spp", args.train_spp + args.test_spp))
        g.n_deposits = int(rep.get("n_deposits", 0))
        g.save_version(VERSION_DIR, label=f"exp_{kind}_move_{pol}_final")
    panel = os.path.join(IMAGE_DIR, f"move_{kind}_panel.png")
    save_panel(panel, tiles, ncols=3)
    chart = os.path.join(IMAGE_DIR, f"move_{kind}_chart.png")
    save_move_chart(chart, f"{SCENE_TITLE_EN[kind]} · after light move", curves)
    return {
        "kind": kind, "label": SCENE_LABEL[kind],
        "train_spp": args.train_spp, "test_spp": args.test_spp,
        "seeds": len(args.seeds), "rows": rows,
        "panel": _web(panel), "chart": _web(chart),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--acceptance", action="store_true",
                    help="只运行统计正确性验收套件（图形界面的辅助入口）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=160)
    ap.add_argument("--lam", type=float, default=0.35)
    ap.add_argument("--eta", type=float, default=0.15)
    ap.add_argument("--half-life", dest="half_life", type=float, default=6.0)
    ap.add_argument("--move-retain", type=float, default=0.25)
    ap.add_argument("--force-ref", action="store_true")
    args = ap.parse_args()

    if args.acceptance:
        # Auxiliary CLI entry: the same acceptance set the GUI runs lives in
        # dirguide.acceptance; results land in outputs/metrics/acceptance.json
        # and outputs/acceptance/.
        from dirguide.acceptance import run_quick_suite
        from dirguide.experiment import ACCEPTANCE_DIR
        os.makedirs(ACCEPTANCE_DIR, exist_ok=True)
        os.makedirs(METRICS_DIR, exist_ok=True)
        report = run_quick_suite(ACCEPTANCE_DIR, quick=args.quick)
        out = os.path.join(METRICS_DIR, "acceptance.json")
        json.dump(report, open(out, "w"), ensure_ascii=False, indent=2)
        print(f"acceptance all_ok={report['all_ok']} "
              f"({report['elapsed_s']}s) -> {out}")
        for c in report["checks"]:
            print(f"  [{'OK' if c.get('ok') else 'FAIL'}] {c['name']}")
        return 0 if report["all_ok"] else 1

    if args.quick:
        args.ref_spp = 128
        args.fixed_spp = 16
        args.train_spp = 6
        args.test_spp = 12
        args.abl_spp = 12
        args.seeds = [101]
        args.workers = min(args.workers, 8)
    else:
        args.ref_spp = 1024
        args.fixed_spp = 64
        args.train_spp = 24
        args.test_spp = 48
        args.abl_spp = 48
        args.seeds = [101, 202, 303]
    args.abl_lam = 0.6

    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(VERSION_DIR, exist_ok=True)
    t0 = time.perf_counter()

    print("[verify] coverage / pdf checks")
    coverage = run_all_checks()
    json.dump(coverage, open(os.path.join(METRICS_DIR, "coverage_checks.json"), "w"),
              ensure_ascii=False, indent=2)
    print("  all_ok =", coverage["all_ok"])

    fixed_blocks, move_blocks = [], []
    for kind in ("window", "uniform"):
        ref_a = ensure_reference(kind, PLACEMENT_A[kind], args)
        ref_b = ensure_reference(kind, PLACEMENT_B[kind], args)
        fixed_blocks.append(fixed_study(kind, ref_a, args))
        move_blocks.append(move_study(kind, ref_b, args))

    gallery = sorted(
        _web(os.path.join(IMAGE_DIR, f))
        for f in os.listdir(IMAGE_DIR) if f.endswith(".png")
    )
    results = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "budgets": {
            "width": args.width, "height": args.height,
            "reference_spp": args.ref_spp, "fixed_spp": args.fixed_spp,
            "train_spp": args.train_spp, "test_spp": args.test_spp,
            "seeds": args.seeds, "lam": args.lam, "eta": args.eta,
            "half_life": args.half_life, "move_retain": args.move_retain,
        },
        "coverage": coverage,
        "fixed": fixed_blocks,
        "move": move_blocks,
        "gallery": gallery,
    }
    out = os.path.join(METRICS_DIR, "results.json")
    json.dump(results, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\nDONE in {results['elapsed_s']}s -> {out}")
    for b in fixed_blocks:
        print("\nFIXED", b["kind"])
        for r in b["rows"]:
            print(" ", r)
    for b in move_blocks:
        print("\nMOVE", b["kind"])
        for r in b["rows"]:
            print(" ", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
