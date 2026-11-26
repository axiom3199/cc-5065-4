"""Progressive rendering session with explicit estimate/train ordering.

Each spp follows this order:

1. The current guide snapshot (version V) is used for every mixture pdf and
   proposal during tracing; the guide weights are not mutated mid-spp.
2. The unbiased path estimate is added to the image accumulator.
3. Only after tracing finishes are observed contributions committed to the
   guide, producing version V+1.

Moving the light resets the *image* accumulator (the old image estimates the
old lighting) while the guide table is kept, decayed or rebuilt according to
the selected staleness policy.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .guiding import DirectionGuide, luminance
from .imaging import image_metrics, roi_rmse
from .scene import Camera, Scene, build_scene, default_grid_spec
from .tracer import trace_spp

# Normalized image regions dominated by hard, narrow indirect paths.
HARD_ROI: dict[str, tuple[float, float, float, float]] = {
    "window": (0.10, 0.52, 0.90, 0.99),
    "uniform": (0.18, 0.45, 0.82, 0.95),
}

STRATEGY_INFO: dict[str, dict[str, Any]] = {
    "bsdf": {
        "label": "纯表面采样 (无引导)",
        "lam": 0.0,
        "train": False,
        "move_policy": "none",
        "desc": "只使用余弦加权 BSDF 采样，作为基线。",
    },
    "guide": {
        "label": "在线方向引导混合",
        "lam": None,  # use guide_mix parameter
        "train": True,
        "move_policy": "stale",
        "desc": "学习空间方向分布，并与表面采样按真实概率混合。",
    },
    "stale": {
        "label": "移灯后：保留旧表 (误导对照)",
        "lam": None,
        "train": True,
        "move_policy": "stale",
        "desc": "光源移动后旧方向权重原样保留，新路径被旧亮区误导，方差升高但估计仍无偏。",
    },
    "decay": {
        "label": "移灯后：指数衰减",
        "lam": None,
        "train": True,
        "move_policy": "decay",
        "desc": "旧观察按半衰期持续衰减，移灯时再整体保留一部分，逐步遗忘旧光源。",
    },
    "rebuild": {
        "label": "移灯后：清空重建",
        "lam": None,
        "train": True,
        "move_policy": "rebuild",
        "desc": "光源移动后清空方向表，仅保留均匀底，从新光路重新学习。",
    },
}


class RenderSession:
    def __init__(
        self,
        kind: str = "window",
        width: int = 256,
        height: int = 160,
        seed: int = 1234,
        strategy: str = "guide",
        guide_mix: float = 0.35,
        eta: float = 0.15,
        half_life: float = 6.0,
        move_retain: float = 0.25,
        max_bounces: int = 8,
        exposure: float | None = None,
        use_nee: bool = True,
    ) -> None:
        self.kind = kind
        self.width = int(width)
        self.height = int(height)
        self.seed0 = int(seed)
        self.strategy = strategy
        self.guide_mix = float(guide_mix)
        self.eta = float(eta)
        self.half_life = float(half_life)
        self.move_retain = float(move_retain)
        self.max_bounces = int(max_bounces)
        self.use_nee = bool(use_nee)

        self.scene: Scene
        self.camera: Camera
        self.scene, self.camera = build_scene(kind)
        self.guide = DirectionGuide(
            default_grid_spec(kind), eta=eta, guide_mix=guide_mix
        )
        self.exposure = (
            float(exposure)
            if exposure is not None
            else self.scene.recommended_exposure
        )

        self.rng = np.random.default_rng(self.seed0)
        self.accum = np.zeros((self.height, self.width, 3), dtype=np.float64)
        self.n_spp = 0
        self.elapsed = 0.0
        self.history: list[dict[str, Any]] = []
        self.reference: np.ndarray | None = None
        self.reference_key: str | None = None
        self.total_trace_rays = 0
        self.total_shadow_rays = 0
        self.feedback: list[dict[str, str]] = []
        self.last_move_message = ""

    # ------------------------------------------------------------- config
    def set_strategy(self, strategy: str) -> None:
        if strategy not in STRATEGY_INFO:
            raise ValueError(f"未知策略: {strategy}")
        self.strategy = strategy
        self._note("strategy", f"切换策略：{STRATEGY_INFO[strategy]['label']}")

    def update_params(self, params: dict[str, Any]) -> list[str]:
        changed = []
        for key, value in params.items():
            if key == "strategy":
                self.set_strategy(str(value))
                changed.append(key)
            elif key == "guide_mix":
                v = float(value)
                if not 0.0 <= v <= 1.0:
                    raise ValueError("引导混合比必须在 0..1")
                self.guide_mix = v
                self.guide.guide_mix = v
                changed.append(key)
            elif key == "eta":
                v = float(value)
                if not 0.01 <= v <= 0.95:
                    raise ValueError("均匀底质量 eta 必须在 0.01..0.95")
                self.eta = v
                self.guide.eta = v
                changed.append(key)
            elif key == "half_life":
                v = float(value)
                if not 0.5 <= v <= 200.0:
                    raise ValueError("半衰期必须在 0.5..200 spp")
                self.half_life = v
                changed.append(key)
            elif key == "move_retain":
                v = float(value)
                if not 0.0 <= v <= 1.0:
                    raise ValueError("移灯保留率必须在 0..1")
                self.move_retain = v
                changed.append(key)
            elif key == "exposure":
                v = float(value)
                if not 0.05 <= v <= 8.0:
                    raise ValueError("曝光必须在 0.05..8")
                self.exposure = v
                changed.append(key)
            elif key == "max_bounces":
                v = int(value)
                if not 1 <= v <= 16:
                    raise ValueError("最大反弹数必须在 1..16")
                self.max_bounces = v
                changed.append(key)
            elif key == "use_nee":
                if isinstance(value, str):
                    v = value.strip().lower() in ("1", "true", "yes", "on", "开")
                else:
                    v = bool(value)
                self.use_nee = v
                changed.append(key)
            else:
                raise KeyError(f"内核不认识控件参数: {key}")
        return changed

    def _note(self, kind: str, message: str) -> None:
        item = {"kind": kind, "message": message}
        self.feedback.append(item)
        self.feedback = self.feedback[-30:]

    # ------------------------------------------------------------ actions
    def reset_image(self) -> None:
        self.accum.fill(0.0)
        self.n_spp = 0
        self.history.clear()

    def reset_all(self) -> None:
        self.reset_image()
        self.guide = DirectionGuide(
            default_grid_spec(self.kind), eta=self.eta, guide_mix=self.guide_mix
        )
        self.rng = np.random.default_rng(self.seed0)
        self.elapsed = 0.0
        self.total_trace_rays = 0
        self.total_shadow_rays = 0
        self.last_move_message = ""
        self._note("reset", "已重置图像与方向表")

    def move_light(self, placement: str | None = None) -> str:
        order = self.scene.placement_order
        if placement is None:
            cur = self.scene.active_placement
            idx = order.index(cur) if cur in order else -1
            placement = order[(idx + 1) % len(order)]
        if placement not in self.scene.placements:
            raise ValueError(f"未知光源位置: {placement}")
        self.scene.set_light_placement(placement)

        info = STRATEGY_INFO[self.strategy]
        policy = info["move_policy"]
        if policy == "none":
            msg = "bsdf 基线：移动光源后重启估计，无方向表需要处理"
            self.guide.light_generation += 1
        else:
            apply_policy = policy if policy in ("stale", "decay", "rebuild") else "stale"
            msg = self.guide.on_light_moved(
                apply_policy, move_retain=self.move_retain
            )
        self.last_move_message = msg
        self._note("light", f"光源移动到 {placement}；{msg}")
        # Old pixels estimate the old lighting: restart estimation, but the
        # guide table state is deliberately preserved/decayed/rebuilt.
        self.reset_image()
        return msg

    # ---------------------------------------------------------- rendering
    def step_spp(self) -> dict[str, Any]:
        info = STRATEGY_INFO[self.strategy]
        lam = self.guide_mix if info["lam"] is None else 0.0
        train = bool(info["train"])

        decay_factor = 1.0
        if self.strategy == "decay":
            decay_factor = 2.0 ** (-1.0 / max(self.half_life, 1e-6))

        used_version = self.guide.version
        t0 = time.perf_counter()
        res = trace_spp(
            self.scene,
            self.camera,
            self.guide,
            self.rng,
            self.width,
            self.height,
            lam=lam,
            train=train,
            max_bounces=self.max_bounces,
            nee=self.use_nee,
        )
        dt = time.perf_counter() - t0
        self.elapsed += dt

        img = res["image"]
        # Guard against rare numerical fireflies poisoning the accumulator:
        # clamp per-sample only at a very high radiance ceiling. This is a
        # variance safeguard, not cached energy; choose ceiling huge.
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(img, 0.0, 1e5, out=img)

        self.accum += img
        self.n_spp += 1
        self.total_trace_rays += res["stats"]["trace_rays"]
        self.total_shadow_rays += res["stats"]["shadow_rays"]

        if train:
            self.guide.commit(
                res["deposit_ids"],
                res["deposit_weights"],
                decay_factor=decay_factor,
            )

        mean_hdr = self.accum / self.n_spp
        m = image_metrics(mean_hdr, self.reference, self.exposure)
        roi = roi_rmse(mean_hdr, self.reference, self.exposure,
                       HARD_ROI.get(self.kind, (0.1, 0.1, 0.9, 0.9)))
        if roi is not None:
            m["rmse_roi"] = roi
        point = {
            "spp": self.n_spp,
            "elapsed": round(self.elapsed, 3),
            "step_ms": round(dt * 1000.0, 1),
            "used_guide_version": used_version,
            "guide_version": self.guide.version,
            "light_generation": self.guide.light_generation,
            "placement": self.scene.active_placement,
            "deposits": res["stats"]["deposits"],
            **{k: v for k, v in m.items() if k != "mean_rgb"},
        }
        self.history.append(point)
        return point

    def render(self, spp: int) -> list[dict[str, Any]]:
        last = {}
        for _ in range(int(spp)):
            last = self.step_spp()
        return self.history

    # ------------------------------------------------------------ outputs
    def image_hdr(self) -> np.ndarray:
        if self.n_spp == 0:
            return np.zeros((self.height, self.width, 3))
        return self.accum / self.n_spp

    def state_summary(self) -> dict[str, Any]:
        last = self.history[-1] if self.history else None
        return {
            "kind": self.kind,
            "strategy": self.strategy,
            "strategy_label": STRATEGY_INFO[self.strategy]["label"],
            "strategy_desc": STRATEGY_INFO[self.strategy]["desc"],
            "width": self.width,
            "height": self.height,
            "spp": self.n_spp,
            "elapsed": round(self.elapsed, 3),
            "guide_version": self.guide.version,
            "used_version_last": last["used_guide_version"] if last else None,
            "light_generation": self.guide.light_generation,
            "placement": self.scene.active_placement,
            "placements": self.scene.placement_order,
            "guide_mix": self.guide_mix,
            "eta": self.eta,
            "half_life": self.half_life,
            "move_retain": self.move_retain,
            "exposure": self.exposure,
            "max_bounces": self.max_bounces,
            "use_nee": self.use_nee,
            "total_trace_rays": self.total_trace_rays,
            "total_shadow_rays": self.total_shadow_rays,
            "n_deposits": self.guide.n_deposits,
            "grid_dims": list(self.guide.grid.dims),
            "theta_bins": self.guide.smap.T,
            "phi_bins": self.guide.smap.P,
            "has_reference": self.reference is not None,
            "reference_key": self.reference_key,
            "last": last,
            "history": self.history[-600:],
            "last_move_message": self.last_move_message,
            "feedback": self.feedback[-8:],
        }

    def probe(self, px: int, py: int) -> dict[str, Any]:
        """Return the directional distribution at an image pixel's region."""
        px = int(np.clip(px, 0, self.width - 1))
        py = int(np.clip(py, 0, self.height - 1))
        o, d = self.camera.ray_for_pixel(px, py, self.width, self.height)
        hit = self.scene.intersect(o[None, :], d[None, :])
        if not bool(hit["hit"][0]):
            return {"pixel": [px, py], "hit": False}
        p = hit["p"][0]
        n = hit["n"][0]
        cell = int(self.guide.locate(p[None, :])[0])
        dist = self.guide.cell_distribution(cell)
        # Evaluate current mixture for the hit point normal across bin centers
        # so the UI can overlay the true proposal density.
        centers = self.guide.smap.bin_center_dirs
        pmix, pg, pb = self.guide.mixture_pdf(
            np.full(centers.shape[0], cell),
            np.broadcast_to(n, centers.shape),
            centers,
            guide_mix=(0.0 if self.strategy == "bsdf" else self.guide_mix),
        )
        dist["mixture_pdf_at_centers"] = pmix.tolist()
        dist["guide_pdf_at_centers"] = pg.tolist()
        dist["bsdf_pdf_at_centers"] = pb.tolist()
        dist["hit"] = True
        dist["pixel"] = [px, py]
        dist["point"] = [float(v) for v in p]
        dist["normal"] = [float(v) for v in n]
        dist["strategy_lam"] = 0.0 if self.strategy == "bsdf" else self.guide_mix
        # Coverage certificate: minimum true mixture pdf on the hemisphere.
        hemi = (centers @ n) > 0.0
        if hemi.any():
            dist["min_mixture_pdf_hemi_centers"] = float(pmix[hemi].min())
            dist["min_bsdf_pdf_hemi_centers"] = float(pb[hemi].min())
        return dist
