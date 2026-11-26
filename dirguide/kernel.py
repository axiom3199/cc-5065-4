"""Guiding kernel: single state owner, schema-driven controls, commands.

The web UI renders its controls from :meth:`Kernel.schema`; every action is
dispatched through :meth:`Kernel.command`.  Invalid commands raise
:class:`KernelError` and are recorded in the session feedback channel so the
UI can show failure inside the guiding panel.
"""

from __future__ import annotations

import io
import json
import os
import threading
from typing import Any

import numpy as np
from PIL import Image

from .experiment import ACCEPTANCE_DIR, IMAGE_DIR, METRICS_DIR, VERSION_DIR, _tag
from .guiding import DirectionGuide
from .imaging import exposure_from_reference, tonemap, to_uint8
from .session import STRATEGY_INFO, RenderSession


class KernelError(Exception):
    pass


DEFAULT_W, DEFAULT_H = 256, 160


class Kernel:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.session = RenderSession(
            "window", width=DEFAULT_W, height=DEFAULT_H, seed=1234
        )
        self._load_reference_for_current()
        self.results_cache: dict[str, Any] | None = None
        self.results_cache_mtime: float | None = None

    # ------------------------------------------------------------- schema
    def schema(self) -> dict[str, Any]:
        return {
            "title": "间接光方向引导实验台",
            "scenes": [
                {"value": "window", "label": "小窗口集中光束"},
                {"value": "uniform", "label": "较均匀室内"},
            ],
            "strategies": [
                {"value": k, "label": v["label"], "desc": v["desc"]}
                for k, v in STRATEGY_INFO.items()
            ],
            "controls": [
                {"key": "guide_mix", "type": "range", "label": "引导混合概率 λ",
                 "min": 0.0, "max": 0.95, "step": 0.05, "default": 0.35,
                 "hint": "p = λ·g + (1−λ)·p_BSDF，采样与分母使用同一个真实混合概率。"},
                {"key": "eta", "type": "range", "label": "均匀底质量 η",
                 "min": 0.02, "max": 0.8, "step": 0.01, "default": 0.15,
                 "hint": "g ≥ η/(4π)：从未观测的方向也保留支撑，早期漏采亮区不会永久丢失。"},
                {"key": "half_life", "type": "range", "label": "衰减半衰期 (spp)",
                 "min": 0.5, "max": 30.0, "step": 0.5, "default": 6.0,
                 "visible_when": {"strategy": "decay"},
                 "hint": "decay 策略下旧观察权重按 2^(−1/半衰期) 逐 spp 衰减。"},
                {"key": "move_retain", "type": "range", "label": "移灯时旧表保留率",
                 "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.25,
                 "visible_when": {"strategy": "decay"},
                 "hint": "移灯瞬间对旧方向表整体乘以此系数；0 等价于清空重建。"},
                {"key": "exposure", "type": "range", "label": "显示曝光",
                 "min": 0.1, "max": 3.5, "step": 0.05, "default": 1.0},
                {"key": "max_bounces", "type": "range", "label": "最大反弹数",
                 "min": 1, "max": 12, "step": 1, "default": 8},
                {"key": "use_nee", "type": "toggle",
                 "label": "下一事件估计 NEE（关闭可隔离观察方向采样收益）",
                 "default": True,
                 "hint": "开启：BSDF/引导与面积光采样做平衡启发式 MIS，画面更干净但引导收益较温和；关闭：方向采样是命中光源的唯一途径，引导降幅最明显，是检验重要性采样的纯净对照。"},
            ],
            "actions": [
                {"cmd": "step", "label": "推进 1 spp"},
                {"cmd": "toggle_play", "label": "连续渲染"},
                {"cmd": "move_light", "label": "移动光源 A↔B"},
                {"cmd": "reset", "label": "重置全部"},
                {"cmd": "save_version", "label": "保存分布版本"},
            ],
            "regression_cases": [
                {"case": "stale", "label": "回归样例：移灯后保留旧表",
                 "desc": "训练后移灯并保留旧方向表（含均匀底），观察能量恢复与方向分布。"},
                {"case": "decay", "label": "回归样例：移灯后指数衰减",
                 "desc": "旧观察按半衰期衰减，逐步遗忘旧光源方向。"},
                {"case": "rebuild", "label": "回归样例：移灯后清空重建",
                 "desc": "移灯即清空方向表，从均匀底重新学习新光路。"},
            ],
            "acceptance": {
                "cmd": "run_acceptance",
                "label": "运行统计正确性验收",
                "desc": "解析小场景 + 中途开灯夹具：均值置信区间、零支持反例、注入错误密度。",
            },
            "dirs": {"images": IMAGE_DIR, "metrics": METRICS_DIR,
                      "versions": VERSION_DIR, "acceptance": ACCEPTANCE_DIR},
        }

    # ----------------------------------------------------------- references
    def _reference_path(self, kind: str, placement: str | None) -> str:
        return os.path.join(
            METRICS_DIR, f"ref_{kind}_{_tag(placement)}.npy"
        )

    def _load_reference_for_current(self) -> bool:
        s = self.session
        path = self._reference_path(s.kind, s.scene.active_placement)
        if os.path.exists(path):
            ref = np.load(path)
            if ref.shape[:2] == (s.height, s.width):
                s.reference = ref
                s.reference_key = os.path.basename(path)
                s.exposure = exposure_from_reference(ref)
                return True
        s.reference = None
        s.reference_key = None
        return False

    # ------------------------------------------------------------ commands
    def command(self, payload: dict[str, Any]) -> dict[str, Any]:
        cmd = payload.get("cmd")
        try:
            with self.lock:
                if cmd == "set_scene":
                    kind = str(payload.get("kind"))
                    if kind not in ("window", "uniform"):
                        raise KernelError(f"未知场景: {kind}")
                    strategy = self.session.strategy
                    self.session = RenderSession(
                        kind, width=DEFAULT_W, height=DEFAULT_H,
                        seed=1234, strategy=strategy,
                        guide_mix=self.session.guide_mix,
                        eta=self.session.eta,
                        half_life=self.session.half_life,
                        move_retain=self.session.move_retain,
                        use_nee=self.session.use_nee,
                    )
                    self._load_reference_for_current()
                    return self.state()

                if cmd == "set":
                    params = payload.get("params") or {}
                    try:
                        self.session.update_params(params)
                    except (ValueError, KeyError, TypeError) as exc:
                        self.session._note("error", f"控件参数被内核拒绝: {exc}")
                        raise KernelError(str(exc)) from exc
                    return self.state()

                if cmd == "step":
                    n = int(payload.get("spp", 1))
                    n = max(1, min(n, 4))
                    last = None
                    for _ in range(n):
                        last = self.session.step_spp()
                    return self.state() | {"step_last": last}

                if cmd == "reset":
                    self.session.reset_all()
                    self._load_reference_for_current()
                    return self.state()

                if cmd == "reset_image":
                    self.session.reset_image()
                    return self.state()

                if cmd == "move_light":
                    placement = payload.get("placement")
                    msg = self.session.move_light(placement)
                    self._load_reference_for_current()
                    return self.state() | {"move_message": msg}

                if cmd == "save_version":
                    label = str(payload.get("label", "ui"))
                    path = self.session.guide.save_version(VERSION_DIR, label)
                    self.session._note("version", f"分布版本已保存: {os.path.basename(path)}")
                    return self.state() | {"saved_path": path}

                if cmd == "run_acceptance":
                    # Run the statistical-correctness acceptance set for real
                    # and persist the report; the UI renders what this run
                    # actually measured, never canned numbers.
                    from .acceptance import run_quick_suite
                    os.makedirs(ACCEPTANCE_DIR, exist_ok=True)
                    report = run_quick_suite(ACCEPTANCE_DIR, quick=True)
                    out = os.path.join(METRICS_DIR, "acceptance.json")
                    os.makedirs(METRICS_DIR, exist_ok=True)
                    with open(out, "w", encoding="utf-8") as fh:
                        json.dump(report, fh, ensure_ascii=False, indent=2)
                    n_ok = sum(1 for c in report["checks"] if c.get("ok"))
                    self.session._note(
                        "acceptance",
                        f"统计验收完成: {n_ok}/{len(report['checks'])} 项通过，"
                        f"报告见 outputs/metrics/acceptance.json")
                    return self.state() | {
                        "acceptance": {
                            "all_ok": report["all_ok"],
                            "n_checks": len(report["checks"]),
                            "n_ok": n_ok,
                            "elapsed_s": report["elapsed_s"],
                        }
                    }

                if cmd == "load_regression_case":
                    # Open a direction-guiding regression example in the live
                    # lighting-experiment window: train on placement A, then
                    # move the light with the requested staleness policy.
                    # Everything below drives the real RenderSession; the
                    # user keeps stepping/switching strategies manually and
                    # watches lighting convergence and direction maps.
                    case = str(payload.get("case", "stale"))
                    if case not in ("stale", "decay", "rebuild", "guide"):
                        raise KernelError(f"未知回归样例: {case}")
                    train_spp = max(1, min(int(payload.get("train_spp", 4)), 16))
                    s = self.session
                    s.reset_all()
                    s.set_strategy(case)
                    for _ in range(train_spp):
                        s.step_spp()
                    msg = s.move_light()
                    self._load_reference_for_current()
                    s._note("regression",
                            f"回归样例已载入：{case}，训练 {train_spp} spp 后移灯；"
                            "可继续单步/连续渲染并点击图像查看方向分布")
                    return self.state() | {"move_message": msg}

                if cmd == "list_versions":
                    return {"versions": self.list_versions()}

                if cmd == "load_version":
                    fname = os.path.basename(str(payload.get("filename", "")))
                    path = os.path.join(VERSION_DIR, fname)
                    if not os.path.exists(path):
                        raise KernelError(f"版本文件不存在: {fname}")
                    with open(path, encoding="utf-8") as fh:
                        data = json.load(fh)
                    g = DirectionGuide.from_snapshot(data)
                    cur = self.session.guide
                    if g.W.shape != cur.W.shape:
                        raise KernelError(
                            f"版本网格/方向分辨率 {g.W.shape} 与当前内核 {cur.W.shape} 不匹配"
                        )
                    self.session.guide = g
                    self.session._note("version", f"已载入分布版本 {fname}（仅检查，不改变图像能量）")
                    return self.state()

                raise KernelError(f"未知命令: {cmd}")
        except KernelError:
            raise
        except Exception as exc:  # surface unexpected kernel failures to UI
            try:
                self.session._note("error", f"内核失败: {type(exc).__name__}: {exc}")
            except Exception:
                pass
            raise KernelError(f"{type(exc).__name__}: {exc}") from exc

    # --------------------------------------------------------------- state
    def state(self) -> dict[str, Any]:
        with self.lock:
            return self.session.state_summary()

    def png(self) -> bytes:
        with self.lock:
            img = to_uint8(
                tonemap(self.session.image_hdr(), self.session.exposure)
            )
            bio = io.BytesIO()
            Image.fromarray(img, "RGB").save(bio, format="PNG")
            return bio.getvalue()

    def distribution(self, px: int, py: int, version: str | None = None) -> dict[str, Any]:
        with self.lock:
            if version:
                fname = os.path.basename(version)
                path = os.path.join(VERSION_DIR, fname)
                if not os.path.exists(path):
                    raise KernelError(f"版本不存在: {fname}")
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                g = DirectionGuide.from_snapshot(data)
                # Hit info (normal/point) still comes from the live scene.
                probe = self.session.probe(px, py)
                if probe.get("hit"):
                    dist = g.cell_distribution(int(probe["cell"]))
                    dist["normal"] = probe["normal"]
                    dist["point"] = probe["point"]
                    dist["hit"] = True
                    dist["pixel"] = [px, py]
                else:
                    dist = {"hit": False, "pixel": [px, py]}
                dist["version_file"] = fname
                return dist
            return self.session.probe(px, py)

    # ------------------------------------------------------------ versions
    def list_versions(self) -> list[dict[str, Any]]:
        if not os.path.isdir(VERSION_DIR):
            return []
        out = []
        for fn in sorted(os.listdir(VERSION_DIR), reverse=True):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(VERSION_DIR, fn)
            item = {"filename": fn, "size_kb": round(os.path.getsize(path) / 1024, 1)}
            try:
                with open(path, encoding="utf-8") as fh:
                    head = json.load(fh)
                item.update({
                    "label": head.get("label", ""),
                    "version": head.get("version"),
                    "spp": head.get("spp"),
                    "light_generation": head.get("light_generation"),
                    "saved_at": head.get("saved_at"),
                })
            except Exception:
                pass
            out.append(item)
        return out[:60]

    # ---------------------------------------------------------- experiment
    def experiment_results(self) -> dict[str, Any]:
        path = os.path.join(METRICS_DIR, "results.json")
        if not os.path.exists(path):
            self.results_cache = None
            return {"available": False}
        mtime = os.path.getmtime(path)
        if self.results_cache is None or self.results_cache_mtime != mtime:
            with open(path, encoding="utf-8") as fh:
                self.results_cache = json.load(fh)
            self.results_cache_mtime = mtime
        self.results_cache["available"] = True
        return self.results_cache

    def acceptance_report(self) -> dict[str, Any]:
        path = os.path.join(METRICS_DIR, "acceptance.json")
        if not os.path.exists(path):
            return {"available": False}
        with open(path, encoding="utf-8") as fh:
            report = json.load(fh)
        report["available"] = True
        return report


KERNEL = Kernel()
