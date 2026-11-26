"""Geometry, cameras and the two laboratory scenes.

All intersection routines are vectorized over arrays of rays shaped (N, 3).
Geometry is intentionally simple: axis rectangles (walls / boxes / lights)
and spheres, which is enough for window-aperture and uniform-room lighting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _v(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z], dtype=np.float64)


@dataclass
class Material:
    name: str
    albedo: np.ndarray
    emission: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def is_emissive(self) -> bool:
        return float(self.emission.max()) > 0.0


class Rect:
    """Axis-aligned (or arbitrarily oriented) rectangle."""

    def __init__(
        self,
        name: str,
        center: np.ndarray,
        axis_u: np.ndarray,
        half_u: float,
        axis_v: np.ndarray,
        half_v: float,
        normal: np.ndarray,
        mat_id: int,
    ) -> None:
        self.name = name
        self.center = np.asarray(center, dtype=np.float64)
        self.axis_u = np.asarray(axis_u, dtype=np.float64)
        self.axis_v = np.asarray(axis_v, dtype=np.float64)
        self.half_u = float(half_u)
        self.half_v = float(half_v)
        self.normal = np.asarray(normal, dtype=np.float64)
        self.mat_id = int(mat_id)

    @property
    def area(self) -> float:
        return 4.0 * self.half_u * self.half_v

    def intersect_t(
        self, o: np.ndarray, d: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (t, valid_mask) for ray arrays."""
        dn = d @ self.normal
        valid0 = np.abs(dn) > 1e-10
        t = ((self.center - o) @ self.normal) / np.where(valid0, dn, 1.0)
        p = o + t[:, None] * d
        rel = p - self.center
        au = rel @ self.axis_u
        av = rel @ self.axis_v
        valid = (
            valid0
            & (t > 1e-5)
            & (np.abs(au) <= self.half_u + 1e-7)
            & (np.abs(av) <= self.half_v + 1e-7)
        )
        t = np.where(valid, t, np.inf)
        return t, valid


class Sphere:
    def __init__(
        self, name: str, center: np.ndarray, radius: float, mat_id: int
    ) -> None:
        self.name = name
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = float(radius)
        self.mat_id = int(mat_id)

    def intersect_t(
        self, o: np.ndarray, d: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        oc = o - self.center
        b = np.sum(oc * d, axis=1)
        c = np.sum(oc * oc, axis=1) - self.radius * self.radius
        disc = b * b - c
        valid0 = disc >= 0.0
        sd = np.sqrt(np.maximum(disc, 0.0))
        t_near = -b - sd
        t_far = -b + sd
        t = np.where(t_near > 1e-5, t_near, t_far)
        valid = valid0 & (t > 1e-5)
        t = np.where(valid, t, np.inf)
        return t, valid


@dataclass
class LightPlacement:
    name: str
    center: np.ndarray
    axis_u: np.ndarray
    half_u: float
    axis_v: np.ndarray
    half_v: float
    normal: np.ndarray


class Scene:
    def __init__(self, name: str) -> None:
        self.name = name
        self.materials: list[Material] = []
        self.rects: list[Rect] = []
        self.spheres: list[Sphere] = []
        self.light_rect: Optional[Rect] = None
        self.placements: dict[str, LightPlacement] = {}
        self.placement_order: list[str] = []
        self.active_placement: Optional[str] = None
        self.recommended_exposure: float = 1.0

    # ------------------------------------------------------------ builders
    def add_material(
        self,
        name: str,
        albedo: tuple[float, float, float],
        emission: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> int:
        idx = len(self.materials)
        self.materials.append(
            Material(name, _v(*albedo), _v(*emission))
        )
        return idx

    def add_rect(
        self,
        name: str,
        center,
        axis_u,
        half_u: float,
        axis_v,
        half_v: float,
        normal,
        mat_id: int,
    ) -> Rect:
        r = Rect(
            name,
            _v(*center),
            _v(*axis_u),
            half_u,
            _v(*axis_v),
            half_v,
            _v(*normal),
            mat_id,
        )
        self.rects.append(r)
        return r

    def add_box(
        self,
        name: str,
        center,
        size,
        mat_id: int,
    ) -> None:
        cx, cy, cz = center
        hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
        self.add_rect(name + "+X", (cx + hx, cy, cz), (0, 0, 1), hz, (0, 1, 0), hy, (1, 0, 0), mat_id)
        self.add_rect(name + "-X", (cx - hx, cy, cz), (0, 0, 1), hz, (0, 1, 0), hy, (-1, 0, 0), mat_id)
        self.add_rect(name + "Y", (cx, cy + hy, cz), (1, 0, 0), hx, (0, 0, 1), hz, (0, 1, 0), mat_id)
        self.add_rect(name + "-Y", (cx, cy - hy, cz), (1, 0, 0), hx, (0, 0, 1), hz, (0, -1, 0), mat_id)
        self.add_rect(name + "+Z", (cx, cy, cz + hz), (1, 0, 0), hx, (0, 1, 0), hy, (0, 0, 1), mat_id)
        self.add_rect(name + "-Z", (cx, cy, cz - hz), (1, 0, 0), hx, (0, 1, 0), hy, (0, 0, -1), mat_id)

    def add_sphere(self, name: str, center, radius: float, mat_id: int) -> None:
        self.spheres.append(Sphere(name, _v(*center), radius, mat_id))

    def add_light_placement(self, key: str, p: LightPlacement) -> None:
        self.placements[key] = p
        self.placement_order.append(key)

    def set_light_placement(self, key: str) -> None:
        if self.light_rect is None or key not in self.placements:
            return
        p = self.placements[key]
        self.light_rect.center = p.center.copy()
        self.light_rect.axis_u = p.axis_u.copy()
        self.light_rect.half_u = p.half_u
        self.light_rect.axis_v = p.axis_v.copy()
        self.light_rect.half_v = p.half_v
        self.light_rect.normal = p.normal.copy()
        self.active_placement = key

    # --------------------------------------------------------- intersection
    def intersect(
        self, o: np.ndarray, d: np.ndarray
    ) -> dict[str, np.ndarray]:
        n_rays = o.shape[0]
        t_best = np.full(n_rays, np.inf)
        n_best = np.zeros((n_rays, 3))
        mat_best = np.full(n_rays, -1, dtype=np.int64)
        kind_best = np.full(n_rays, -1, dtype=np.int8)  # 0 rect, 1 sphere
        prim_best = np.full(n_rays, -1, dtype=np.int64)

        for i, r in enumerate(self.rects):
            t, valid = r.intersect_t(o, d)
            closer = valid & (t < t_best)
            if closer.any():
                t_best = np.where(closer, t, t_best)
                nn = np.broadcast_to(r.normal, n_best.shape)
                n_best = np.where(closer[:, None], nn, n_best)
                mat_best = np.where(closer, r.mat_id, mat_best)
                kind_best = np.where(closer, 0, kind_best)
                prim_best = np.where(closer, i, prim_best)

        for i, s in enumerate(self.spheres):
            t, valid = s.intersect_t(o, d)
            closer = valid & (t < t_best)
            if closer.any():
                t_best = np.where(closer, t, t_best)
                p = o + t[:, None] * d
                nn = (p - s.center) / s.radius
                n_best = np.where(closer[:, None], nn, n_best)
                mat_best = np.where(closer, s.mat_id, mat_best)
                kind_best = np.where(closer, 1, kind_best)
                prim_best = np.where(closer, i, prim_best)

        hit = np.isfinite(t_best)
        p = o + np.where(hit, t_best, 0.0)[:, None] * d
        # Face-forward shading normal.
        face = (n_best * d).sum(axis=1) > 0.0
        n_shade = np.where(face[:, None], -n_best, n_best)
        n_shade /= np.maximum(np.linalg.norm(n_shade, axis=1, keepdims=True), 1e-12)
        return {
            "hit": hit,
            "t": t_best,
            "p": p,
            "n": n_shade,
            "geo_n": n_best,
            "mat": mat_best,
            "kind": kind_best,
            "prim": prim_best,
        }

    def occluded(
        self, o: np.ndarray, d: np.ndarray, max_t: np.ndarray
    ) -> np.ndarray:
        """Return True if any geometry lies before max_t along each ray.

        Emissive rectangles are skipped: a shadow ray targets a point on the
        light itself, whose own surface must never count as an occluder.
        """
        blocked = np.zeros(o.shape[0], dtype=bool)
        limit = np.asarray(max_t, dtype=np.float64)
        for r in self.rects:
            if self.materials[r.mat_id].is_emissive:
                continue
            t, valid = r.intersect_t(o, d)
            blocked |= valid & (t < limit - 1e-3)
        for s in self.spheres:
            t, valid = s.intersect_t(o, d)
            blocked |= valid & (t < limit - 1e-3)
        return blocked

    # ------------------------------------------------------------- lighting
    def sample_light_point(
        self, n: int, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        assert self.light_rect is not None
        r = self.light_rect
        a = rng.random(n) * 2.0 - 1.0
        b = rng.random(n) * 2.0 - 1.0
        points = (
            r.center
            + a[:, None] * r.axis_u * r.half_u
            + b[:, None] * r.axis_v * r.half_v
        )
        return {
            "p": points,
            "n": np.broadcast_to(r.normal, (n, 3)).copy(),
            "pdf_area": np.full(n, 1.0 / r.area),
            "emission": np.broadcast_to(
                self.materials[r.mat_id].emission, (n, 3)
            ).copy(),
        }

    def light_solid_angle_pdf(
        self, pdf_area: np.ndarray, dist2: np.ndarray, cos_light: np.ndarray
    ) -> np.ndarray:
        return pdf_area * dist2 / np.maximum(np.abs(cos_light), 1e-6)


class Camera:
    def __init__(
        self,
        eye,
        target,
        fov_deg: float = 52.0,
        up=(0.0, 1.0, 0.0),
    ) -> None:
        self.eye = _v(*eye)
        self.target = _v(*target)
        self.fov_deg = fov_deg
        self.up0 = _v(*up)
        self.forward = self.target - self.eye
        self.forward /= np.linalg.norm(self.forward)
        self.right = np.cross(self.forward, self.up0)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.forward)
        self.tan_fov = math.tan(math.radians(fov_deg) / 2.0)

    def ray_for_pixel(
        self, px: float, py: float, width: int, height: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single deterministic ray (px,py in pixel coords, y down)."""
        aspect = width / height
        sx = (2.0 * (px + 0.5) / width - 1.0) * self.tan_fov * aspect
        sy = (1.0 - 2.0 * (py + 0.5) / height) * self.tan_fov
        d = self.forward + sx * self.right + sy * self.up
        d /= np.linalg.norm(d)
        return self.eye.copy(), d

    def generate_rays(
        self, width: int, height: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        n = width * height
        ys, xs = np.mgrid[0:height, 0:width]
        jx = rng.random(n)
        jy = rng.random(n)
        px = xs.reshape(-1) + jx
        py = ys.reshape(-1) + jy
        aspect = width / height
        sx = (2.0 * (px / width) - 1.0) * self.tan_fov * aspect
        sy = (1.0 - 2.0 * (py / height)) * self.tan_fov
        d = (
            self.forward[None, :]
            + sx[:, None] * self.right[None, :]
            + sy[:, None] * self.up[None, :]
        )
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        o = np.broadcast_to(self.eye, (n, 3)).copy()
        return o, d


# ---------------------------------------------------------------- scenes
ROOM = dict(x0=-3.0, x1=3.0, y0=0.0, y1=3.0, z0=-2.5, z1=2.5)


def _add_room(scn: Scene, wall_col=(0.74, 0.72, 0.68), floor_col=(0.52, 0.44, 0.36),
              back_wall: bool = True) -> None:
    wall = scn.add_material("wall", wall_col)
    floor = scn.add_material("floor", floor_col)
    ceil = scn.add_material("ceiling", (0.82, 0.82, 0.80))
    x0, x1 = ROOM["x0"], ROOM["x1"]
    y0, y1 = ROOM["y0"], ROOM["y1"]
    z0, z1 = ROOM["z0"], ROOM["z1"]
    xc, zc = (x0 + x1) / 2, (z0 + z1) / 2
    # floor / ceiling
    scn.add_rect("floor", (xc, y0, zc), (1, 0, 0), (x1 - x0) / 2, (0, 0, 1), (z1 - z0) / 2, (0, 1, 0), floor)
    scn.add_rect("ceiling", (xc, y1, zc), (1, 0, 0), (x1 - x0) / 2, (0, 0, 1), (z1 - z0) / 2, (0, -1, 0), ceil)
    # back / front
    if back_wall:
        scn.add_rect("back_wall", (xc, (y0 + y1) / 2, z0), (1, 0, 0), (x1 - x0) / 2, (0, 1, 0), (y1 - y0) / 2, (0, 0, 1), wall)
    scn.add_rect("front_wall", (xc, (y0 + y1) / 2, z1), (1, 0, 0), (x1 - x0) / 2, (0, 1, 0), (y1 - y0) / 2, (0, 0, -1), wall)


def _wall_with_window_x(
    scn: Scene, x: float, inward_sign: float, hole_z: float, hole_y: float,
    half_z: float, half_y: float, mat_id: int, prefix: str,
) -> None:
    """Build an x=const wall split around a rectangular hole."""
    z0, z1 = ROOM["z0"], ROOM["z1"]
    y0, y1 = ROOM["y0"], ROOM["y1"]
    normal = (inward_sign, 0, 0)  # inward direction
    hy_low = hole_y - half_y
    hy_high = hole_y + half_y
    hz_low = hole_z - half_z
    hz_high = hole_z + half_z
    # below / above hole (full z span)
    scn.add_rect(prefix + "_below", (x, (y0 + hy_low) / 2, (z0 + z1) / 2),
                 (0, 0, 1), (z1 - z0) / 2, (0, 1, 0), (hy_low - y0) / 2, normal, mat_id)
    scn.add_rect(prefix + "_above", (x, (hy_high + y1) / 2, (z0 + z1) / 2),
                 (0, 0, 1), (z1 - z0) / 2, (0, 1, 0), (y1 - hy_high) / 2, normal, mat_id)
    # side segments (only if positive size)
    if hz_low > z0 + 1e-4:
        scn.add_rect(prefix + "_sideL", (x, hole_y, (z0 + hz_low) / 2),
                     (0, 0, 1), (hz_low - z0) / 2, (0, 1, 0), half_y, normal, mat_id)
    if z1 > hz_high + 1e-4:
        scn.add_rect(prefix + "_sideR", (x, hole_y, (hz_high + z1) / 2),
                     (0, 0, 1), (z1 - hz_high) / 2, (0, 1, 0), half_y, normal, mat_id)


def _back_wall_two_windows(scn: Scene, z: float, mat_id: int) -> None:
    """Back wall z=const with two small square apertures (A left, B right)."""
    x0, x1 = ROOM["x0"], ROOM["x1"]
    y0, y1 = ROOM["y0"], ROOM["y1"]
    wy, hy = 1.95, 0.42
    hx = 0.42
    centers_x = (-1.55, 1.55)
    y_lo, y_hi = wy - hy, wy + hy
    n = (0, 0, 1)
    scn.add_rect("back_bottom", (0, (y0 + y_lo) / 2, z), (1, 0, 0), (x1 - x0) / 2,
                 (0, 1, 0), (y_lo - y0) / 2, n, mat_id)
    scn.add_rect("back_top", (0, (y_hi + y1) / 2, z), (1, 0, 0), (x1 - x0) / 2,
                 (0, 1, 0), (y1 - y_hi) / 2, n, mat_id)
    edges = [x0,
             centers_x[0] - hx, centers_x[0] + hx,
             centers_x[1] - hx, centers_x[1] + hx, x1]
    for i, (a, b) in enumerate([(edges[0], edges[1]),
                                (edges[2], edges[3]),
                                (edges[4], edges[5])]):
        scn.add_rect(f"back_mid_{i}", ((a + b) / 2, wy, z), (1, 0, 0), (b - a) / 2,
                     (0, 1, 0), hy, n, mat_id)


def build_window_scene(
    hole_centers_x: tuple[float, ...] = (-1.55, 1.55),
    hole_half_x: float = 0.42,
    hole_y0: float = 1.53,
    hole_y1: float = 2.37,
    cam_eye=(0.0, 1.35, 2.42),
    cam_target=(0.0, 0.95, -0.55),
    fov_deg: float = 58.0,
) -> tuple[Scene, Camera]:
    """Concentrated light through small apertures; A/B are left/right windows."""
    scn = Scene("window")
    _add_room(
        scn,
        wall_col=(0.76, 0.73, 0.67),
        floor_col=(0.46, 0.38, 0.29),
        back_wall=False,
    )
    wall_id = 0  # _add_room material order: wall=0, floor=1, ceiling=2
    x0, x1 = ROOM["x0"], ROOM["x1"]
    y0, y1 = ROOM["y0"], ROOM["y1"]
    z0, z1 = ROOM["z0"], ROOM["z1"]
    zc = (z0 + z1) / 2
    scn.add_rect("left_wall", (ROOM["x0"], 1.5, zc), (0, 0, 1), (z1 - z0) / 2,
                 (0, 1, 0), 1.5, (1, 0, 0), wall_id)
    scn.add_rect("right_wall", (ROOM["x1"], 1.5, zc), (0, 0, 1), (z1 - z0) / 2,
                 (0, 1, 0), 1.5, (-1, 0, 0), wall_id)
    _back_wall_two_windows(scn, z0, wall_id)

    # Sealed full-height partition with compact rectangular apertures aligned
    # to the A/B emitters.  An earlier build left the side segments below the
    # lintel height, accidentally opening broad high clerestory routes; the
    # learned maps then correctly pointed up-and-over and stayed flat.
    door_z = -0.55
    pn = (0, 0, 1)
    # Horizontal bands below / above the holes span the full room width.
    if hole_y0 > y0 + 1e-4:
        scn.add_rect("partition_below", (0.0, (y0 + hole_y0) / 2, door_z),
                     (1, 0, 0), (x1 - x0) / 2,
                     (0, 1, 0), (hole_y0 - y0) / 2, pn, wall_id)
    if y1 > hole_y1 + 1e-4:
        scn.add_rect("partition_above", (0.0, (hole_y1 + y1) / 2, door_z),
                     (1, 0, 0), (x1 - x0) / 2,
                     (0, 1, 0), (y1 - hole_y1) / 2, pn, wall_id)
    # Middle band: solid x-strips between the rectangular holes.
    edges = [x0]
    for cx in hole_centers_x:
        edges += [cx - hole_half_x, cx + hole_half_x]
    edges += [x1]
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        if b - a <= 1e-4:
            continue
        # Edges alternate: intervals right after a hole start are openings.
        # Edge list is x0, l0, r0, l1, r1, ..., x1; openings are (l_k, r_k),
        # i.e. index pairs (1,2),(3,4),... -> skip odd i.
        if i % 2 == 1:
            continue
        scn.add_rect(f"partition_mid_{i}", ((a + b) / 2, (hole_y0 + hole_y1) / 2, door_z),
                     (1, 0, 0), (b - a) / 2,
                     (0, 1, 0), (hole_y1 - hole_y0) / 2, pn, wall_id)

    red = scn.add_material("red_box", (0.66, 0.17, 0.12))
    teal = scn.add_material("teal_sphere", (0.22, 0.52, 0.52))
    ivory = scn.add_material("ivory_sphere", (0.86, 0.83, 0.76))
    scn.add_box("crate_left", (-1.35, 0.45, 0.95), (0.9, 0.9, 0.9), red)
    scn.add_sphere("sphere_right", (1.15, 0.58, 0.85), 0.58, ivory)
    scn.add_sphere("sphere_beyond_door", (-0.25, 0.3, -1.75), 0.3, teal)

    light_mat = scn.add_material("window_emitter", (0.0, 0.0, 0.0), (46.0, 40.0, 30.0))
    scn.light_rect = scn.add_rect(
        "window_light", (-1.55, 1.95, -2.74),
        (1, 0, 0), 0.42, (0, 1, 0), 0.42, (0, 0, 1), light_mat
    )
    scn.add_light_placement("A:left_window", LightPlacement(
        "A:left_window", _v(-1.55, 1.95, -2.74),
        _v(1, 0, 0), 0.42, _v(0, 1, 0), 0.42, _v(0, 0, 1)))
    scn.add_light_placement("B:right_window", LightPlacement(
        "B:right_window", _v(1.55, 1.95, -2.74),
        _v(1, 0, 0), 0.42, _v(0, 1, 0), 0.42, _v(0, 0, 1)))
    scn.set_light_placement("A:left_window")
    scn.recommended_exposure = 0.78

    cam = Camera(cam_eye, cam_target, fov_deg=fov_deg)
    return scn, cam


def build_uniform_scene() -> tuple[Scene, Camera]:
    """Broad ceiling panel -> soft, fairly uniform indoor illumination."""
    scn = Scene("uniform")
    _add_room(scn, wall_col=(0.80, 0.80, 0.78), floor_col=(0.62, 0.58, 0.50))
    wall_id = 0
    x0, x1 = ROOM["x0"], ROOM["x1"]
    y1 = ROOM["y1"]
    z0, z1 = ROOM["z0"], ROOM["z1"]
    xc, zc = 0.0, 0.0
    # Full side walls (no holes).
    scn.add_rect("left_wall", (x0, 1.5, zc), (0, 0, 1), (z1 - z0) / 2, (0, 1, 0), 1.5, (1, 0, 0), wall_id)
    scn.add_rect("right_wall", (x1, 1.5, zc), (0, 0, 1), (z1 - z0) / 2, (0, 1, 0), 1.5, (-1, 0, 0), wall_id)

    blue = scn.add_material("blue_box", (0.18, 0.32, 0.62))
    ochre = scn.add_material("ochre_sphere", (0.75, 0.52, 0.18))
    ivory = scn.add_material("ivory_sphere", (0.86, 0.84, 0.78))
    scn.add_box("box_left", (-1.5, 0.45, -0.9), (0.9, 0.9, 0.9), blue)
    scn.add_sphere("sphere_center", (0.2, 0.55, -0.6), 0.55, ivory)
    scn.add_sphere("sphere_right", (1.55, 0.34, -1.3), 0.34, ochre)

    light_mat = scn.add_material("ceiling_emitter", (0, 0, 0), (5.2, 5.3, 5.6))
    scn.light_rect = scn.add_rect(
        "ceiling_panel", (0.0, y1 - 0.02, 0.0),
        (1, 0, 0), 1.7, (0, 0, 1), 1.05, (0, -1, 0), light_mat
    )
    scn.add_light_placement("A:ceiling_front", LightPlacement(
        "A:ceiling_front", _v(0.0, y1 - 0.02, 0.55),
        _v(1, 0, 0), 1.7, _v(0, 0, 1), 1.05, _v(0, -1, 0)))
    scn.add_light_placement("B:ceiling_back", LightPlacement(
        "B:ceiling_back", _v(0.0, y1 - 0.02, -1.15),
        _v(1, 0, 0), 1.7, _v(0, 0, 1), 1.05, _v(0, -1, 0)))
    scn.set_light_placement("A:ceiling_front")
    scn.recommended_exposure = 1.15

    cam = Camera((0.15, 1.45, 2.3), (0.0, 1.1, -0.9), fov_deg=50)
    return scn, cam


def build_scene(kind: str) -> tuple[Scene, Camera]:
    if kind == "window":
        return build_window_scene()
    if kind == "uniform":
        return build_uniform_scene()
    raise ValueError(f"unknown scene kind: {kind}")


def default_grid_spec(kind: str):
    from .guiding import GridSpec

    # Cells roughly 0.8 m; room 6 x 3 x 5 -> 8 x 4 x 7.
    if kind in ("window", "uniform"):
        return GridSpec((-3.0, 0.0, -2.5), (3.0, 3.0, 2.5), (8, 4, 7))
    raise ValueError(kind)
