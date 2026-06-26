"""Random course generator for testing.

Produces a closed outer wall as a random polygon whose corners are rounded with
*circular-arc fillets*, so every course exercises both wall primitive types and,
more to the point, the curved-section handling. Gradients come from Perlin fBm;
a couple of friction patches (sand / fast felt) are scattered inside; tee and
cup are placed far apart and comfortably clear of the walls.
"""

from __future__ import annotations
import numpy as np

from perlin import PerlinNoise2D
from geometry import Segment, Arc, closest_boundary
from course import Course, SurfacePatch, _point_in_polygon


def _rounded_boundary(vertices, centroid, rng, frac=(0.25, 0.45)):
    """Turn a polygon into a loop of Segments + concave Arc fillets."""
    n = len(vertices)
    info = []  # per-vertex: (arc, T_prev_side, T_next_side)
    for i in range(n):
        V = vertices[i]
        P = vertices[(i - 1) % n]
        N = vertices[(i + 1) % n]
        u = P - V; w = N - V
        lu = np.hypot(*u); lw = np.hypot(*w)
        u /= lu; w /= lw
        cosphi = np.clip(np.dot(u, w), -0.9999, 0.9999)
        phi = np.arccos(cosphi)                      # interior angle at V
        # tangent length, clamped so fillets never eat past an edge midpoint
        rf_from = min(lu, lw) * rng.uniform(*frac)
        t = rf_from
        rf = t * np.tan(phi / 2)
        t = min(t, 0.45 * lu, 0.45 * lw)
        rf = t * np.tan(phi / 2)
        bis = u + w
        bis /= (np.hypot(*bis) + 1e-12)
        center = V + bis * (rf / max(np.sin(phi / 2), 1e-3))
        T1 = V + u * t                               # toward prev
        T2 = V + w * t                               # toward next
        aT1 = np.arctan2(*(T1 - center)[::-1])
        aT2 = np.arctan2(*(T2 - center)[::-1])
        aV = np.arctan2(*(V - center)[::-1])
        # choose CCW direction that bulges toward the original vertex V
        def ccw_contains(a, b, x):
            twopi = 2 * np.pi
            return ((x - a) % twopi) <= ((b - a) % twopi)
        if ccw_contains(aT1, aT2, aV):
            a0, a1 = aT1, aT2
        else:
            a0, a1 = aT2, aT1
        arc = Arc(center, rf, a0, a1, concave=True)
        info.append((arc, T1, T2))

    prims = []
    for i in range(n):
        arc, T1, T2 = info[i]
        prims.append(arc)
        nxt_T1 = info[(i + 1) % n][1]
        prims.append(Segment(T2, nxt_T1, interior=centroid))
    return prims


def _gaussian_bump(X, Y, cx, cy, sx, sy, theta=0.0):
    """A single smooth (C-infinity) anisotropic Gaussian feature, peak 1.0."""
    ct, st = np.cos(theta), np.sin(theta)
    dx = X - cx
    dy = Y - cy
    u = dx * ct + dy * st
    v = -dx * st + dy * ct
    return np.exp(-(u * u) / (2 * sx * sx) - (v * v) / (2 * sy * sy))


def _designed_heightfield(X, Y, width, height_m, rng, seed, amplitude,
                          n_features=(3, 5), feature_sigma=(0.45, 0.95),
                          base_fraction=0.30):
    """Compose a height field that looks like a *designed* mini-golf green:
    a handful of distinct, smooth hills and troughs rather than fractal fuzz.

      * a few placed Gaussian mounds/bowls give the deliberate, trackable
        features (sized well above the ball radius so the ball can follow them);
      * a tamed, LOW-frequency 2-octave Perlin term adds gentle organic
        undulation without high-frequency speckle;
      * a slight overall tilt mimics a green that runs one way.

    The Gaussians dominate; Perlin is just seasoning. `feature_sigma` controls
    how broad/punchy the hills are, `n_features` how many.
    """
    # 1) tamed low-frequency Perlin base (broad, smooth - no speckle)
    noise = PerlinNoise2D(seed=seed + 1)
    base = noise.fbm(X * 0.30, Y * 0.30, octaves=2, gain=0.5)
    H = base_fraction * amplitude * base

    # 2) distinct hills and troughs, guaranteed at least one of each
    nf = int(rng.integers(n_features[0], n_features[1] + 1))
    signs = [1.0, -1.0] + list(rng.choice([1.0, -1.0], size=max(0, nf - 2)))
    rng.shuffle(signs)
    for sgn in signs:
        cx = rng.uniform(0.18, 0.82) * width
        cy = rng.uniform(0.18, 0.82) * height_m
        sx = rng.uniform(*feature_sigma)
        sy = sx * rng.uniform(0.7, 1.4)               # mild anisotropy
        theta = rng.uniform(0, np.pi)
        amp = sgn * rng.uniform(0.6, 1.0) * amplitude
        H = H + amp * _gaussian_bump(X, Y, cx, cy, sx, sy, theta)

    # 3) gentle overall run/tilt
    tilt = rng.uniform(-0.015, 0.015, size=2)
    H = H + tilt[0] * (X - width / 2) + tilt[1] * (Y - height_m / 2)

    H -= H.mean()
    return H


def generate_course(seed=0, width=6.0, height_m=4.0, n_vertices=7,
                    grid=(160, 240), amplitude=0.09, ball_radius=0.021,
                    n_features=(3, 5), feature_sigma=(0.45, 0.95)):
    rng = np.random.default_rng(seed)
    cx, cy = width / 2, height_m / 2

    # --- random star-shaped polygon (sorted angles guarantee no self-overlap)
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_vertices))
    # keep angles reasonably spread
    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False) + \
        rng.uniform(-0.25, 0.25, n_vertices)
    rad_x = 0.40 * width
    rad_y = 0.40 * height_m
    verts = []
    for a in angles:
        rr = rng.uniform(0.72, 1.0)
        verts.append([cx + np.cos(a) * rad_x * rr,
                      cy + np.sin(a) * rad_y * rr])
    verts = np.array(verts)
    centroid = verts.mean(axis=0)
    boundaries = _rounded_boundary(verts, centroid, rng)

    # --- designed height field: distinct smooth hills/troughs, low noise
    ny, nx = grid
    xs = np.linspace(0, width, nx)
    ys = np.linspace(0, height_m, ny)
    X, Y = np.meshgrid(xs, ys)
    H = _designed_heightfield(X, Y, width, height_m, rng, seed, amplitude,
                              n_features=n_features, feature_sigma=feature_sigma)

    # --- pick tee and cup: inside, far apart, clear of walls
    def deep_point(margin):
        for _ in range(4000):
            p = np.array([rng.uniform(0, width), rng.uniform(0, height_m)])
            if not _point_in_polygon(p, verts):
                continue
            _, _, _, d = closest_boundary(boundaries, p)
            if d > margin:
                return p
        return centroid.copy()

    margin = 6 * ball_radius
    start = deep_point(margin)
    hole = deep_point(margin)
    for _ in range(200):
        if np.hypot(*(hole - start)) > 0.45 * width:
            break
        hole = deep_point(margin)

    # flatten a small dish around the cup so the ball can settle and drop
    cup_flat = 0.18
    R = np.hypot(X - hole[0], Y - hole[1])
    blend = np.clip(1 - R / cup_flat, 0, 1) ** 2
    H = H * (1 - blend) + H[
        np.argmin(np.abs(ys - hole[1])), np.argmin(np.abs(xs - hole[0]))
    ] * blend

    # --- friction patches: a sand trap (grippy, holds on steep) and a fast strip
    surfaces = []
    sand_c = deep_point(margin + 0.1)
    if np.hypot(*(sand_c - hole)) > 0.4 and np.hypot(*(sand_c - start)) > 0.4:
        surfaces.append(SurfacePatch(mu_k=0.6, c_rr=0.20, k_v=3.0,   # hold ~12 deg
                                     circle=(sand_c, rng.uniform(0.25, 0.4)),
                                     name="sand"))
    fast_c = deep_point(margin + 0.1)
    if np.hypot(*(fast_c - hole)) > 0.4 and np.hypot(*(fast_c - start)) > 0.4:
        surfaces.append(SurfacePatch(mu_k=0.20, c_rr=0.15, k_v=0.25,  # hold ~1 deg
                                     circle=(fast_c, rng.uniform(0.3, 0.45)),
                                     name="fast"))

    return Course(width, height_m, H, boundaries, hole, hole_radius=0.08,
                  start=start, surfaces=surfaces,
                  base_mu_k=0.22, base_c_rr=0.025, base_k_v=0.40), verts
    # green: hold ~2 deg, roll-out = Stimp 8 ft, verts