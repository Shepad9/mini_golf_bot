"""Random course generator for testing.

Produces a closed outer wall as a random polygon whose corners are rounded with
*circular-arc fillets*, so every course exercises both wall primitive types and,
more to the point, the curved-section handling. On top of that bare arena each
hole grows **1-3 designed features** - narrow choke points, L-shaping corner
blocks, interior obstacles, and pure elevation mounds/valleys - whose shape,
placement and (optionally) elevation character are all drawn from the seed. No
two seeds produce the same arena: the size, the polygon, the feature count, the
feature kinds and every per-feature parameter are randomised independently.

Gradients come from Perlin fBm plus the per-feature elevation; a couple of
friction patches (sand / fast felt) are scattered inside; tee and cup are placed
far apart, clear of the walls, and never inside an obstacle.
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
                          n_features=(2, 4), feature_sigma=(0.45, 0.95),
                          base_fraction=0.30):
    """Compose a height field that looks like a *designed* mini-golf green:
    a handful of distinct, smooth hills and troughs rather than fractal fuzz.

      * a few placed Gaussian mounds/bowls give the deliberate, trackable
        features (sized well above the ball radius so the ball can follow them);
      * a tamed, LOW-frequency 2-octave Perlin term adds gentle organic
        undulation without high-frequency speckle;
      * a slight overall tilt mimics a green that runs one way.

    The Gaussians dominate; Perlin is just seasoning. `feature_sigma` controls
    how broad/punchy the hills are, `n_features` how many. The per-feature
    elevations added later by `generate_course` sit on top of this base.
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


# --------------------------------------------------------------------------- #
#  Interior obstacle primitives
#
#  Obstacles are closed loops of walls whose inward (playable) normals point
#  *away* from the obstacle body, so the physics - which only ever asks each
#  wall for (closest_point, inward_normal, distance) - pushes the ball around
#  them with no special-casing. Each builder also returns an `inside(p)` test so
#  the tee / cup / friction-patch placement can refuse points buried in a wall.
# --------------------------------------------------------------------------- #


def _circle_obstacle(center, r):
    """A smooth circular pillar: one full-circle Arc with the interior on the
    *outside* (concave=False), plus a strict point-inside test."""
    center = np.asarray(center, float)
    arc = Arc(center, r, 0.0, 2 * np.pi, concave=False)

    def inside(p, _c=center, _r=float(r)):
        return np.hypot(*(np.asarray(p, float) - _c)) < _r

    return [arc], inside


def _rect_obstacle(center, w, h, theta):
    """A rotated rectangular block: four Segments with outward normals. Used for
    corner blocks and choke-point gates. Returns (walls, inside_test, poly)."""
    c = np.asarray(center, float)
    ct, st = np.cos(theta), np.sin(theta)
    R = np.array([[ct, -st], [st, ct]])
    hw, hh = w / 2.0, h / 2.0
    local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    poly = local @ R.T + c
    walls = []
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        mid = 0.5 * (a + b)
        exterior = mid + (mid - c)        # always outside a convex body
        walls.append(Segment(a, b, interior=exterior))

    def inside(p, _poly=poly):
        return _point_in_polygon(np.asarray(p, float), _poly)

    return walls, inside, poly


def _feature_elevation(X, Y, center, scale, rng, amplitude):
    """Give a geometric feature a random elevation character so the *same* shape
    can sit on flat ground, perch on a hill, drop into a divot, or hug a slope
    beside it. Returns an additive height field (0.0 when flat)."""
    kind = rng.choice(["hill", "divot", "beside", "flat"])
    if kind == "flat":
        return 0.0
    cx, cy = float(center[0]), float(center[1])
    sigma = scale * rng.uniform(0.9, 1.7)
    amp = rng.uniform(0.5, 1.15) * amplitude
    if kind == "beside":                  # mound/divot just off to one side
        ang = rng.uniform(0, 2 * np.pi)
        off = scale * rng.uniform(1.3, 2.1)
        cx += np.cos(ang) * off
        cy += np.sin(ang) * off
        sgn = float(rng.choice([1.0, -1.0]))
    else:
        sgn = 1.0 if kind == "hill" else -1.0
    sx = sigma * rng.uniform(0.7, 1.3)
    sy = sigma * rng.uniform(0.7, 1.3)
    theta = rng.uniform(0, np.pi)
    return sgn * amp * _gaussian_bump(X, Y, cx, cy, sx, sy, theta)


# --------------------------------------------------------------------------- #
#  Feature builders. Each returns (walls, inside_tests, height_add, center).
# --------------------------------------------------------------------------- #


def _build_choke(width, height_m, centroid, rng, ball_radius, X, Y, amplitude):
    """A narrow choke point: two baffles reach in from opposite sides leaving
    only a slim gap between their tips. The baffles overshoot the outer wall so
    there is no sneak-by lane at the rim."""
    diam = 2.0 * ball_radius
    phi = rng.uniform(0, np.pi)                       # gate axis
    d = np.array([np.cos(phi), np.sin(phi)])          # baffles extend along d
    a = np.array([-d[1], d[0]])                       # passage runs along a
    L = min(width, height_m)
    gc = (centroid + a * rng.uniform(-0.15, 0.15) * L
          + d * rng.uniform(-0.10, 0.10) * L)
    gap = rng.uniform(4.0, 7.0) * diam                # the choke itself
    thick = rng.uniform(1.0, 1.8) * diam
    reach = 0.62 * max(width, height_m)               # overshoots the rim

    walls, tests = [], []
    for side in (1.0, -1.0):
        inner = gc + side * d * (gap / 2.0)
        outer = gc + side * d * reach
        mid = 0.5 * (inner + outer)
        length = reach - gap / 2.0
        w, ins, _ = _rect_obstacle(mid, length, thick, phi)
        walls += w
        tests.append(ins)

    h_add = _feature_elevation(X, Y, gc, 0.5 * gap + thick, rng, amplitude)
    return walls, tests, h_add, gc


def _build_corner(width, height_m, centroid, rng, X, Y, amplitude):
    """An L-shaping corner block: an axis-aligned rectangle wedged into one
    corner of the arena so the playable region bends around it."""
    corners = np.array([[0.0, 0.0], [width, 0.0],
                        [0.0, height_m], [width, height_m]])
    C = corners[int(rng.integers(0, 4))]
    direction = np.sign(centroid - C)                 # inward, +/-1 per axis
    sw = rng.uniform(0.28, 0.50) * width
    sh = rng.uniform(0.28, 0.50) * height_m
    # near edge sits ~just outside the corner (overshoot), far edges run inward
    center = C + direction * 0.44 * np.array([sw, sh])
    walls, ins, _ = _rect_obstacle(center, sw, sh, 0.0)
    h_add = _feature_elevation(X, Y, center, 0.35 * (sw + sh), rng, amplitude)
    return walls, [ins], h_add, center


def _build_pillar(center, width, height_m, rng, X, Y, amplitude):
    """A standalone interior object - a round boulder or a rotated block."""
    center = np.asarray(center, float)
    r = rng.uniform(0.06, 0.14) * min(width, height_m)
    if rng.random() < 0.55:
        walls, ins = _circle_obstacle(center, r)
    else:
        walls, ins, _ = _rect_obstacle(center, 2 * r, 2 * r,
                                       rng.uniform(0, np.pi / 2))
    h_add = _feature_elevation(X, Y, center, r * 1.6, rng, amplitude)
    return walls, [ins], h_add, center


def _build_mound(center, width, height_m, rng, X, Y, amplitude):
    """A pure elevation feature with no walls: a hill, a valley, or a ramp/ridge
    that the ball must read and account for. Stronger than the base undulation."""
    cx, cy = float(center[0]), float(center[1])
    kind = rng.choice(["hill", "valley", "ramp"])
    sigma = rng.uniform(0.40, 0.85)
    amp = rng.uniform(0.8, 1.4) * amplitude * (-1.0 if kind == "valley" else 1.0)
    if kind == "ramp":                                # an elongated ridge/slope
        sx = sigma * rng.uniform(1.8, 2.6)
        sy = sigma * rng.uniform(0.5, 0.8)
    else:
        sx = sigma * rng.uniform(0.8, 1.2)
        sy = sigma * rng.uniform(0.8, 1.2)
    theta = rng.uniform(0, np.pi)
    return amp * _gaussian_bump(X, Y, cx, cy, sx, sy, theta)


def generate_course(seed=0, width=6.0, height_m=4.0, n_vertices=None,
                    grid=(160, 240), amplitude=0.09, ball_radius=0.021,
                    n_features=(1, 3), vary_size=True):
    """Generate a unique, seeded mini-golf hole.

    The bare arena is a rounded random polygon; on top of it 1-3 designed
    features (`n_features`) are placed - choke points, corner blocks, interior
    objects, elevation mounds - each with its own seed-drawn shape and (for the
    geometric ones) elevation character. With `vary_size` the arena dimensions
    and vertex count are also randomised per seed, so holes differ in size too.
    """
    rng = np.random.default_rng(seed)

    # --- per-seed arena size / complexity ---------------------------------
    if vary_size:
        width = width * rng.uniform(0.85, 1.25)
        height_m = height_m * rng.uniform(0.85, 1.25)
        amplitude = amplitude * rng.uniform(0.8, 1.3)
    if n_vertices is None:
        n_vertices = int(rng.integers(5, 10))         # 5..9 corners
    cx, cy = width / 2, height_m / 2

    # --- random star-shaped polygon (sorted angles guarantee no self-overlap)
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
    H = _designed_heightfield(X, Y, width, height_m, rng, seed, amplitude)

    # --- placement helper: inside the arena, clear of every wall & obstacle
    obstacle_tests = []        # grows as features are added

    def is_inside(p):
        if not _point_in_polygon(p, verts):
            return False
        return not any(test(p) for test in obstacle_tests)

    def deep_point(margin, avoid=(), sep=0.0):
        for _ in range(4000):
            p = np.array([rng.uniform(0, width), rng.uniform(0, height_m)])
            if not is_inside(p):
                continue
            _, _, _, d = closest_boundary(boundaries, p)
            if d <= margin:
                continue
            if sep > 0 and any(np.hypot(*(p - q)) < sep for q in avoid):
                continue
            return p
        return centroid.copy()

    # --- 1..3 designed features, all parameters drawn from the seed --------
    pool = ["choke", "corner", "pillar", "mound"]
    n_feat = int(rng.integers(n_features[0], n_features[1] + 1))
    kinds = list(rng.choice(pool, size=n_feat, replace=True))
    feat_margin = 5 * ball_radius
    feature_centers = []
    for kind in kinds:
        if kind == "choke":
            w, tests, h_add, ctr = _build_choke(
                width, height_m, centroid, rng, ball_radius, X, Y, amplitude)
        elif kind == "corner":
            w, tests, h_add, ctr = _build_corner(
                width, height_m, centroid, rng, X, Y, amplitude)
        elif kind == "pillar":
            ctr = deep_point(feat_margin, avoid=feature_centers, sep=0.6)
            w, tests, h_add, center = _build_pillar( # stray center with no use
                ctr, width, height_m, rng, X, Y, amplitude)
        else:  # mound (pure elevation, no walls)
            ctr = deep_point(feat_margin, avoid=feature_centers, sep=0.5)
            w, tests = [], []
            h_add = _build_mound(ctr, width, height_m, rng, X, Y, amplitude)
        boundaries += w
        obstacle_tests += tests
        H = H + h_add
        feature_centers.append(np.asarray(ctr, float))

    # --- pick tee and cup: inside, far apart, clear of walls and obstacles
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
