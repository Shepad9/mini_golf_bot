"""The Course aggregates everything the physics needs to query:

  height(x, y)      -> surface elevation (metres)      [from the Perlin field]
  gradient(x, y)    -> (dh/dx, dh/dy)                  [drives downhill accel]
  friction(x, y)    -> rolling-friction coefficient    [from surface patches]
  boundaries        -> list of Segment / Arc walls     [collision]
  hole, hole_radius -> the cup
  start             -> tee position

The height field is stored as a regular grid and sampled with bilinear
interpolation; the gradient uses central differences on that grid. Sampling a
pre-built grid (rather than re-evaluating noise per query) keeps every physics
step cheap and, crucially, gives a *consistent* height and slope - the slope the
ball feels is exactly the finite-difference slope of the surface it sits on.
"""

from __future__ import annotations
import numpy as np


class SurfacePatch:
    """A region with its own surface coefficients. Circle or polygon.

    mu_k : kinetic (sliding) friction coefficient - governs the skid phase
           right after a strike/bounce and the cushion tangential scrub.
    c_rr : Coulomb rolling resistance - a constant deceleration c_rr*g that
           dominates at low speed. It alone sets the STATIC HOLD THRESHOLD: a
           stationary ball stays put on slopes up to tan(theta) = 1.4 * c_rr.
    k_v  : viscous rolling drag (1/s) - a speed-dependent deceleration k_v*v
           that vanishes at rest. Together with c_rr it sets the ROLL-OUT
           DISTANCE, so distance and hold-slope can be tuned independently.
    """

    def __init__(self, mu_k, c_rr, k_v, circle=None, polygon=None, name="patch"):
        self.mu_k = float(mu_k)
        self.c_rr = float(c_rr)
        self.k_v = float(k_v)
        self.name = name
        self.center = None if circle is None else np.asarray(circle[0], float)
        self.radius = None if circle is None else float(circle[1])
        self.polygon = None if polygon is None else np.asarray(polygon, float)

    def contains(self, p):
        if self.center is not None:
            return np.hypot(*(np.asarray(p, float) - self.center)) <= self.radius
        return _point_in_polygon(np.asarray(p, float), self.polygon)


def _point_in_polygon(p, poly):
    x, y = p
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


class Course:
    def __init__(self, width, height_m, heightfield, boundaries,
                 hole, hole_radius, start, surfaces=None,
                 base_mu_k=0.20, base_c_rr=0.025, base_k_v=0.40):
        self.width = float(width)
        self.height_m = float(height_m)        # physical y-extent (metres)
        self.H = np.asarray(heightfield, float)  # shape (ny, nx)
        self.ny, self.nx = self.H.shape
        self.boundaries = list(boundaries)
        self.hole = np.asarray(hole, float)
        self.hole_radius = float(hole_radius)
        self.start = np.asarray(start, float)
        self.surfaces = list(surfaces or [])
        self.base_mu_k = float(base_mu_k)
        self.base_c_rr = float(base_c_rr)
        self.base_k_v = float(base_k_v)

        # grid spacing in metres
        self.dx = self.width / (self.nx - 1)
        self.dy = self.height_m / (self.ny - 1)
        # precompute gradient grids (central differences)
        gy, gx = np.gradient(self.H, self.dy, self.dx)
        self._gx, self._gy = gx, gy

    # ---- bilinear grid sampling -------------------------------------------
    def _sample(self, grid, x, y):
        fx = np.clip(x / self.dx, 0, self.nx - 1.000001)
        fy = np.clip(y / self.dy, 0, self.ny - 1.000001)
        x0 = int(np.floor(fx)); y0 = int(np.floor(fy))
        tx = fx - x0; ty = fy - y0
        v00 = grid[y0, x0];     v10 = grid[y0, x0 + 1]
        v01 = grid[y0 + 1, x0]; v11 = grid[y0 + 1, x0 + 1]
        return (v00 * (1 - tx) * (1 - ty) + v10 * tx * (1 - ty) +
                v01 * (1 - tx) * ty + v11 * tx * ty)

    def height(self, x, y):
        return float(self._sample(self.H, x, y))

    def gradient(self, x, y):
        return np.array([float(self._sample(self._gx, x, y)),
                         float(self._sample(self._gy, x, y))])

    def surface_props(self, x, y):
        """Return (mu_k, c_rr, k_v) for the surface under (x, y)."""
        p = (x, y)
        for s in self.surfaces:            # later patches override earlier ones
            if s.contains(p):
                return s.mu_k, s.c_rr, s.k_v
        return self.base_mu_k, self.base_c_rr, self.base_k_v

    def in_bounds(self, x, y):
        return 0 <= x <= self.width and 0 <= y <= self.height_m