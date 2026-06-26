"""Boundary primitives.

The central design decision for "reliable projections" against curved walls:
represent curves as *analytic circular arcs* rather than dense polylines.

Why this matters for a rolling-ball simulation
-----------------------------------------------
Collision resolution needs two things every step: the closest point on the wall
to the ball centre, and the surface normal there. With a polyline approximation
of a curve you get:
  * faceting - the normal is piecewise-constant and jumps at every vertex, so a
    ball rolling along the wall receives a juddering, direction-dependent bounce;
  * vertex ambiguity - near a shared vertex two segments fight over ownership and
    the projected point/normal can flip frame-to-frame.
An arc gives an *exact* closest point (centre + R * direction) and a normal that
varies continuously along its whole length. One arc replaces dozens of segments
and removes both artefacts. Segments are still used for the straight runs.

Every primitive exposes the same tiny interface:
    closest_point(p) -> point on the primitive nearest to p
    signed_offset(p) -> (closest_point, outward_unit_normal, distance)
where "outward" means pointing toward the playable interior (the side the ball
is on). The physics only ever calls signed_offset, so segments and arcs are
fully interchangeable.
"""

from __future__ import annotations
import numpy as np


def normalize(v):
    n = float(np.hypot(v[0], v[1]))
    if n < 1e-12:
        return np.array([0.0, 0.0]), 0.0
    return v / n, n


class Segment:
    """Line segment wall. `interior` is any point on the playable side, used
    once at construction to orient the normal consistently inward."""

    def __init__(self, a, b, interior=None):
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)
        ab = self.b - self.a
        n = np.array([-ab[1], ab[0]])
        nu, _ = normalize(n)
        if interior is not None:
            mid = 0.5 * (self.a + self.b)
            if np.dot(np.asarray(interior, float) - mid, nu) < 0:
                nu = -nu
        self._normal = nu  # fixed inward normal

    def closest_point(self, p):
        ab = self.b - self.a
        denom = float(np.dot(ab, ab))
        t = 0.0 if denom < 1e-12 else float(np.dot(p - self.a, ab) / denom)
        t = min(1.0, max(0.0, t))
        return self.a + t * ab

    def signed_offset(self, p):
        c = self.closest_point(p)
        d = p - c
        nrm, dist = normalize(d)
        if dist < 1e-12:               # exactly on the line: use stored normal
            nrm = self._normal
        return c, nrm, dist


class Arc:
    """Circular arc wall from angle a0 to a1 (radians, CCW) on a circle.

    `concave=True` means the playable interior is on the *centre* side of the
    arc (a fillet rounding a convex corner - the common mini-golf case). With
    concave=False the interior is on the far side (an outward bulge / island).
    The inward normal is radial, which is exactly why arcs project so cleanly.
    """

    def __init__(self, center, radius, a0, a1, concave=True):
        self.c = np.asarray(center, dtype=float)
        self.r = float(radius)
        # Normalise so a1 >= a0, sweeping CCW.
        while a1 < a0:
            a1 += 2 * np.pi
        self.a0, self.a1 = float(a0), float(a1)
        self.concave = concave

    def _clamp_angle(self, ang):
        # bring ang into [a0 - pi, a0 + pi) reference then clamp to [a0, a1]
        twopi = 2 * np.pi
        a = a0 = self.a0
        ang = (ang - a0) % twopi + a0
        if ang > self.a1:
            # pick whichever endpoint is angularly nearer
            d1 = ang - self.a1
            d0 = (a0 + twopi) - ang
            ang = self.a1 if d1 <= d0 else self.a0
        return ang

    def closest_point(self, p):
        d = np.asarray(p, float) - self.c
        ang = np.arctan2(d[1], d[0])
        ang = self._clamp_angle(ang)
        return self.c + self.r * np.array([np.cos(ang), np.sin(ang)])

    def signed_offset(self, p):
        c = self.closest_point(p)
        radial, _ = normalize(c - self.c)        # points outward from centre
        inward = -radial if self.concave else radial
        dist = float(np.hypot(*(p - c)))
        return c, inward, dist


def closest_boundary(primitives, p):
    """Return (primitive, closest_point, inward_normal, distance) for the wall
    nearest to point p. O(n) over primitives - fine for a few dozen walls; swap
    in a grid/BVH if a course ever has thousands."""
    best = None
    for prim in primitives:
        c, n, d = prim.signed_offset(p)
        if best is None or d < best[3]:
            best = (prim, c, n, d)
    return best