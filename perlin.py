"""Improved Perlin noise (Ken Perlin, 2002) in 2D, vectorised over NumPy arrays.

Used to generate smooth, tileable-free gradient fields for course heightmaps.
A fractal (fBm) sum of several octaves gives natural-looking undulation that
is C1-continuous, which matters: the physics differentiates the height field to
get slope, so a smooth source avoids spurious acceleration spikes.
"""

from __future__ import annotations
import numpy as np


class PerlinNoise2D:
    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Classic 256-entry permutation table, duplicated to avoid index wrap.
        p = rng.permutation(256)
        self._perm = np.concatenate([p, p]).astype(np.int32)

    @staticmethod
    def _fade(t):
        # 6t^5 - 15t^4 + 10t^3 : zero 1st and 2nd derivative at the ends.
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def _grad(h, x, y):
        # 8 gradient directions selected by the low 3 bits of the hash.
        h = h & 7
        u = np.where(h < 4, x, y)
        v = np.where(h < 4, y, x)
        u = np.where((h & 1) == 0, u, -u)
        v = np.where((h & 2) == 0, v, -v)
        return u + v

    def noise(self, x, y):
        """Single-octave noise in roughly [-1, 1]. x, y are arrays."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        xi = np.floor(x).astype(np.int32) & 255
        yi = np.floor(y).astype(np.int32) & 255
        xf = x - np.floor(x)
        yf = y - np.floor(y)
        u = self._fade(xf)
        v = self._fade(yf)

        p = self._perm
        aa = p[p[xi] + yi]
        ab = p[p[xi] + yi + 1]
        ba = p[p[xi + 1] + yi]
        bb = p[p[xi + 1] + yi + 1]

        x1 = np.where(True, self._lerp(self._grad(aa, xf, yf),
                                       self._grad(ba, xf - 1, yf), u), 0)
        x2 = self._lerp(self._grad(ab, xf, yf - 1),
                        self._grad(bb, xf - 1, yf - 1), u)
        return self._lerp(x1, x2, v)

    @staticmethod
    def _lerp(a, b, t):
        return a + t * (b - a)

    def fbm(self, x, y, octaves=4, lacunarity=2.0, gain=0.5, frequency=1.0):
        """Fractal Brownian motion: sum of octaves. Returns array in ~[-1, 1]."""
        total = np.zeros_like(np.asarray(x, dtype=float))
        amp = 1.0
        freq = frequency
        norm = 0.0
        for _ in range(octaves):
            total = total + amp * self.noise(np.asarray(x) * freq,
                                             np.asarray(y) * freq)
            norm += amp
            amp *= gain
            freq *= lacunarity
        return total / max(norm, 1e-9)