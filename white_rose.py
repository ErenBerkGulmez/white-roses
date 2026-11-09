"""
White Roses: a bouquet of white roses that grows and blooms in coloured ASCII.

Every frame, the bouquet is painted as small shaded 3D patches onto a tiny
offscreen image (one pixel per character cell). Each pixel is then redrawn as
a glyph from RAMP, picked by brightness and coloured with the pixel's colour.
A faint blurred copy underneath gives the glow.

The bloom, sped up to about 13 seconds:
  1. Stems rise and gather at a point; arching green sprigs fan out.
  2. Compound rose leaves unfurl, and a blush satin ribbon ties the bunch.
  3. Green buds swell at every tip and pale to cream.
  4. Branches of baby's breath fork through the bunch and burst into
     clouds of tiny white flowers.
  5. One by one the roses open from the outside in. A couple stay as buds.
  6. It keeps swaying, dust drifts up, and a message types itself out.

Run:
    py white_rose.py                       # window (needs pygame)
    py white_rose.py --terminal            # right in the terminal
    py white_rose.py --message "for you"   # your own message ("" for none)
    py white_rose.py --t 9                 # start 9 seconds into the bloom
    py white_rose.py --snapshot 15 --out roses.png   # save one frame

Window controls: click / Space / R replays the bloom, F toggles fullscreen,
Esc quits.

Requires: numpy, pillow, and pygame for the window mode.
"""

import argparse
import math
import os
import shutil
import sys
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
from PIL import Image, ImageDraw

RAMP = " .:-=+*cxoaX#%&@"
SS = 4  # supersampling: each character cell is painted as SS x SS pixels

# ---------------------------------------------------------------- timeline (s)
STEM_DUR = 2.6
SPRIG_START = 0.8
RIBBON_START = 2.4
BUD_START, BUD_DUR = 2.6, 1.8
SEPAL_START, SEPAL_DUR = 3.8, 1.6
GYP_START = 3.0
OPEN_START, OPEN_DUR = 4.6, 4.0
DUST_START = 10.0
MSG_START = 11.5

TIE = np.array([0.0, 0.0, -330.0])  # where the ribbon gathers the stems
GROUND_Z = -545.0
CAM_EL = math.radians(20)


# ---------------------------------------------------------------- helpers
def clamp(v, a, b):
    return max(a, min(b, v))


def lerp(a, b, t):
    return a + (b - a) * t


def prog(T, start, dur):
    return clamp((T - start) / dur, 0.0, 1.0)


def ease_out(t):
    return 1 - (1 - t) ** 3


def ease_in_out(t):
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def smooth(a, b, t):
    x = clamp((t - a) / (b - a), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-12)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def align_z(d):
    """Rotation that turns the +z axis onto direction d."""
    d = unit(d)
    k = np.cross([0, 0, 1.0], d)
    s, c = np.linalg.norm(k), d[2]
    if s < 1e-8:
        return np.eye(3)
    k /= s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def bez3(P, s):
    s = np.asarray(s, dtype=float)[..., None]
    u = 1 - s
    return u ** 3 * P[0] + 3 * u * u * s * P[1] + 3 * u * s * s * P[2] + s ** 3 * P[3]


def leaf_frame(d, tilt):
    """Side and normal vectors for a blade along d that mostly faces the viewer."""
    v = np.array([0, -1.0, 0])
    n = unit(v - np.dot(v, d) * d)
    e = np.cross(n, d)
    return (e * math.cos(tilt) + n * math.sin(tilt),
            n * math.cos(tilt) - e * math.sin(tilt))


# ---------------------------------------------------------------- shapes
def profile(v, shape):
    """Half-width along a blade, 0 at the base and v = 1 at the tip."""
    if shape == "petal":  # narrow claw, broad rounded top
        a = np.sin(0.5 * np.pi * np.clip(v / 0.6, 0, 1)) ** 0.8
        b = np.sqrt(np.clip(1 - (np.clip(v - 0.6, 0, None) / 0.4) ** 2.4, 0, 1))
        return a * b
    if shape == "sepal":  # long and pointed
        return 1.3 * np.clip(v / 0.12, 0, 1) ** 0.6 * np.clip(1 - v, 0, 1) ** 1.4
    if shape == "leaf":  # pointed oval with a serrated edge
        base = np.sin(np.pi * np.clip(v, 0, 1)) ** 0.75 * (1 - 0.25 * v)
        return base * (1 + 0.08 * np.abs(np.sin(9 * np.pi * v)))
    if shape == "loop":  # ribbon bow loop: pinched at the knot, round at the end
        return np.sin(np.pi * np.clip(v, 0, 1) ** 0.7) ** 0.6
    return np.ones_like(v)


def petal_surface(phi, r0, z0, L, W, alpha, curl, wrap, lip, dr, nv=8, nt=6,
                  shape="petal"):
    """Petals (or sepals) around the flower axis, as a (k, nv, nt, 3) grid.

    Every parameter is an array with one value per petal. alpha: lean outward
    from vertical at the base; curl: extra lean toward the tip (the tip rolls
    back); wrap: 1 = wrapped around the axis like a bud, 0 = flat and open;
    lip: side edges roll outward.
    """
    a = [np.atleast_1d(np.asarray(x, dtype=float)) for x in
         (phi, r0, z0, L, W, alpha, curl, wrap, lip, dr)]
    phi, r0, z0, L, W, alpha, curl, wrap, lip, dr = np.broadcast_arrays(*a)
    v = np.linspace(0, 1, nv)
    t = np.linspace(-1, 1, nt)
    th = alpha[:, None] + curl[:, None] * v ** 2.2
    ds = (L / (nv - 1))[:, None]
    mid = 0.5 * (th[:, 1:] + th[:, :-1])
    zero = np.zeros((len(phi), 1))
    h = r0[:, None] + np.concatenate([zero, np.cumsum(np.sin(mid), 1) * ds], 1)
    z = z0[:, None] + np.concatenate([zero, np.cumsum(np.cos(mid), 1) * ds], 1)
    w = W[:, None] * profile(v, shape)[None]

    A = t[None, None, :] * w[:, :, None]  # arc length across the petal
    roll = lip[:, None, None] * (np.abs(t) ** 3)[None, None, :] * (v ** 1.5)[None, :, None]
    Lb = L[:, None, None]
    H = h[:, :, None] + roll * Lb * 0.14
    Z = z[:, :, None] - roll * Lb * 0.06 + 0 * A

    ph = phi[:, None, None]
    cp, sp = np.cos(ph), np.sin(ph)
    xf, yf = H * cp - A * sp, H * sp + A * cp
    rho = np.maximum(H, w[:, :, None] / 1.9 + 0.5) + dr[:, None, None]
    ang = ph + A / rho
    xw, yw = rho * np.cos(ang), rho * np.sin(ang)
    wr = wrap[:, None, None]
    return np.stack([xf + (xw - xf) * wr, yf + (yw - yf) * wr, Z], -1)


def blade(origin, d, e, n, L, W, shape, fold=0.0, droop=0.0, nv=8, nt=5):
    """A flat blade (leaflet, ribbon, petiole) from origin pointing along d."""
    v = np.linspace(0, 1, nv)[:, None, None]
    t = np.linspace(-1, 1, nt)[None, :, None]
    w = W * profile(v, shape)
    return (np.asarray(origin) + v * L * np.asarray(d) + t * w * np.asarray(e)
            + fold * np.abs(t) * w * np.asarray(n)
            + droop * v ** 2 * L * np.array([0, 0, -1.0]))


def tube(C, r, m=8):
    """A round tube along centerline C (n, 3) with radius r (n,)."""
    Tn = np.gradient(C, axis=0)
    Tn /= np.linalg.norm(Tn, axis=-1, keepdims=True) + 1e-12
    N1 = np.cross(Tn, [0, 1.0, 0])
    N1 /= np.linalg.norm(N1, axis=-1, keepdims=True) + 1e-12
    N2 = np.cross(Tn, N1)
    a = np.linspace(0, 2 * np.pi, m + 1)
    return C[:, None, :] + np.asarray(r)[:, None, None] * (
        np.cos(a)[None, :, None] * N1[:, None, :] + np.sin(a)[None, :, None] * N2[:, None, :])


# ---------------------------------------------------------------- materials
def material(shadow, light, gamma=0.85, warm=0.0, green=0.0, petal=0.0):
    """A material as one row: shadow rgb, light rgb, gamma, warm, green, petal."""
    return np.array([*shadow, *light, gamma, warm, green, petal], dtype=float)


PETAL_SHADOW, PETAL_LIGHT = (92, 92, 124), (255, 254, 251)
CREAM = np.array([255, 238, 200.0]) / 255
SAP = np.array([170, 208, 122.0]) / 255

STEM_MAT = material((14, 56, 20), (110, 186, 80))
SEPAL_MAT = material((8, 40, 14), (92, 160, 64))
LEAF_MATS = [material((10, 48, 16), (140, 214, 100)), material((12, 54, 20), (120, 204, 112))]
SPRIG_MAT = material((6, 40, 22), (96, 190, 120), gamma=0.7)
RIBBON_MAT = material((120, 62, 82), (255, 222, 230), gamma=0.6)
THORN_RGB = (128, 70, 54)
PETAL_EDGE = (8, 8, 12)
GREEN_EDGE = (2, 12, 4)
RIBBON_EDGE = (40, 12, 22)
LIGHT = unit([-0.5, -0.6, 0.65])


def ease_in_out_v(t):
    t = np.clip(t, 0, 1)
    return np.where(t < 0.5, 4 * t ** 3, 1 - (-2 * t + 2) ** 3 / 2)


# (count, r0, L, W, alpha closed/open (deg), curl, wrap open, lip, start)
LAYERS = [
    (5, 20, 175, 100, 12, 94, 1.05, 0.12, 0.65, 0.0),
    (5, 15, 165, 90, 9, 62, 0.75, 0.30, 0.50, 0.45),
    (4, 10, 152, 76, 6, 36, 0.50, 0.55, 0.35, 0.85),
    (4, 6, 138, 62, 4, 18, 0.30, 0.82, 0.20, 1.2),
    (3, 3, 120, 46, 2, 7, 0.14, 1.00, 0.08, 1.5),
]


# ---------------------------------------------------------------- the bouquet
class Flower:
    """One rose: its head, stem from the ground through the tie, and a leaf."""

    def __init__(self, rng, head, scale, delay, max_open):
        self.H = np.array(head, float)
        self.scale, self.delay, self.max_open = scale, delay, max_open
        out = self.H - TIE
        self.dir = unit(out * [1.35, 1.35, 1.0] + [0, -70, 0])
        self.R = align_z(self.dir) @ rot_z(rng.uniform(0, 2 * np.pi))
        self.stem_start = rng.uniform(0, 0.45)
        layers = LAYERS if max_open > 0.3 else LAYERS[:3]
        petals = []
        for li, (n, r0, L, W, ac, ao, curl, wo, lip, start) in enumerate(layers):
            for i in range(n):
                petals.append(dict(
                    phi=math.radians(li * 37 + i * 360 / n + rng.uniform(-9, 9)),
                    r0=r0, L=L * rng.uniform(0.94, 1.06), W=W * rng.uniform(0.94, 1.06),
                    ac=math.radians(ac), ao=math.radians(ao + rng.uniform(-6, 6)),
                    curl=curl, wo=wo, lip=lip, dr=i * 1.7, z0=li * 4.0,
                    open_at=OPEN_START + delay + start + i * 0.12 + rng.uniform(0, 0.1),
                    outer=1 - li / 4, warm=0.25 + 0.75 * li / 4,
                ))
        self.pa = {k: np.array([pt[k] for pt in petals]) for k in petals[0]}
        self.sepals = [math.radians(i * 72 + rng.uniform(0, 40)) for i in range(5)]
        g = TIE + [rng.uniform(-10, 10), rng.uniform(-10, 10), 0]
        self.base = np.array([out[0] * 0.13 + rng.uniform(-8, 8), out[1] * 0.1, GROUND_Z])
        self.g = g
        self.upper = [g, g + [0, 0, 120], self.H - self.dir * 120, self.H]
        side = 1 if out[0] >= 0 else -1
        self.leaves = [dict(s=s, side=sd * side, up=math.radians(rng.uniform(15, 40)),
                            size=rng.uniform(0.85, 1.1))
                       for s, sd in [(0.6, 1), (0.8, -1)]]
        self.thorns = [(rng.uniform(0.4, 0.9), rng.choice([-1, 1])) for _ in range(2)]

    def stem(self, s):
        """Centerline: straight from the ground to the tie, then curving up."""
        s = np.asarray(s, dtype=float)
        lo = self.base + (self.g - self.base) * np.clip(s / 0.3, 0, 1)[..., None]
        hi = bez3(self.upper, np.clip((s - 0.3) / 0.7, 0, 1))
        return np.where((s < 0.3)[..., None], lo, hi)


class Bouquet:
    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        # head position, size, open delay, how far it opens (1 = full bloom)
        heads = [
            ((0, 16, 62), 0.66, 0.0, 1.0),
            ((-205, 30, -6), 0.57, 0.6, 1.0),
            ((200, 34, 4), 0.58, 1.0, 1.0),
            ((-118, -104, -138), 0.55, 1.5, 1.0),
            ((126, -100, -150), 0.53, 1.9, 0.9),
            ((-96, 74, 130), 0.5, 2.3, 0.28),
            ((112, 84, 146), 0.46, 2.6, 0.0),
            ((-250, -20, -150), 0.44, 2.8, 0.0),
        ]
        self.flowers = [Flower(rng, *h) for h in heads]

        # arching sprigs of glossy greenery fanning out from the tie
        self.sprigs = []
        for end in [(-360, 10, -150), (360, 20, -120), (-320, 70, 90), (330, 70, 70),
                    (-40, 120, 250), (200, -120, -260), (-210, -110, -270)]:
            end = np.array(end, float)
            base = TIE + [rng.uniform(-12, 12), rng.uniform(-8, 8), 30]
            mid = (base + end) / 2 + [0, 0, 70 + rng.uniform(0, 40)]
            pts = [base, lerp(base, mid, 0.7), lerp(mid, end, 0.4), end]
            leaves = []
            t, i = 0.24, 0
            while t <= 1.0:
                leaves.append((t, 1 if i % 2 else -1, rng.uniform(32, 44), rng.uniform(10, 13)))
                t += rng.uniform(0.08, 0.11)
                i += 1
            leaves.append((1.0, 0, 46, 12))
            self.sprigs.append(dict(pts=pts, leaves=leaves,
                                    start=SPRIG_START + rng.uniform(0, 0.9)))

        # baby's breath: thin stems that fork again and again into tiny flowers
        segs, flowers = [], []
        centers = [(-250, 50, 70), (250, 60, 80), (10, 110, 210), (-60, -120, -190),
                   (220, -70, -190), (-300, -40, -60), (60, 110, 150), (-150, 100, 170)]
        for ci, c in enumerate(centers):
            c = np.array(c, float)
            start = GYP_START + ci * 0.22 + rng.uniform(0, 0.3)
            front = c[1] < 0
            base = TIE + [rng.uniform(-14, 14), rng.uniform(-10, 10), 10]
            knee = c - [0, 0, 50]
            segs.append((base, knee, 0.0, start, front, 2.2))
            out = unit(c - TIE)

            def branch(p, d, length, depth, t):
                q = p + d * length
                segs.append((p, q, t, start, front, 1.0))
                t2 = t + 0.17
                if depth == 0:
                    flowers.append((q, rng.uniform(5, 8), t2 + rng.uniform(0, 0.3), start, front))
                    return
                if depth < 3 and rng.random() < 0.7:
                    flowers.append((q, rng.uniform(4, 6), t2 + rng.uniform(0, 0.3), start, front))
                for _ in range(2 + (rng.random() < 0.45)):
                    nd = unit(d + rng.normal(0, 0.75, 3) + out * 0.45 + [0, 0, 0.25])
                    branch(q, nd, length * rng.uniform(0.5, 0.8), depth - 1, t2)

            branch(knee, unit(out + [0, 0, 0.6]), 36, 4, 0.5)
        self.gyp_a = np.array([s[0] for s in segs])
        self.gyp_b = np.array([s[1] for s in segs])
        self.gyp_t = np.array([s[2] for s in segs])
        self.gyp_start = np.array([s[3] for s in segs])
        self.gyp_front = np.array([s[4] for s in segs])
        self.gyp_w = np.array([s[5] for s in segs])
        self.gf_p = np.array([f[0] for f in flowers])
        self.gf_s = np.array([f[1] for f in flowers])
        self.gf_t = np.array([f[2] for f in flowers])
        self.gf_start = np.array([f[3] for f in flowers])
        self.gf_front = np.array([f[4] for f in flowers])
        self.gf_ph = rng.uniform(0, 2 * np.pi, len(flowers))
        self.dust = rng.uniform(0, 1, size=(26, 5))


class Painter:
    """Turns the bouquet at time T into a small RGB image, one pixel per cell."""

    def __init__(self, bouquet):
        self.b = bouquet
        self.cache = {}  # finished parts: their geometry no longer changes
        self.rec = None

    def cached(self, name, done, build):
        """Build a part, or replay it from the cache once it has finished growing.

        Sway and the camera are applied later, at projection, so a finished
        part's world geometry can be reused every frame.
        """
        if done and name in self.cache:
            for kind, args in self.cache[name]:
                (self.surface if kind == "s" else self.triangle)(*args)
            return
        self.rec = rec = []
        build()
        self.rec = None
        if done:
            self.cache[name] = rec

    # --- camera & projection -------------------------------------------
    def setup(self, T, cols, rows, aspect):
        self.T = T
        az = -0.08 + 0.1 * math.sin(T * 0.21)
        self.cam = rot_x(CAM_EL) @ rot_z(az)
        self.view = self.cam.T @ np.array([0, -1.0, 0])  # toward the viewer
        sx = min(cols / 880, rows * aspect / 900)
        self.sx, self.sy = sx, sx / aspect
        self.cx = cols / 2
        self.cy = (rows - 880 * self.sy) / 2 + 330 * self.sy
        self.lw = max(1, round(SS * 0.7))
        self.polys, self.depth, self.cols, self.flags, self.ecols = [], [], [], [], []
        self.groups = {}

    def sway(self, P):
        P = np.array(P, dtype=float)
        lift = np.clip((P[..., 2] - TIE[2]) / 560, 0, 1)
        T = self.T
        P[..., 0] += (np.sin(T * 0.9 + P[..., 2] * 0.005) * 8
                      + np.sin(T * 0.43 + P[..., 0] * 0.004) * 4) * lift
        return P

    def project(self, P):
        PV = self.sway(P) @ self.cam.T
        X = (self.cx + PV[..., 0] * self.sx) * SS
        Y = (self.cy - PV[..., 2] * self.sy) * SS
        return X, Y, PV[..., 1]

    # --- geometry to quads ---------------------------------------------
    def surface(self, P, mat, edge=None, rib=False):
        """Queue grid surface(s) P of shape (nv, nt, 3) or (k, nv, nt, 3)."""
        if self.rec is not None:
            self.rec.append(("s", (P, mat, edge, rib)))
        P = np.asarray(P, dtype=float)
        if P.ndim == 3:
            P = P[None]
        k, nv, nt = P.shape[:3]
        if nv < 2 or nt < 2 or k == 0:
            return
        mat = np.broadcast_to(np.asarray(mat, float), (k, 10))
        g = self.groups.setdefault((nv, nt, edge, rib), ([], []))
        g[0].append(P)
        g[1].append(mat)

    def flush(self):
        """Shade, project and emit quads for every queued surface, in batches."""
        for (nv, nt, edge, rib), (Ps, Ms) in self.groups.items():
            P = np.concatenate(Ps)
            M = np.concatenate(Ms)
            K = len(P)
            X, Y, D = self.project(P)
            n = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=2))
            n = n[:, :-1, :-1] + n[:, 1:, :-1] + n[:, :-1, 1:] + n[:, 1:, 1:]
            n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
            facing = n @ self.view
            back = facing < 0
            n = np.where(back[..., None], -n, n)
            lum = np.clip(0.13 + 0.87 * np.clip(n @ LIGHT, 0, 1)
                          + 0.12 * (1 - np.abs(facing)) ** 3, 0, 1)

            vq = ((np.arange(nv - 1) + 0.5) / (nv - 1))[None, :, None]
            m = M[:, None, None, :]
            warm, green, petal = m[..., 7], m[..., 8], m[..., 9]
            lum = np.maximum(lum, 0.42 * warm)  # a rose's heart glows a little
            c = m[..., 0:3] + (m[..., 3:6] - m[..., 0:3]) * (lum ** m[..., 6])[..., None]
            kc = (warm * np.clip(1 - vq * 1.25, 0, 1))[..., None]
            tint = 1 - kc + kc * CREAM
            gg = np.clip(np.clip(1 - vq / 0.2, 0, 1) * 0.55 * petal + green, 0, 1)[..., None]
            c = c * (tint * (1 - gg) + SAP * gg)
            c = np.where(back[..., None], c * 0.8, c)

            poly = np.stack([X[:, :-1, :-1], Y[:, :-1, :-1], X[:, 1:, :-1], Y[:, 1:, :-1],
                             X[:, 1:, 1:], Y[:, 1:, 1:], X[:, :-1, 1:], Y[:, :-1, 1:]], -1)
            flags = np.zeros((nv - 1, nt - 1), np.int8)
            if edge:
                flags[:, 0] |= 1  # left side
                flags[:, -1] |= 2  # right side
                flags[-1, :] |= 4  # tip
                if rib:
                    flags[:, nt // 2] |= 1  # midrib
            q = K * (nv - 1) * (nt - 1)
            self.polys.append(poly.reshape(q, 8))
            self.depth.append((0.25 * (D[:, :-1, :-1] + D[:, 1:, :-1] + D[:, :-1, 1:]
                                       + D[:, 1:, 1:])).ravel())
            self.cols.append(np.clip(c, 0, 255).reshape(q, 3))
            self.flags.append(np.broadcast_to(flags, (K,) + flags.shape).ravel())
            self.ecols.append(np.tile(edge or (0, 0, 0), (q, 1)))

    def triangle(self, P, rgb, bias=0.0):
        if self.rec is not None:
            self.rec.append(("t", (P, rgb, bias)))
        X, Y, D = self.project(P)
        self.polys.append(np.array([[X[0], Y[0], X[1], Y[1], X[2], Y[2], X[2], Y[2]]]))
        self.depth.append(np.array([D.mean() + bias]))
        self.cols.append(np.array([rgb], float))
        self.flags.append(np.zeros(1, np.int8))
        self.ecols.append(np.zeros((1, 3)))

    # --- parts -----------------------------------------------------------
    def compound_leaf(self, base, d0, size, start, mi):
        T = self.T
        u = ease_out(prog(T, start, 1.3))
        if u <= 0.02:
            return
        e, npl = leaf_frame(d0, 0.5)
        Lp = 66 * size * u
        self.surface(blade(base, d0, e, npl, Lp, 2.6, "rect", nv=4, nt=3), STEM_MAT)
        for f, sgn, sz in [(0.42, -1, 0.66), (0.42, 1, 0.66), (1.0, 0, 1.0)]:
            ul = ease_out(prog(T, start + 0.2 + f * 0.45, 1.2))
            if ul <= 0.02:
                continue
            o = base + d0 * Lp * f
            ang = sgn * (0.95 - 0.35 * (1 - ul))
            d = unit(d0 * math.cos(ang) + e * math.sin(ang))
            ee = unit(np.cross(npl, d))
            P = blade(o, d, ee, npl, 62 * size * sz * ul, 23 * size * sz * ul, "leaf",
                      fold=lerp(1.1, 0.32, ul), droop=0.14)
            self.surface(P, LEAF_MATS[(mi + int(f * 3)) % 2], GREEN_EDGE, rib=True)

    def stems(self):
        T = self.T
        for fi, fl in enumerate(self.b.flowers):
            p = ease_in_out(prog(T, fl.stem_start, STEM_DUR))
            if p <= 0:
                continue
            s = np.linspace(0, p, max(2, int(26 * p) + 2))
            self.surface(tube(fl.stem(s), 7.0 - 1.5 * s), STEM_MAT)
            for ts, side in fl.thorns:
                k = smooth(ts, ts + 0.06, p)
                if k > 0:
                    c = fl.stem(ts)
                    side_v = np.array([side * 1.0, 0, 0])
                    self.triangle(np.array([c + side_v * 5 + [0, 0, -7], c + side_v * 5 + [0, 0, 6],
                                            c + side_v * (5 + 12 * k) + [0, -2, 8 * k]]),
                                  THORN_RGB, bias=-3)
            for lf in fl.leaves:
                if p < lf["s"]:
                    continue
                c = fl.stem(lf["s"])
                a = lf["up"]
                d0 = unit([lf["side"] * math.cos(a), -0.35, math.sin(a)])
                start = fl.stem_start + STEM_DUR * lf["s"] * 0.8 + 0.3
                self.compound_leaf(c, d0, lf["size"], start, fi)

    def sprigs(self):
        T = self.T
        for sp in self.b.sprigs:
            p = ease_out(prog(T, sp["start"], 2.2))
            if p <= 0:
                continue
            s = np.linspace(0, p, max(2, int(18 * p) + 2))
            self.surface(tube(bez3(sp["pts"], s), np.full(len(s), 2.6), m=5), SPRIG_MAT)
            for t, side, ln, w in sp["leaves"]:
                lp = ease_out(clamp((p - t + 0.24) / 0.24, 0, 1))  # full size once the tip passes
                if lp <= 0.03:
                    continue
                o = bez3(sp["pts"], t)
                tan = unit(bez3(sp["pts"], min(1, t + 0.01)) - bez3(sp["pts"], max(0, t - 0.01)))
                e, n = leaf_frame(tan, 0.45 * (side or 1))
                d = tan if side == 0 else unit(tan * math.cos(0.75) + e * math.sin(0.75) * side)
                ee = unit(np.cross(n, d))
                self.surface(blade(o, d, ee, n, ln * lp, w * lp, "leaf", fold=0.35, droop=0.05,
                                   nv=6, nt=5), SPRIG_MAT, GREEN_EDGE, rib=True)

    def ribbon(self):
        u = ease_out(prog(self.T, RIBBON_START, 1.1))
        if u <= 0.02:
            return
        z = np.linspace(-20, 16, 3) * u
        band = np.stack([np.zeros(3), np.zeros(3), TIE[2] + z], -1)
        self.surface(tube(band, np.full(3, 30.0), m=14), RIBBON_MAT, RIBBON_EDGE)
        knot = TIE + [0, -31, 0]
        for side in (-1, 1):
            d = unit([side, -0.35, 0.55])
            e, n = leaf_frame(d, 0.9 * side)
            self.surface(blade(knot, d, e, n, 66 * u, 24 * u, "loop", fold=0.9, nv=8, nt=5),
                         RIBBON_MAT, RIBBON_EDGE)
            d = unit([side * 0.4, -0.3, -1])
            e, n = leaf_frame(d, 0.4 * side)
            self.surface(blade(knot, d, e, n, 130 * u, 9, "rect", fold=0.25, nv=8, nt=3),
                         RIBBON_MAT, RIBBON_EDGE)
        sph = tube(np.stack([np.zeros(4), np.full(4, -33.0), TIE[2] + np.linspace(-9, 9, 4)], -1),
                   np.array([3, 9, 9, 3.0]) * u, m=8)
        self.surface(sph, RIBBON_MAT)

    def flowers(self):
        T = self.T
        for fl in self.b.flowers:
            g = ease_out(prog(T, BUD_START + fl.delay * 0.25, BUD_DUR))
            if g <= 0:
                continue
            scale = lerp(0.2, 1.0, g) * fl.scale
            R, H = fl.R, fl.H

            def place(P):
                return (P * scale) @ R.T + H

            u = np.linspace(0, 2 * np.pi, 11)
            q = np.linspace(0.15, np.pi, 6)
            U, Qg = np.meshgrid(u, q)
            hip = np.stack([24 * np.sin(Qg) * np.cos(U), 24 * np.sin(Qg) * np.sin(U),
                            -14 + 20 * np.cos(Qg)], -1)
            self.surface(place(hip), SEPAL_MAT)

            so = ease_in_out(prog(T, SEPAL_START + fl.delay * 0.5, SEPAL_DUR))
            so *= lerp(0.45, 1.0, min(1.0, fl.max_open * 2))
            P = petal_surface(fl.sepals, 12, -8, 108, 19, lerp(0.10, 2.1, so),
                              lerp(0.1, 0.45, so), lerp(0.85, 0.05, so), 0, 0,
                              nv=7, nt=5, shape="sepal")
            self.surface(place(P), SEPAL_MAT, GREEN_EDGE, rib=True)

            pa = fl.pa
            pale = smooth(BUD_START + 0.8, OPEN_START + 1.2 + fl.delay, T)
            bud_green = 0.25 * (1 - fl.max_open)
            o = ease_in_out_v((T - pa["open_at"]) / OPEN_DUR) * fl.max_open
            breathe = 0.03 * np.sin(T * 0.8 + pa["phi"]) * o
            P = petal_surface(
                pa["phi"], lerp(pa["r0"] * 0.5, pa["r0"], o), pa["z0"],
                pa["L"] * lerp(0.82, 1, o), pa["W"] * lerp(0.74, 1, o),
                lerp(pa["ac"], pa["ao"], o) + breathe, lerp(0.05, pa["curl"], o),
                lerp(1.0, pa["wo"], o), pa["lip"] * o, pa["dr"])
            green = (1 - pale) * (0.35 + 0.5 * pa["outer"]) + bud_green * pa["outer"]
            mats = np.stack([material(PETAL_SHADOW, PETAL_LIGHT, 0.7, w, g, 1.0)
                             for w, g in zip(pa["warm"], green)])
            self.surface(place(P), mats, PETAL_EDGE)

    def gyp(self, draw, front):
        b, T = self.b, self.T
        local = T - b.gyp_start
        sel = (b.gyp_front == front) & (local > b.gyp_t)
        if sel.any():
            k = np.clip((local[sel] - b.gyp_t[sel]) / 0.25, 0, 1)
            # the main stem grows over the first 0.5 s
            main = b.gyp_w[sel] > 2
            k = np.where(main, np.clip(local[sel] / 0.5, 0, 1), k)
            A, Bp = b.gyp_a[sel], b.gyp_b[sel]
            E = A + (Bp - A) * k[:, None]
            Xa, Ya, _ = self.project(A)
            Xb, Yb, _ = self.project(E)
            for xa, ya, xb, yb, w in zip(Xa.tolist(), Ya.tolist(), Xb.tolist(), Yb.tolist(),
                                         b.gyp_w[sel].tolist()):
                draw.line([xa, ya, xb, yb], fill=(92, 124, 76), width=max(1, round(SS * 0.3 * w)))
        local = T - b.gf_start
        sel = (b.gf_front == front) & (local > b.gf_t)
        if sel.any():
            k = np.clip((local[sel] - b.gf_t[sel]) / 0.5, 0, 1)
            k = 1 - (1 - k) ** 3
            X, Y, _ = self.project(b.gf_p[sel])
            r = np.maximum(SS * 0.35, b.gf_s[sel] * self.sx * SS * 0.55) * k
            tw = 0.86 + 0.14 * np.sin(T * 2.5 + b.gf_ph[sel])
            for x, y, rr, t in zip(X.tolist(), Y.tolist(), r.tolist(), tw.tolist()):
                c = (int(252 * t), int(250 * t), int(242 * t))
                draw.ellipse([x - rr, y - rr, x + rr, y + rr], fill=c)

    def dust(self, draw):
        T = self.T
        fade = smooth(DUST_START, DUST_START + 2.5, T)
        if fade <= 0:
            return
        d = self.b.dust
        x = -400 + 800 * d[:, 0]
        y = -120 + 240 * d[:, 1]
        z = -300 + np.mod(560 * d[:, 2] + T * (10 + 16 * d[:, 3]), 640)
        X, Y, _ = self.project(np.stack([x, y, z], -1))
        tw = 0.5 + 0.5 * np.sin(T * 2.6 + 40 * d[:, 4])
        edge = np.clip(np.minimum(z + 300, 340 - z) / 80, 0, 1)
        r = SS * 0.45
        for X_, Y_, b in zip(X.tolist(), Y.tolist(), (fade * tw * edge * 230).tolist()):
            if b > 20:
                draw.ellipse([X_ - r, Y_ - r, X_ + r, Y_ + r],
                             fill=(int(b), int(b * 0.97), int(b * 0.92)))

    # --- frame -------------------------------------------------------------
    def paint(self, T, cols, rows, aspect):
        self.setup(T, cols, rows, aspect)
        self.cached("sprigs", T > SPRIG_START + 3.4, self.sprigs)
        self.cached("stems", T > 8.0, self.stems)
        self.cached("ribbon", T > RIBBON_START + 1.2, self.ribbon)
        self.flowers()
        self.flush()
        img = Image.new("RGB", (cols * SS, rows * SS))
        draw = ImageDraw.Draw(img)
        self.gyp(draw, front=False)
        if self.polys:
            polys = np.concatenate(self.polys)
            depth = np.concatenate(self.depth)
            fills = np.concatenate(self.cols)
            flags = np.concatenate(self.flags)
            ecols = np.concatenate(self.ecols)
            # skip slivers (petal bases and tips taper to nothing) and anything off-screen
            xs, ys = polys[:, 0::2], polys[:, 1::2]
            w, h = img.size
            keep = (((xs.max(1) - xs.min(1)) * (ys.max(1) - ys.min(1)) > 0.25)
                    & (xs.max(1) > 0) & (xs.min(1) < w) & (ys.max(1) > 0) & (ys.min(1) < h))
            order = np.nonzero(keep)[0]
            order = order[np.argsort(-depth[order], kind="stable")]
            polys = polys[order].tolist()
            fills = ink(fills[order]).tolist()
            flags = flags[order].tolist()
            ecols = ink(ecols[order]).tolist()
            # Pillow's low-level calls: same result as draw.polygon / draw.line, less overhead
            poly_fn, line_fn, lw = draw.draw.draw_polygon, draw.draw.draw_lines, self.lw
            for p, c, f, e in zip(polys, fills, flags, ecols):
                poly_fn(p, c, 1)
                if f:
                    if f & 1:
                        line_fn(p[0:4], e, lw)
                    if f & 2:
                        line_fn(p[4:8], e, lw)
                    if f & 4:
                        line_fn(p[2:6], e, lw)
        self.gyp(draw, front=True)
        self.dust(draw)
        return np.asarray(img.reduce(SS))


def ink(rgb):
    """Pack (n, 3) colours the way Pillow stores RGB ink (opaque, signed 32-bit)."""
    c = np.clip(np.asarray(rgb), 0, 255).astype(np.int64)
    packed = c[:, 0] | (c[:, 1] << 8) | (c[:, 2] << 16) | (255 << 24)
    return packed.astype(np.uint32).view(np.int32)


# ---------------------------------------------------------------- ascii pass
def to_glyphs(arr, T):
    """Pick a glyph and colour for every cell, like the canvas version."""
    rgb = arr.astype(np.float32)
    m = rgb.max(-1)
    mask = m >= 14
    v = m / 255
    n = len(RAMP) - 1
    yy, xx = np.indices(m.shape, dtype=np.int64)
    h = (xx * 73856093) ^ (yy * 19349663) ^ (int(T * 8) * 83492791)
    jitter = np.where(h % 7 == 0, np.where(h & 8, 1, -1), 0)  # glyph shimmer
    idx = np.clip(np.round(v ** 0.7 * n).astype(int) + jitter, 1, n)
    k = np.minimum(1.25, (40 + 215 * v) / np.maximum(m, 1))
    col = np.minimum(255, rgb * k[..., None]).astype(np.int32) & 0xF8
    return mask, idx, col


def typed(message, T):
    n = int(max(0.0, T - MSG_START) * 14)
    text = message[:n]
    if n < len(message) + 10 and int(T * 2.5) % 2 == 0:
        text += "_"
    return text


# ---------------------------------------------------------------- window mode
def run_window(args):
    if args.snapshot is not None:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame

    pygame.init()
    if args.size:
        W, H = map(int, args.size.lower().split("x"))
    else:
        info = pygame.display.Info()
        H = int(min(info.current_h * 0.86, 1000)) if info.current_h > 0 else 900
        W = int(H * 0.92)
    flags = 0 if args.snapshot is not None else pygame.RESIZABLE
    # window icon: next to the script, or unpacked beside the bundled exe
    here = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    icon = os.path.join(here, "rose_icon.png")
    if os.path.exists(icon):
        pygame.display.set_icon(pygame.image.load(icon))
    screen = pygame.display.set_mode((W, H), flags)
    pygame.display.set_caption("White Roses")

    painter = Painter(Bouquet())
    state = {}

    def layout(W, H):
        fs = int(clamp(min(W / 112, H / 88), 6, 15))
        font = pygame.font.SysFont("consolas,menlo,dejavusansmono,couriernew", fs, bold=True)
        cw, ch = font.size("M")
        msg_font = pygame.font.SysFont("consolas,menlo,dejavusansmono,couriernew",
                                       int(fs * 1.8), italic=True)
        cols, rows = W // cw, H // ch
        msg_rows = 5 if args.message else 1
        state.update(font=font, msg_font=msg_font, cw=cw, ch=ch, cols=cols,
                     rows=rows - msg_rows, aspect=ch / cw, cache={}, W=W, H=H)

    layout(W, H)

    def draw_frame(T):
        s = state
        arr = painter.paint(T, s["cols"], s["rows"], s["aspect"])
        screen.fill((0, 0, 0))
        # glow: a blurred copy of the low-res image underneath the glyphs
        small = pygame.surfarray.make_surface(np.ascontiguousarray(arr.swapaxes(0, 1)))
        half = pygame.transform.smoothscale(small, (max(1, s["cols"] // 2), max(1, s["rows"] // 2)))
        glow = pygame.transform.smoothscale(half, (s["cols"] * s["cw"], s["rows"] * s["ch"]))
        glow.set_alpha(110)
        screen.blit(glow, (0, 0))

        mask, idx, col = to_glyphs(arr, T)
        cache, font, cw, ch = s["cache"], s["font"], s["cw"], s["ch"]
        if len(cache) > 30000:
            cache.clear()
        ys, xs = np.nonzero(mask)
        idx_l = idx[ys, xs].tolist()
        col_l = col[ys, xs].tolist()
        blits = []
        for y, x, i, c in zip(ys.tolist(), xs.tolist(), idx_l, col_l):
            key = (i, c[0], c[1], c[2])
            g = cache.get(key)
            if g is None:
                g = cache[key] = font.render(RAMP[i], True, (c[0], c[1], c[2]))
            blits.append((g, (x * cw, y * ch)))
        screen.blits(blits, doreturn=False)

        if args.message:
            text = typed(args.message, T)
            if text:
                surf = s["msg_font"].render(text, True, (236, 230, 224))
                y = s["rows"] * ch + (s["H"] - s["rows"] * ch - surf.get_height()) // 2
                x = (s["W"] - s["msg_font"].size(args.message + "_")[0]) // 2
                screen.blit(surf, (max(8, x), y))

    if args.snapshot is not None:
        draw_frame(args.snapshot)
        pygame.image.save(screen, args.out)
        print(f"Saved {args.out}")
        return

    clock = pygame.time.Clock()
    t0 = time.perf_counter() - args.t
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.VIDEORESIZE:
                layout(ev.w, ev.h)
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                t0 = time.perf_counter()
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key in (pygame.K_SPACE, pygame.K_r):
                    t0 = time.perf_counter()
                elif ev.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()
                    layout(*screen.get_size())
        draw_frame(time.perf_counter() - t0)
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


# ---------------------------------------------------------------- terminal mode
def run_terminal(args):
    if os.name == "nt":
        os.system("")  # turn on ANSI colour handling in the Windows console
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    out = sys.stdout
    painter = Painter(Bouquet())
    out.write("\x1b[?25l\x1b[2J")
    t0 = time.perf_counter() - args.t
    last_size = None
    try:
        while True:
            frame_start = time.perf_counter()
            T = frame_start - t0
            cols, rows = shutil.get_terminal_size((100, 40))
            rows -= 1
            if (cols, rows) != last_size:
                out.write("\x1b[2J")
                last_size = (cols, rows)
            msg_rows = 2 if args.message else 0
            scene_rows = max(10, rows - msg_rows)
            arr = painter.paint(T, cols, scene_rows, 2.0)
            mask, idx, col = to_glyphs(arr, T)
            lines = []
            for y in range(scene_rows):
                parts, prev = [], None
                m_row, i_row, c_row = mask[y].tolist(), idx[y].tolist(), col[y].tolist()
                for x in range(cols):
                    if not m_row[x]:
                        parts.append(" ")
                        continue
                    c = c_row[x]
                    if c != prev:
                        parts.append(f"\x1b[38;2;{c[0]};{c[1]};{c[2]}m")
                        prev = c
                    parts.append(RAMP[i_row[x]])
                lines.append("".join(parts) + "\x1b[0m")
            if args.message:
                text = typed(args.message, T)
                pad = max(0, (cols - len(args.message)) // 2)
                lines.append("")
                lines.append("\x1b[38;2;236;230;224m\x1b[3m" + (" " * pad + text).ljust(cols)[:cols] + "\x1b[0m")
            out.write("\x1b[H" + "\n".join(lines))
            out.flush()
            if args.duration and T > args.duration:
                break
            time.sleep(max(0.0, 1 / 20 - (time.perf_counter() - frame_start)))
    except KeyboardInterrupt:
        pass
    finally:
        out.write("\x1b[0m\x1b[?25h\n")
        out.flush()


def main():
    p = argparse.ArgumentParser(description="A bouquet of white roses that blooms in ASCII.")
    p.add_argument("--terminal", action="store_true", help="draw in the terminal instead of a window")
    p.add_argument("--message", default="white roses, just for you")
    p.add_argument("--t", type=float, default=0.0, help="start this many seconds into the bloom")
    p.add_argument("--size", help="window size, e.g. 900x1000")
    p.add_argument("--snapshot", type=float, help="render the frame at this time and save it")
    p.add_argument("--out", default="white_roses.png", help="file for --snapshot")
    p.add_argument("--duration", type=float, help="terminal mode: stop after this many seconds")
    args = p.parse_args()

    if args.terminal:
        run_terminal(args)
        return
    try:
        import pygame  # noqa: F401
    except ImportError:
        print("pygame not found, drawing in the terminal instead (py -m pip install pygame)")
        time.sleep(1.5)
        run_terminal(args)
        return
    run_window(args)


if __name__ == "__main__":
    main()
