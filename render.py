"""Matplotlib renderer + interactive play.

`draw_course` paints the board (height as a shaded contour, friction patches,
walls, tee, cup) onto an Axes and is used both for the static figure and the
live game. `InteractiveGame` adds click-and-drag aiming: press near the ball,
drag to set direction/power, release to putt; the shot is then animated.

Matplotlib is chosen over a game library mainly because it renders reliably under
WSL2/TkAgg without extra display plumbing. The drawing is backend-agnostic, so
swapping in pygame later only means re-implementing this one file.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

from geometry import Segment, Arc
from physics import Ball, step, simulate


def select_interactive_backend():
    """Try to switch Matplotlib to a GUI backend so a window can open.

    Returns the backend name on success, or None if only headless Agg is
    available (e.g. python3-tk not installed, or no DISPLAY). Must be called
    before any Figure is created - which is why main.py calls it before
    constructing InteractiveGame.
    """
    import matplotlib
    if matplotlib.get_backend().lower() != "agg":
        return matplotlib.get_backend()
    for bk in ("TkAgg", "QtAgg", "Qt5Agg", "GTK3Agg", "wxAgg", "MacOSX"):
        try:
            plt.switch_backend(bk)
            return bk
        except Exception:
            continue
    return None


def record_shot(course, velocity, filename="shot.gif", fps=60, max_frames=600):
    """Headless fallback: simulate one shot and save it as an animated GIF.

    Lets you *see* the physics without any GUI backend. Uses Pillow as the
    writer (no ffmpeg needed). The trajectory is simulated first, then the
    recorded path is sampled down to keep the GIF light.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle

    ball = Ball(course.start.copy())
    ball.strike(velocity)
    path = simulate(course, ball, dt=1 / fps, record=True)
    if len(path) > max_frames:                       # subsample long rolls
        idx = np.linspace(0, len(path) - 1, max_frames).astype(int)
        path = path[idx]

    fig, ax = plt.subplots(figsize=(8, 8 * course.height_m / course.width))
    draw_course(ax, course)
    bp = Circle(path[0], 0.045, color="white", ec="black", zorder=9)
    ax.add_patch(bp)
    title = "HOLED OUT" if ball.holed else "ball at rest"
    ax.set_title(f"recorded shot  -  {title}", fontsize=11)

    def update(i):
        bp.center = tuple(path[i])
        return (bp,)

    anim = FuncAnimation(fig, update, frames=len(path),
                         interval=1000 / fps, blit=True, cache_frame_data=False)
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return filename, ball.holed


def _wall_polylines(boundaries, arc_pts=24):
    """Sample each wall primitive to a polyline purely for drawing."""
    segs = []
    for prim in boundaries:
        if isinstance(prim, Segment):
            segs.append([prim.a, prim.b])
        elif isinstance(prim, Arc):
            a0, a1 = prim.a0, prim.a1
            ts = np.linspace(a0, a1, arc_pts)
            pts = prim.c + prim.r * np.column_stack([np.cos(ts), np.sin(ts)])
            segs.append(pts)
    return segs


def draw_course(ax, course, show_height=True):
    ax.clear()
    nx, ny = course.nx, course.ny
    xs = np.linspace(0, course.width, nx)
    ys = np.linspace(0, course.height_m, ny)
    if show_height:
        ax.contourf(xs, ys, course.H, levels=18, cmap="YlGn", alpha=0.9)
        ax.contour(xs, ys, course.H, levels=10, colors="k",
                   linewidths=0.3, alpha=0.25)

    for s in course.surfaces:
        if s.center is not None:
            col = "#d9c089" if s.c_rr > course.base_c_rr else "#bfe3ef"
            ax.add_patch(Circle(s.center, s.radius, color=col, alpha=0.55,
                                zorder=2, lw=0))

    lc = LineCollection(_wall_polylines(course.boundaries),
                        colors="#3a2c12", linewidths=3.0, zorder=5)
    ax.add_collection(lc)

    ax.add_patch(Circle(course.hole, course.hole_radius, color="black", zorder=6))
    ax.add_patch(Circle(course.hole, course.hole_radius * 0.55,
                        color="#202020", zorder=6))
    ax.plot(*course.start, marker="o", ms=7, color="#1d6fb8", zorder=6)
    ax.text(course.start[0], course.start[1] + 0.07, "tee", ha="center",
            fontsize=8, zorder=7)

    ax.set_xlim(-0.1, course.width + 0.1)
    ax.set_ylim(-0.1, course.height_m + 0.1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def render_static(course, path=None, ball_pos=None, filename=None, title=None):
    """One-shot figure; optionally overlay a recorded trajectory `path`."""
    fig, ax = plt.subplots(figsize=(8, 8 * course.height_m / course.width))
    draw_course(ax, course)
    if path is not None and len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], "-", color="crimson", lw=1.6, zorder=8)
    bp = ball_pos if ball_pos is not None else course.start
    ax.add_patch(Circle(bp, 0.045, color="white", ec="black", zorder=9))
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=110)
    return fig, ax


class InteractiveGame:
    POWER = 6.0   # m/s of launch speed per metre of drag

    def __init__(self, course):
        self.course = course
        self.ball = Ball(course.start.copy())
        self.fig, self.ax = plt.subplots(
            figsize=(9, 9 * course.height_m / course.width))
        self.dragging = False
        self.drag_start = None
        self.aim_line = None
        self.ball_patch = Circle(self.ball.p, 0.045, color="white",
                                 ec="black", zorder=9)
        self.strokes = 0
        self.timer = None
        self._draw()
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)

    def _draw(self):
        draw_course(self.ax, self.course)
        self.ball_patch = Circle(self.ball.p, 0.045, color="white",
                                 ec="black", zorder=9)
        self.ax.add_patch(self.ball_patch)
        self._title()
        self.fig.canvas.draw_idle()

    def _title(self):
        msg = f"Strokes: {self.strokes}"
        if self.ball.holed:
            msg += "  -  HOLED OUT!"
        elif self.ball.resting:
            msg += "  -  drag from the ball to putt"
        self.ax.set_title(msg, fontsize=11)

    def on_press(self, event):
        if event.inaxes != self.ax or self.ball.holed:
            return
        if not (self.ball.resting or self.strokes == 0):
            return
        if np.hypot(event.xdata - self.ball.p[0],
                    event.ydata - self.ball.p[1]) < 0.25:
            self.dragging = True
            self.drag_start = np.array([event.xdata, event.ydata])

    def on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax:
            return
        if self.aim_line:
            self.aim_line.remove()
        # aim opposite the drag (pull back like a slingshot)
        tip = 2 * self.ball.p - np.array([event.xdata, event.ydata])
        (self.aim_line,) = self.ax.plot(
            [self.ball.p[0], tip[0]], [self.ball.p[1], tip[1]],
            "-", color="crimson", lw=2, zorder=8)
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if not self.dragging:
            return
        self.dragging = False
        if event.inaxes != self.ax or event.xdata is None:
            return
        pull = self.drag_start - np.array([event.xdata, event.ydata])
        self.ball.strike(pull * self.POWER)
        self.strokes += 1
        if self.aim_line:
            self.aim_line.remove(); self.aim_line = None
        self._animate()

    def _animate(self):
        from matplotlib.animation import FuncAnimation

        def update(_):
            moving = step(self.course, self.ball, 1 / 120) #120 fps
            self.ball_patch.center = tuple(self.ball.p)
            self._title()
            if not moving:
                self.anim.event_source.stop()
            return (self.ball_patch,)

        self.anim = FuncAnimation(self.fig, update, interval=16,
                                  blit=False, cache_frame_data=False)
        self.fig.canvas.draw_idle()

    def play(self):
        import matplotlib
        if matplotlib.get_backend().lower() == "agg":
            print("No interactive backend available - cannot open a window.\n"
                  "Saving the board to board.png instead. To play interactively,\n"
                  "install a GUI backend (see the note printed by main.py), or run\n"
                  "  python main.py --record\nto save an animated GIF of a shot.")
            self.fig.savefig("board.png", dpi=110)
            return
        plt.show()