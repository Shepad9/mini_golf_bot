"""Rolling/sliding-ball physics on a height field, with spin.

This is a rigid-sphere contact model rather than the earlier "always rolling"
shortcut. The ball carries translational velocity v (2D, in the surface plane)
and angular velocity omega (3D). Friction acts at the contact point, and the
*slip velocity* there decides the regime:

    u = (v_x - R*omega_y, v_y + R*omega_x)        # contact-point slip (2D)

  * |u| > tol  -> SLIDING. Kinetic friction mu_k*N opposes u, decelerating the
    centre AND torquing the ball (spinning it up). This is the skid that follows
    every strike and every cushion rebound. With a struck ball (omega = 0) the
    slip falls to zero at v = 5/7 * v_strike - the textbook skid->roll result,
    here emergent rather than hard-coded.
  * |u| ~ 0    -> ROLLING. The centre obeys a = (5/7)*g_slope (the 5/7 now comes
    from the rolling constraint, not an input), and rolling resistance bleeds
    speed. Resistance has two parts: a Coulomb term c_rr*g (constant, dominant at
    low speed) and a viscous term k_v*v (vanishes at rest). c_rr alone sets the
    static hold slope (a ball holds up to tan(theta) = 1.4*c_rr); c_rr and k_v
    together set the roll-out distance (calibrated to green speed / Stimpmeter).
    The two are therefore independent knobs.

Cushions use the standard billiard model: normal restitution e, plus Coulomb
tangential friction coupled to vertical spin omega_z (so side-spin bends the
rebound and the bounce scrubs tangential speed). Hole capture uses Holmes'
criterion: the ball is captured if free-fall lets it drop one ball-radius while
its centre is over the cup - reproducing drop / skip-over / lip-out from speed
and offset alone.

Bodies of mass cancel throughout (everything is written per unit mass / as
velocity impulses), so no mass parameter is needed. Solid sphere: I = (2/5)mR^2.
"""

from __future__ import annotations
import numpy as np

from geometry import closest_boundary

G = 9.81
SLIP_TOL = 0.01            # m/s; below this the contact is treated as rolling
REST_SPEED = 0.035         # m/s; candidate speed for coming to rest
SPIN_DECAY = 1.5           # 1/s; slow drilling-friction decay of vertical spin


class Ball:
    def __init__(self, pos, radius=0.021, restitution=0.5, wall_friction=0.2):
        self.p = np.asarray(pos, float)
        self.v = np.zeros(2)               # translational velocity (plane)
        self.omega = np.zeros(3)           # angular velocity (3D)
        self.radius = float(radius)
        self.restitution = float(restitution)
        self.wall_friction = float(wall_friction)
        self.resting = False
        self.holed = False

    def strike(self, velocity):
        """Hit the ball: set translational velocity, no spin (a clean putt
        slides first, then rolls). Clears the resting flag."""
        self.v = np.asarray(velocity, float).copy()
        self.omega[:] = 0.0
        self.resting = False


def _slope_accel(course, p):
    """In-plane gravitational acceleration of the centre on the tangent plane,
    a = -g * grad(h) / (1 + |grad h|^2). NOT scaled by 5/7 here - that factor
    belongs to the rolling phase and is applied in step()."""
    g = course.gradient(p[0], p[1])
    s2 = 1.0 + float(g[0] ** 2 + g[1] ** 2)
    return -G * g / s2


def _normal_gravity(course, p):
    """g * cos(theta): the component pressing the ball onto the surface, used to
    scale floor friction/resistance with slope."""
    g = course.gradient(p[0], p[1])
    return G / np.sqrt(1.0 + float(g[0] ** 2 + g[1] ** 2))


def _slip(v, omega, R):
    return np.array([v[0] - R * omega[1], v[1] + R * omega[0]])


def _roll_spin(v, omega, R):
    """Angular velocity consistent with rolling without slipping for this v
    (leaves the vertical-axis spin omega_z untouched)."""
    return np.array([-v[1] / R, v[0] / R, omega[2]])


def _resolve_walls(course, ball):
    """Cushion collisions: normal restitution + tangential Coulomb friction
    coupled to vertical spin. Iterated a few times for tight corners."""
    R = ball.radius
    for _ in range(4):
        hit = closest_boundary(course.boundaries, ball.p)
        if hit is None:
            return
        _, c, n, dist = hit
        if dist >= R:
            return
        # push out to contact along the inward normal n
        ball.p = ball.p + n * (R - dist + 1e-6)
        vn = float(np.dot(ball.v, n))
        if vn >= 0:                      # separating already; no impulse
            continue
        t = np.array([-n[1], n[0]])      # in-plane tangent
        # --- normal restitution
        dvn = (1.0 + ball.restitution) * (-vn)     # >0, the change in v.n
        ball.v = ball.v + dvn * n
        # --- tangential friction with spin coupling
        vt = float(np.dot(ball.v, t))
        wz = float(ball.omega[2])
        s = vt - R * wz                  # tangential surface speed at contact
        jt_roll = -(2.0 / 7.0) * s       # tangential dv that would kill slip
        jt_max = ball.wall_friction * dvn
        jt = float(np.clip(jt_roll, -jt_max, jt_max))
        ball.v = ball.v + jt * t
        ball.omega[2] = wz - (5.0 / (2.0 * R)) * jt


def _check_capture(course, ball, p0):
    """Holmes capture test over the swept segment p0 -> ball.p.

    The ball is captured if its centre passes within the cup mouth (offset
    b < R_hole) and is slow enough that, during the time its centre spends over
    the opening, gravity pulls it down by at least a ball radius:

        v <= sqrt( 2 g (R_hole^2 - b^2) / r )

    Faster or more off-centre balls fail the test -> they skip across or lip out.
    """
    H = course.hole
    A = p0
    B = ball.p
    AB = B - A
    denom = float(np.dot(AB, AB))
    if denom < 1e-12:
        b = float(np.hypot(*(H - A)))
    else:
        tt = float(np.clip(np.dot(H - A, AB) / denom, 0.0, 1.0))
        b = float(np.hypot(*(H - (A + tt * AB))))
    Rh = course.hole_radius
    if b >= Rh:
        return False
    speed = float(np.hypot(*ball.v))
    v_crit = np.sqrt(max(2.0 * G * (Rh * Rh - b * b) / ball.radius, 0.0))
    if speed <= v_crit:
        ball.holed = True
        ball.v[:] = 0.0
        ball.omega[:] = 0.0
        ball.p = H.copy()
        return True
    return False


def step(course, ball, dt):
    """Advance the ball by dt seconds. Returns True while still in motion."""
    if ball.holed or ball.resting:
        return False
    R = ball.radius

    speed = float(np.hypot(*ball.v))
    max_move = 0.4 * R
    nsub = int(np.ceil(max(1.0, speed * dt / max_move)))
    nsub = min(nsub, 64)
    h = dt / nsub

    for _ in range(nsub):
        p0 = ball.p.copy()
        a_grav = _slope_accel(course, ball.p)
        gn = _normal_gravity(course, ball.p)            # = g cos(theta)
        u = _slip(ball.v, ball.omega, R)
        slip_speed = float(np.hypot(*u))

        if slip_speed > SLIP_TOL:
            # ---- SLIDING: kinetic friction at the contact
            mu_k, _, _ = course.surface_props(ball.p[0], ball.p[1])
            uhat = u / slip_speed
            a_fric = -mu_k * gn * uhat                  # decelerates the centre
            # angular acceleration from the friction force (solid sphere)
            alpha = (5.0 / (2.0 * R)) * np.array([a_fric[1], -a_fric[0], 0.0])
            v_new = ball.v + (a_fric + a_grav) * h
            w_new = ball.omega + alpha * h
            # detect skid->roll: slip crossed through zero this sub-step
            u_new = _slip(v_new, w_new, R)
            if float(np.dot(u_new, u)) <= 0.0 or np.hypot(*u_new) < SLIP_TOL:
                ball.v = v_new
                ball.omega = _roll_spin(v_new, w_new, R)   # snap to rolling
            else:
                ball.v, ball.omega = v_new, w_new
        else:
            # ---- ROLLING: 5/7 of gravity drives the centre; rolling resistance
            # bleeds speed. Resistance = c_rr*g (Coulomb, low-speed/hold) +
            # k_v*v (viscous, vanishes at rest). The two are independent knobs.
            _, c_rr, k_v = course.surface_props(ball.p[0], ball.p[1])
            ball.v = ball.v + (5.0 / 7.0) * a_grav * h
            sp = float(np.hypot(*ball.v))
            if sp > 1e-9:
                dv = min((c_rr * gn + k_v * sp) * h, sp)
                ball.v = ball.v - (ball.v / sp) * dv
            ball.omega = _roll_spin(ball.v, ball.omega, R)
            ball.omega[2] *= np.exp(-SPIN_DECAY * h)       # drilling friction

        ball.p = ball.p + ball.v * h
        _resolve_walls(course, ball)
        if _check_capture(course, ball, p0):
            return False

    # Rest test: crawling, and the slope can no longer drive the ball. When the
    # ball is resting against a wall, the wall cancels the into-wall component of
    # gravity, so only the ALONG-wall part counts toward the drive - otherwise a
    # ball pressed into a wall by a slope can never satisfy the rest test and
    # creeps/jitters along the wall forever.
    sp = float(np.hypot(*ball.v))
    if sp < REST_SPEED:
        _, c_rr, _ = course.surface_props(ball.p[0], ball.p[1])
        gn = _normal_gravity(course, ball.p)
        a_g = _slope_accel(course, ball.p)
        hit = closest_boundary(course.boundaries, ball.p)
        if hit is not None:
            _, _, n, dist = hit
            an = float(np.dot(a_g, n))
            if dist <= R + 1e-3 and an < 0.0:      # pressed into the wall
                a_g = a_g - an * n                 # wall absorbs the normal part
        drive = (5.0 / 7.0) * float(np.hypot(*a_g))
        # Hold threshold depends only on the Coulomb term: viscous drag is zero
        # at rest, so it cannot help hold the ball on a slope.
        if drive <= c_rr * gn:
            ball.v[:] = 0.0
            ball.omega[:] = 0.0
            ball.resting = True
            return False
    return True


def simulate(course, ball, dt=1 / 240, max_time=30.0, record=False):
    """Run until the ball stops or is holed. Optionally record the trajectory."""
    t = 0.0
    path = [ball.p.copy()]
    moving = True
    while moving and t < max_time:
        moving = step(course, ball, dt)
        t += dt
        if record:
            path.append(ball.p.copy())
    return np.array(path) if record else None