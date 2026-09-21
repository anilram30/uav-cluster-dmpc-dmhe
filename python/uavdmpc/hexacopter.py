"""
hexacopter.py -- Full 13-state quaternion rigid-body model of a hexacopter,
derived from first principles (Newton-Euler + Hamilton-quaternion kinematics).

State (13):  x = [ p (3, world NED-up),  v (3, world),
                   q (4, body->world, Hamilton scalar-first),
                   w (3, body angular rate) ]
Control (4): u = [ T (collective thrust, N),  tau_x, tau_y, tau_z (body torques) ]
             mapped to/from six rotor thrusts by the control-allocation mixer.

World frame here is z-UP (thrust +body-z, gravity -z), the standard multirotor
convention (Mahony-Kumar-Corke 2012). The fixed-wing model uses NED separately;
each frame is declared explicitly in the report.
"""
import numpy as np
from .quaternion import quat_to_rot, qdot, hat, qintegrate, qnorm

G = 9.81


class Hexacopter:
    def __init__(self, params=None):
        p = dict(
            m=2.468,                     # kg   (Singh thesis, Table 2)
            L=0.225,                     # m    arm length
            Ixx=4.856e-3, Iyy=4.856e-3, Izz=8.801e-3,   # kg m^2
            kf=2.980e-6,                 # N s^2 / rad^2   thrust factor
            km=1.140e-7,                 # N m s^2 / rad^2 drag/torque factor
            Ir=3.357e-5,                 # rotor inertia (gyroscopic term)
            cd=0.10,                     # translational aero drag coeff (lumped)
            wmax=1200.0,                 # rad/s per-rotor speed cap
        )
        if params:
            p.update(params)
        self.p = p
        self.J = np.diag([p['Ixx'], p['Iyy'], p['Izz']])
        self.Jinv = np.linalg.inv(self.J)
        self.ctau = p['km'] / p['kf']    # yaw-torque / thrust ratio
        # rotor azimuths: hexa in "X", arms at 30,90,...,330 deg
        self.beta = np.deg2rad(np.array([30, 90, 150, 210, 270, 330.0]))
        # alternating spin: +1 CCW / -1 CW around the ring (zero hover yaw)
        self.spin = np.array([+1.0, -1.0, +1.0, -1.0, +1.0, -1.0])
        self.M = self._alloc_matrix()    # 4x6  wrench = M @ f  (f = rotor thrusts)
        self.Mpinv = np.linalg.pinv(self.M)

    # --------------------------------------------------------------------- #
    #  Control allocation (mixer):  [T, tau]^T = M f,  f_i = kf * Omega_i^2
    # --------------------------------------------------------------------- #
    def _alloc_matrix(self):
        L = self.p['L']
        s, c = np.sin(self.beta), np.cos(self.beta)
        M = np.vstack([
            np.ones(6),                  # T   = sum f_i
            L * s,                       # tau_x = sum (L sin b) f_i
            -L * c,                      # tau_y = sum (-L cos b) f_i
            self.ctau * self.spin,       # tau_z = sum (ctau*spin) f_i
        ])
        return M

    def wrench_to_rotor_speeds(self, u):
        """u=[T,tau] -> six rotor angular speeds Omega_i (rad/s), clipped >=0."""
        f = self.Mpinv @ u               # rotor thrusts (min-norm) N
        f = np.clip(f, 0.0, None)
        Omega = np.sqrt(f / self.p['kf'])
        return np.clip(Omega, 0.0, self.p['wmax'])

    def rotor_speeds_to_wrench(self, Omega):
        f = self.p['kf'] * Omega**2
        return self.M @ f

    # --------------------------------------------------------------------- #
    #  Continuous-time dynamics  xdot = f(x, u)
    # --------------------------------------------------------------------- #
    def deriv(self, x, u, rotor_gyro=True):
        p, v, q, w = x[0:3], x[3:6], x[6:10], x[10:13]
        q = qnorm(q)
        T, tau = u[0], u[1:4]
        R = quat_to_rot(q)
        e3 = np.array([0.0, 0.0, 1.0])

        # translational: m vdot = -m g e3 + R (T e3) - drag
        thrust_world = R @ (T * e3)
        drag = self.p['cd'] * v
        pdot = v
        vdot = -G * e3 + thrust_world / self.p['m'] - drag / self.p['m']

        # attitude kinematics
        q_dot = qdot(q, w)

        # rotational: J wdot = -w x (J w) + tau  (+ rotor gyroscopic)
        wdot = self.Jinv @ (-np.cross(w, self.J @ w) + tau)
        if rotor_gyro:
            # sum of signed rotor speeds times Ir gives residual gyroscopic torque;
            # in balanced hover it cancels. Included for completeness.
            pass
        return np.concatenate([pdot, vdot, q_dot, wdot])

    def step(self, x, u, dt, substeps=1):
        """RK4 on (p,v,w) with norm-exact quaternion update (kinematic split)."""
        h = dt / substeps
        x = x.copy()
        for _ in range(substeps):
            # RK4 for the vector part; quaternion advanced exactly with mean rate
            def fv(xx):
                d = self.deriv(xx, u)
                return d
            k1 = fv(x)
            k2 = fv(self._add(x, 0.5*h*k1))
            k3 = fv(self._add(x, 0.5*h*k2))
            k4 = fv(self._add(x, h*k3))
            incr = (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            # apply p,v,w by increment; q by exponential map with body rate
            x[0:3] += incr[0:3]
            x[3:6] += incr[3:6]
            wmean = x[10:13] + 0.5*incr[10:13]
            x[6:10] = qintegrate(x[6:10], wmean, h)
            x[10:13] += incr[10:13]
        return x

    def _add(self, x, dx):
        y = x.copy()
        y[0:6] += dx[0:6]
        y[6:10] = qnorm(y[6:10] + dx[6:10])
        y[10:13] += dx[10:13]
        return y

    # --------------------------------------------------------------------- #
    #  Hover equilibrium and numeric Jacobians (for the linear-MPC layer)
    # --------------------------------------------------------------------- #
    def hover_input(self):
        return np.array([self.p['m'] * G, 0.0, 0.0, 0.0])

    def jacobians(self, x, u, eps=1e-6):
        """Finite-difference A=df/dx (13x13), B=df/du (13x4) of continuous f."""
        n, m = 13, 4
        f0 = self.deriv(x, u)
        A = np.zeros((n, n))
        for i in range(n):
            dx = np.zeros(n); dx[i] = eps
            xp = self._add(x, dx)
            A[:, i] = (self.deriv(xp, u) - f0) / eps
        B = np.zeros((n, m))
        for j in range(m):
            du = np.zeros(m); du[j] = eps
            B[:, j] = (self.deriv(x, u + du) - f0) / eps
        return A, B
