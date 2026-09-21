"""
control.py -- per-agent control stack for a hexacopter:

  (1) LinearMPC       : finite-horizon condensed MPC on the translational
                        double-integrator -> commanded acceleration a_cmd.
  (2) cbf_safety_filter: high-order CBF QP that minimally alters a_cmd to keep
                        pairwise separation (decentralized collision avoidance).
  (3) flatness         : a_cmd -> (collective thrust T, desired attitude q_des)
                        via multirotor differential flatness (Mellinger-Kumar).
  (4) geometric_attitude: SO(3) attitude tracking -> body torques (Lee et al.).

The MPC here is the reference stand-in, at the translational level, for the
real-time nonlinear MPC each agent runs on hardware; the report is explicit
about this. The structure (predict -> optimize -> apply the first move, with
safety constraints imposed on the applied input) is the same, and both sit
behind the same interface (see c_api/uav_solver.h).
"""
import numpy as np
from .quaternion import quat_to_rot, hat
from .qpsolve import solve_qp

G = 9.81


class LinearMPC:
    """
    Condensed linear MPC for a 1-axis double integrator, applied per axis.
    Model:  s_{k+1} = A s_k + B a_k,  s=[pos,vel],  a=accel.
    Cost:   sum ||p_k - p_ref||^2 Qp + ||v_k||^2 Qv + ||a_k||^2 Ru  over horizon.
    Returns the first optimal acceleration (receding horizon).
    """
    def __init__(self, dt=0.05, N=20, Qp=8.0, Qv=1.5, Ru=0.4, a_max=8.0):
        self.dt, self.N = dt, N
        self.a_max = a_max
        A = np.array([[1, dt], [0, 1.0]])
        B = np.array([[0.5*dt*dt], [dt]])
        # build condensed prediction  P = Sx s0 + Su A_seq
        nx, nu = 2, 1
        Sx = np.zeros((nx*N, nx))
        Su = np.zeros((nx*N, nu*N))
        Apow = np.eye(nx)
        for i in range(N):
            Apow = Apow @ A if i > 0 else A
            Sx[nx*i:nx*i+nx, :] = Apow
            Aj = np.eye(nx)
            for j in range(i+1):
                # coefficient of a_{i-j}
                blk = Aj @ B
                col = i - j
                Su[nx*i:nx*i+nx, nu*col:nu*col+nu] = blk
                Aj = A @ Aj
        self.A, self.B, self.Sx, self.Su = A, B, Sx, Su
        Qp_ = np.tile([Qp, Qv], N)
        self.Qbar = np.diag(Qp_)
        self.Rbar = Ru*np.eye(N)
        # precompute condensed Hessian
        self.Hc = Su.T @ self.Qbar @ Su + self.Rbar
        self.Hc_inv = np.linalg.inv(self.Hc)

    def solve_axis(self, s0, p_ref_seq):
        """s0=[p,v]; p_ref_seq length N of reference positions -> first accel."""
        ref = np.zeros(2*self.N)
        ref[0::2] = p_ref_seq
        # gradient: Su^T Q (Sx s0 - ref)
        g = self.Su.T @ self.Qbar @ (self.Sx @ s0 - ref)
        a_seq = -self.Hc_inv @ g
        a0 = a_seq[0]
        return np.clip(a0, -self.a_max, self.a_max)

    def command(self, p, v, p_ref_traj):
        """3-axis: p,v in R^3, p_ref_traj (N,3) preview -> a_cmd in R^3."""
        a = np.zeros(3)
        for ax in range(3):
            s0 = np.array([p[ax], v[ax]])
            a[ax] = self.solve_axis(s0, p_ref_traj[:, ax])
        return a


def cbf_safety_filter(i, p, v, a_nom, neighbors, D_safe=1.2,
                      k1=3.0, k2=3.0, a_max=10.0):
    """
    High-order CBF QP for pairwise collision avoidance (relative degree 2).
    h_ij = ||p_i-p_j||^2 - D^2,  enforce  hddot + k1 hdot + k2 h >= 0.
    Decentralized: agent i actuates its own accel, treats neighbor accel = 0.
    min ||a - a_nom||^2  s.t. the CBF rows.   Returns safe accel.
    """
    pi, vi = p[i], v[i]
    rows_A, rows_b = [], []
    for j in neighbors:
        if j == i:
            continue
        dp = pi - p[j]
        dv = vi - v[j]
        dist2 = dp @ dp
        h = dist2 - D_safe**2
        hdot = 2*dp @ dv
        # hddot = 2 dv.dv + 2 dp . (a_i - a_j);  a_j treated as 0
        # constraint: 2 dv.dv + 2 dp.a_i + k1 hdot + k2 h >= 0
        #   ->  -2 dp . a_i <= 2 dv.dv + k1 hdot + k2 h
        rows_A.append(-2*dp)
        rows_b.append(2*(dv @ dv) + k1*hdot + k2*h)
    H = 2*np.eye(3)
    g = -2*a_nom
    # add box on accel as constraints too (keeps QP well posed)
    if rows_A:
        A = np.array(rows_A)
        b = np.array(rows_b)
    else:
        A, b = None, None
    a_safe, _ = solve_qp(H, g, A, b)
    return np.clip(a_safe, -a_max, a_max)


def flatness(a_cmd, m, yaw_des=0.0, tilt_max=np.deg2rad(30.0)):
    """
    Multirotor differential flatness (z-up world, thrust +body z):
      required specific force  f = a_cmd + g e3
      thrust  T = m ||f||,  b3_des = f/||f||
      build R_des from b3_des and desired yaw.
    A tilt limit caps the horizontal specific force so the commanded attitude
    never exceeds tilt_max from vertical -- essential so the inner attitude loop
    can track the outer command without the vehicle flipping (standard practice).
    Returns (T, q_des, R_des).
    """
    e3 = np.array([0, 0, 1.0])
    a = a_cmd.copy()
    a_h = a[0:2]
    a_v = a[2] + G                                  # vertical specific force
    a_v = max(a_v, 0.5*G)                            # keep positive thrust margin
    h_norm = np.linalg.norm(a_h)
    h_max = a_v*np.tan(tilt_max)                     # max horizontal at this thrust
    if h_norm > h_max and h_norm > 1e-9:
        a_h = a_h*(h_max/h_norm)
    f = np.array([a_h[0], a_h[1], a_v])
    Tmag = m*np.linalg.norm(f)
    if np.linalg.norm(f) < 1e-6:
        b3 = e3.copy()
    else:
        b3 = f/np.linalg.norm(f)
    b1c = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])
    b2 = np.cross(b3, b1c)
    if np.linalg.norm(b2) < 1e-6:
        b2 = np.array([0, 1.0, 0])
    b2 = b2/np.linalg.norm(b2)
    b1 = np.cross(b2, b3)
    R_des = np.column_stack([b1, b2, b3])
    return Tmag, _rot_to_quat(R_des), R_des


def _rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr+1.0)*2
        qw = 0.25*S
        qx = (R[2, 1]-R[1, 2])/S
        qy = (R[0, 2]-R[2, 0])/S
        qz = (R[1, 0]-R[0, 1])/S
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            S = np.sqrt(1.0+R[0, 0]-R[1, 1]-R[2, 2])*2
            qw = (R[2, 1]-R[1, 2])/S; qx = 0.25*S
            qy = (R[0, 1]+R[1, 0])/S; qz = (R[0, 2]+R[2, 0])/S
        elif i == 1:
            S = np.sqrt(1.0+R[1, 1]-R[0, 0]-R[2, 2])*2
            qw = (R[0, 2]-R[2, 0])/S; qx = (R[0, 1]+R[1, 0])/S
            qy = 0.25*S; qz = (R[1, 2]+R[2, 1])/S
        else:
            S = np.sqrt(1.0+R[2, 2]-R[0, 0]-R[1, 1])*2
            qw = (R[1, 0]-R[0, 1])/S; qx = (R[0, 2]+R[2, 0])/S
            qy = (R[1, 2]+R[2, 1])/S; qz = 0.25*S
    q = np.array([qw, qx, qy, qz])
    return q/np.linalg.norm(q)


def geometric_attitude(q, w, R_des, J, w_des=None, kR=None, kw=None):
    """
    SO(3) geometric attitude controller (Lee, Leok, McClamroch 2010).
      e_R = 1/2 (R_des^T R - R^T R_des)^vee
      e_w = w - R^T R_des w_des
      tau = -kR e_R - kw e_w + w x J w   (feedforward)
    """
    if w_des is None:
        w_des = np.zeros(3)
    if kR is None:
        kR = np.diag([2.2, 2.2, 0.6])
    if kw is None:
        kw = np.diag([0.45, 0.45, 0.2])
    R = quat_to_rot(q)
    E = 0.5*(R_des.T @ R - R.T @ R_des)
    e_R = np.array([E[2, 1], E[0, 2], E[1, 0]])
    e_w = w - R.T @ R_des @ w_des
    tau = -kR @ e_R - kw @ e_w + np.cross(w, J @ w)
    return tau
