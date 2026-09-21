"""
quaternion.py -- Hamilton unit-quaternion algebra used throughout the
distributed UAV estimation/control framework.

Convention (fixed for the entire project):
    * Hamilton product (NOT JPL), ijk = -1.
    * q = [q_w, q_x, q_y, q_z] = [q_w, q_v],  scalar-first.
    * q maps BODY -> WORLD:  v_world = R(q) v_body.
    * Right-handed frames.

Every routine here mirrors, one-to-one, the equations stated in the report
(Sec. "Preliminaries: rotations and quaternions").
"""
import numpy as np

# --------------------------------------------------------------------------- #
#  Core products
# --------------------------------------------------------------------------- #
def qmul(p, q):
    """Hamilton product  p ⊗ q  (scalar-first)."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ])


def qconj(q):
    """Quaternion conjugate  q* = [q_w, -q_v]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qnorm(q):
    return q / np.linalg.norm(q)


def hat(w):
    """Skew-symmetric cross-product matrix [w]_x  with  [w]_x a = w × a."""
    x, y, z = w
    return np.array([[0.0, -z,  y],
                     [z,  0.0, -x],
                     [-y,  x, 0.0]])


# --------------------------------------------------------------------------- #
#  Rotation matrix  R(q) : body -> world      (eq. R-quat in the report)
# --------------------------------------------------------------------------- #
def quat_to_rot(q):
    qw, qx, qy, qz = q
    # R = (q_w^2 - q_v^T q_v) I + 2 q_v q_v^T + 2 q_w [q_v]_x
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)],
    ])


# --------------------------------------------------------------------------- #
#  Attitude kinematics  qdot = 1/2 q ⊗ [0, omega]  (eq. q-kin)
#  Written as the linear map  qdot = 1/2 Omega(omega) q  as well.
# --------------------------------------------------------------------------- #
def Omega(w):
    """4x4 matrix with  1/2 Omega(w) q  =  1/2 q ⊗ [0,w]."""
    x, y, z = w
    return np.array([
        [0.0, -x, -y, -z],
        [x,  0.0,  z, -y],
        [y,  -z, 0.0,  x],
        [z,   y,  -x, 0.0],
    ])


def qdot(q, w):
    return 0.5 * qmul(q, np.array([0.0, w[0], w[1], w[2]]))


def qintegrate(q, w, dt):
    """
    Exponential (norm-exact) integrator of the body-rate kinematics over dt
    with omega held constant:  q_{k+1} = q_k ⊗ exp(1/2 [0, w dt]).
    Falls back to first order for tiny angles. Keeps ||q|| = 1 to machine eps.
    """
    theta = np.linalg.norm(w) * dt
    if theta < 1e-9:
        dq = np.array([1.0, 0.5*w[0]*dt, 0.5*w[1]*dt, 0.5*w[2]*dt])
        return qnorm(qmul(q, dq))
    axis = w / np.linalg.norm(w)
    dq = np.array([np.cos(theta/2),
                   *(axis * np.sin(theta/2))])
    return qnorm(qmul(q, dq))


# --------------------------------------------------------------------------- #
#  Euler <-> quaternion  (ZYX / aerospace 3-2-1, phi-theta-psi)
# --------------------------------------------------------------------------- #
def euler_to_quat(phi, theta, psi):
    cy, sy = np.cos(psi/2), np.sin(psi/2)
    cp, sp = np.cos(theta/2), np.sin(theta/2)
    cr, sr = np.cos(phi/2), np.sin(phi/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])


def quat_to_euler(q):
    qw, qx, qy, qz = q
    phi = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    s = 2*(qw*qy - qz*qx)
    s = np.clip(s, -1.0, 1.0)
    theta = np.arcsin(s)
    psi = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    return np.array([phi, theta, psi])
