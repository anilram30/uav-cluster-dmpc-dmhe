"""
sim_single_c.py -- single-hexacopter C-in-the-loop validation:

    nonlinear quaternion plant -> C MHE -> C NMPC -> (flatness + attitude) -> plant

versus the identical loop with the Python reference estimator/controller, on the
SAME noise stream. Mission: take-off + step to (3,2,6), then a moving target.
Prints trajectory/estimation parity and in-loop solver timings.
"""
import numpy as np
from uavdmpc.hexacopter import Hexacopter
from uavdmpc.control import LinearMPC, flatness, geometric_attitude
from uavdmpc.estimation import LocalKF
from uavdmpc import cnode


def target(t):
    if t < 6:
        return np.array([3.0, 2.0, min(1.2*t, 6.0)])
    return np.array([3.0 + 1.5*np.sin(0.5*(t-6)), 2.0 - 1.0*np.sin(0.35*(t-6)), 6.0])


def run(impl, seed=5, T=20.0, dt=0.05, inner=15):
    rng = np.random.default_rng(seed)
    hexa = Hexacopter(); m = hexa.p['m']
    nsteps = int(T/dt)
    x = np.zeros(13); x[6] = 1.0
    a_meas = np.zeros(3)
    if impl == 'py':
        est = LocalKF(dt=dt, q_pos=0.03, q_vel=0.10, r_gps=0.30, p0_pos=0.3, p0_vel=0.05)
        mpc = LinearMPC(dt=dt, N=25, Qp=3.0, Qv=3.5, Ru=1.2, a_max=4.0)
    else:
        cm = cnode.CMHE(1); cc = cnode.CNMPC(1)
        inited = False
    P, E, V = [], [], []
    for k in range(nsteps):
        t = k*dt
        gps = x[0:3] + rng.standard_normal(3)*0.30
        if impl == 'py':
            s = est.step(gps, a_meas)
            phat, vhat = s[0:3], s[3:6]
            a = mpc.command(phat, vhat, np.tile(target(t), (mpc.N, 1)))
        else:
            if not inited:
                cm.init(0, gps); inited = True
            cm.prepare(0, a_meas)
            s = cm.update(0, gps)
            phat, vhat = s[0:3], s[3:6]
            a = cc.command(0, s, target(t))
        Tm, qd, Rd = flatness(a, m)
        vb = x[3:6].copy()
        for _ in range(inner):
            tau = geometric_attitude(x[6:10], x[10:13], Rd, hexa.J)
            Om = hexa.wrench_to_rotor_speeds(np.array([Tm, *tau]))
            x = hexa.step(x, hexa.rotor_speeds_to_wrench(Om), dt/inner)
        a_meas = (x[3:6]-vb)/dt + rng.standard_normal(3)*0.08
        P.append(x[0:3].copy())
        E.append(np.linalg.norm(phat - x[0:3]))
        V.append(np.linalg.norm(x[0:3] - target(t)))
    out = dict(P=np.array(P), est_err=np.array(E), trk_err=np.array(V))
    if impl != 'py':
        tm = np.array(cm.times); tc = np.array(cc.times)
        out['mhe_us'] = np.median(tm, 0)
        out['mpc_us'] = np.median(tc, 0)
    return out


if __name__ == '__main__':
    py = run('py')
    cc = run('c')
    dtraj = np.linalg.norm(py['P']-cc['P'], axis=1)
    print("single-hexacopter C-in-the-loop vs Python reference (same noise):")
    print(f"  tracking err  py mean {py['trk_err'][40:].mean():.3f} m | "
          f"C {cc['trk_err'][40:].mean():.3f} m")
    print(f"  estim.  err   py mean {py['est_err'][40:].mean():.3f} m | "
          f"C {cc['est_err'][40:].mean():.3f} m")
    print(f"  trajectory difference: mean {dtraj.mean():.3f} m, max {dtraj.max():.3f} m")
    print(f"  in-loop C MHE  med (prep, latency, fb) us = {np.round(cc['mhe_us'],2)}")
    print(f"  in-loop C NMPC med (prep, fb) us          = {np.round(cc['mpc_us'],2)}")
