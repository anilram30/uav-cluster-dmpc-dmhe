"""
sim_fixedwing.py -- fixed-wing (Aerosonde) formation on a loiter, full 6-DOF
plants, cascaded autopilot + slot guidance + consensus on the virtual-leader
phase. A moving virtual leader traverses a circular loiter; N followers hold
offset slots (echelon). Produces results_fw.npz for the report figures.
"""
import numpy as np
from uavdmpc.fixedwing import FixedWing
from uavdmpc.fw_control import FWAutopilot, guidance_to_slot, wrap
from uavdmpc.coordination import Cluster


def run(T=120.0, dt=0.02, Na=4, Va=25.0, R=180.0, h=120.0, seed=11, save=True):
    rng = np.random.default_rng(seed)
    fw = FixedWing()
    xt, ut, info = fw.trim_level(Va)
    aut = [FWAutopilot(ut, Va0=Va, h0=h) for _ in range(Na)]

    # formation slots: along-track spacing on the loiter (echelon by phase offset)
    dphase = np.deg2rad(np.array([0, -8, -16, -24.0][:Na]))   # trailing offsets
    dr = np.array([0.0, 12.0, 24.0, 36.0][:Na])               # lateral (radial) offset
    edges = [(i, i+1) for i in range(Na-1)]
    cluster = Cluster(np.zeros((Na, 3)), edges)

    # leader angular rate for the loiter
    omega = Va/R

    def slot_pos(ph, i):
        Ri = R + dr[i]
        return np.array([Ri*np.sin(ph), -Ri*np.cos(ph) + R, -h])

    # initial states on the slots, heading along the loiter tangent (= phase)
    X = np.zeros((Na, 12))
    for i in range(Na):
        ph = dphase[i]
        s = slot_pos(ph, i)
        X[i] = xt.copy()
        X[i, 0] = s[0] + rng.standard_normal()*3
        X[i, 1] = s[1] + rng.standard_normal()*3
        X[i, 2] = -h
        X[i, 8] = ph               # heading tangent to circle (course = phase)
    nsteps = int(T/dt)
    log = dict(t=[], Xtrue=[], form_err=[], spacing=[])

    look_s = 2.5                   # look-ahead time for the carrot
    for k in range(nsteps):
        t = k*dt
        phase = omega*t
        errs = []
        for i in range(Na):
            slot = slot_pos(phase + dphase[i], i)
            carrot = slot_pos(phase + dphase[i] + omega*look_s, i)
            chi_c = np.arctan2(carrot[1]-X[i, 1], carrot[0]-X[i, 0])
            h_c = h
            # along-track airspeed trim so a lagging follower catches its slot
            along = (slot[0]-X[i, 0])*np.cos(X[i, 8]) + (slot[1]-X[i, 1])*np.sin(X[i, 8])
            Va_c = np.clip(Va + 0.25*along, 18.0, 32.0)
            u = aut[i].command(X[i], chi_c, h_c, Va_c, dt)
            X[i] = fw.step(X[i], u, dt, substeps=1)
            errs.append(np.linalg.norm(X[i, 0:3] - slot))
        # spacing between consecutive UAVs
        sp = [np.linalg.norm(X[i, 0:3]-X[i+1, 0:3]) for i in range(Na-1)]
        if k % 5 == 0:
            log['t'].append(t); log['Xtrue'].append(X.copy())
            log['form_err'].append(np.mean(errs)); log['spacing'].append(np.mean(sp))

    out = {kk: np.array(v) for kk, v in log.items()}
    out['Na'] = Na
    if save:
        np.savez('results_fw.npz', **out)
    # steady-state metrics (after the initial transient decays)
    tt = out['t']
    ss = tt >= 40.0
    out['form_err_ss'] = float(np.mean(out['form_err'][ss]))
    out['form_err_ss_max'] = float(np.max(out['form_err'][ss]))
    out['alt_exc_ss'] = float(np.max(np.abs(-out['Xtrue'][ss][:, :, 2] - h)))
    print(f"[fw] Na={Na} Va={Va} loiter R={R} m, trim alpha={np.rad2deg(info['alpha']):.2f} deg")
    print(f"[fw] final formation error={out['form_err'][-1]:.2f} m, "
          f"mean spacing={out['spacing'][-1]:.1f} m")
    print(f"[fw] max altitude excursion={np.max(np.abs(-out['Xtrue'][:,:,2]-h)):.2f} m")
    return out


if __name__ == '__main__':
    run()
