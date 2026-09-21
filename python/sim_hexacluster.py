"""
sim_hexacluster.py -- closed-loop distributed simulation of a hexacopter
cluster on the full nonlinear 13-state quaternion plants.

Parameterised for Monte-Carlo robustness studies:
  * seed             : noise realisation
  * denied / window  : which agents lose GNSS, and when
  * topology         : 'chords' (ring + 2-hop, deg 4) or 'ring' (deg 2)
  * edge_drop        : (i, j, t0, t1) temporarily removes one comm edge
  * ctrl_est         : 'dmhe' (default) or 'ekf' -- which estimate feeds control
  * use_reconfig     : contract-and-rotate reconfiguration after t=20 s

Per agent, every outer step (dt):
  1. DMHE: IMU predict + GNSS update if available; GNSS-denied agents fuse
     relative measurements to GNSS-good neighbours (two-pass, time-aligned).
     A local dead-reckoning KF runs in parallel as the baseline.
  2. Metropolis consensus agrees on the formation-centre reference.
  3. LinearMPC -> commanded acceleration toward the agent's formation slot.
  4. CBF safety filter minimally alters the accel to keep pairwise separation.
  5. Differential flatness (tilt-limited) -> thrust + desired attitude;
     inner geometric attitude loop drives the full nonlinear plant.

run() returns a metrics dict; save=True additionally writes results_hexa.npz
for the report figures.
"""
import os
import time
import numpy as np
from uavdmpc.hexacopter import Hexacopter
from uavdmpc.control import LinearMPC, cbf_safety_filter, flatness, geometric_attitude
from uavdmpc.estimation import DMHE, LocalKF
from uavdmpc.coordination import Cluster

D_SAFE = 1.4


def formation_offsets(Na, radius=3.0):
    ang = np.linspace(0, 2*np.pi, Na, endpoint=False)
    return np.column_stack([radius*np.cos(ang), radius*np.sin(ang), np.zeros(Na)])


def center_trajectory(t):
    """Formation-center reference over the mission (world, z-up)."""
    z = min(1.0*t, 6.0)                      # climb to 6 m
    if t < 8:
        return np.array([0.0, 0.0, z])
    tt = t - 8
    x = 6.0*np.sin(0.08*np.pi*tt)            # sweep in x
    y = 3.0*np.sin(0.16*np.pi*tt)
    return np.array([x, y, 6.0])


def run(T=30.0, dt=0.05, inner=15, Na=6, use_reconfig=True, seed=7,
        denied=(3, 4, 5), deny_window=(12.0, 22.0), topology='chords',
        edge_drop=None, ctrl_est='dmhe', impl='py', multihop=False,
        save=False, verbose=False):
    """impl: 'py' reference layer | 'c_mhe' compiled C estimator in the loop |
    'c_mpc' compiled C controller | 'c_both' full compiled estimation+control
    path (see uavdmpc/cnode.py; needs a solver node library). multihop: leveled chained anchoring (DAG over anchor levels) so
    denied agents without a GNSS-good neighbour anchor through already-anchored
    ones."""
    rng = np.random.default_rng(seed)
    denied = frozenset(denied)
    T0d, T1d = deny_window
    use_c_mhe = impl in ('c_mhe', 'c_both')
    use_c_mpc = impl in ('c_mpc', 'c_both')
    if use_c_mhe or use_c_mpc:
        from uavdmpc import cnode
        cm = cnode.CMHE(Na) if use_c_mhe else None
        cc = cnode.CNMPC(Na) if use_c_mpc else None

    def gps_available(i, t):
        return not (i in denied and T0d <= t < T1d)

    hexas = [Hexacopter() for _ in range(Na)]
    m = hexas[0].p['m']
    off0 = formation_offsets(Na, 3.0)
    edges = [(i, (i+1) % Na) for i in range(Na)]
    if topology == 'chords':
        edges += [(i, (i+2) % Na) for i in range(Na)]
    cluster = Cluster(off0, edges)
    cluster_drop = None
    if edge_drop is not None:
        di, dj, dt0, dt1 = edge_drop
        e2 = [e for e in edges if set(e) != {di, dj}]
        cluster_drop = Cluster(off0, e2)

    def graph_at(t):
        if cluster_drop is not None and edge_drop[2] <= t < edge_drop[3]:
            return cluster_drop
        return cluster

    lam2 = cluster.algebraic_connectivity()
    mpc = LinearMPC(dt=dt, N=25, Qp=3.0, Qv=3.5, Ru=1.2, a_max=4.0)
    mhes = [DMHE(dt=dt, q_pos=0.03, q_vel=0.10, r_gps=0.30, r_rel=0.12,
                 p0_pos=0.3, p0_vel=0.05) for _ in range(Na)]
    kfs = [LocalKF(dt=dt, q_pos=0.03, q_vel=0.10, r_gps=0.30,
                   p0_pos=0.3, p0_vel=0.05) for _ in range(Na)]
    a_meas = np.zeros((Na, 3))

    X = []
    for i in range(Na):
        x = np.zeros(13); x[6] = 1.0
        x[0:2] = off0[i, 0:2]*0.7 + rng.standard_normal(2)*0.15
        X.append(x)
    X = np.array(X)

    nsteps = int(T/dt)
    log = dict(t=[], Xtrue=[], p_hat_mhe=[], p_hat_kf=[], form_err=[],
               min_dist=[], effort=[], step_ms=[], center=[], rotor=[])
    c_local = np.tile(center_trajectory(0.0), (Na, 1))
    qp_fail = 0
    diverged = False

    for k in range(nsteps):
        t = k*dt
        G = graph_at(t)
        center_ref = center_trajectory(t)
        scale, rot = 1.0, 0.0
        if use_reconfig and t > 20:
            s = (t-20)/6.0
            scale = 1.0 - 0.40*min(s, 1.0)
            rot = 0.6*min(s, 1.0)
        Rz = np.array([[np.cos(rot), -np.sin(rot), 0],
                       [np.sin(rot), np.cos(rot), 0], [0, 0, 1.0]])
        offs = (off0*scale) @ Rz.T

        # ---- distributed estimation ----
        p_hat_mhe = np.zeros((Na, 3)); v_hat_mhe = np.zeros((Na, 3))
        p_hat_kf = np.zeros((Na, 3)); v_hat_kf = np.zeros((Na, 3))
        gps = X[:, 0:3] + rng.standard_normal((Na, 3))*0.30
        have = [gps_available(i, t) for i in range(Na)]

        # anchor levels (topological, implementation-independent): level 0 =
        # GNSS-good; level l = denied agent with an anchored neighbour of
        # strictly lower level (DAG -> no mutual reinforcement). Without
        # multihop only level 1 exists (the original one-hop policy).
        level = {i: 0 for i in range(Na) if have[i]}
        anchors_of = {}
        Lmax = 3 if multihop else 1
        for lv in range(1, Lmax+1):
            newly = []
            for i in range(Na):
                if have[i] or i in level:
                    continue
                js = [j for j in G.graph[i] if j in level and level[j] < lv]
                if js:
                    anchors_of[i] = js; newly.append((i, lv))
            for i, lv_ in newly:
                level[i] = lv_
        # draw relative-measurement noise in a fixed, impl-independent order
        relmeas = {}
        for lv in range(1, Lmax+1):
            for i in range(Na):
                if level.get(i) != lv or have[i]:
                    continue
                for j in anchors_of[i]:
                    relmeas[(i, j)] = (X[i, 0:3] - X[j, 0:3]) \
                        + rng.standard_normal(3)*0.10

        t0 = time.perf_counter()
        if not use_c_mhe:
            # ---- Python reference estimator ----
            for i in range(Na):
                mhes[i].predict(a_meas[i])
                if have[i]:
                    mhes[i].update_gps(gps[i])
            for lv in range(1, Lmax+1):        # fuse level by level (chaining)
                for i in range(Na):
                    if level.get(i) != lv or have[i]:
                        continue
                    rel, ns, nP = {}, {}, {}
                    for j in anchors_of[i]:
                        rel[j] = relmeas[(i, j)]
                        ns[j] = mhes[j].s.copy(); nP[j] = mhes[j].P.copy()
                    mhes[i].update_relative(rel, ns, nP)
            for i in range(Na):
                p_hat_mhe[i] = mhes[i].s[0:3]; v_hat_mhe[i] = mhes[i].s[3:6]
        else:
            # ---- compiled C estimator in the loop ----
            if k == 0:
                for i in range(Na):
                    cm.init(i, gps[i])
            xh = np.zeros((Na, 6)); done = np.zeros(Na, dtype=bool)
            for i in range(Na):
                cm.prepare(i, a_meas[i])
            for i in range(Na):                # level 0: GNSS measurement
                if have[i]:
                    xh[i] = cm.update(i, gps[i]); done[i] = True
            for lv in range(1, Lmax+1):        # anchored: pseudo-measurement
                for i in range(Na):
                    if level.get(i) != lv or have[i]:
                        continue
                    zs = [relmeas[(i, j)] + xh[j][0:3] for j in anchors_of[i]]
                    xh[i] = cm.update(i, np.mean(zs, axis=0)); done[i] = True
            for i in range(Na):                # unanchored: coast (zero innov.)
                if not done[i]:
                    xh[i] = cm.update(i, cm.pred_pos(i))
            p_hat_mhe = xh[:, 0:3].copy(); v_hat_mhe = xh[:, 3:6].copy()
        for i in range(Na):
            kfs[i].predict(a_meas[i])
            if have[i]:
                kfs[i].update_gps(gps[i])
            p_hat_kf[i] = kfs[i].s[0:3]; v_hat_kf[i] = kfs[i].s[3:6]
        step_ms = (time.perf_counter()-t0)*1e3

        # which estimate drives control (baseline study: 'ekf')
        if ctrl_est == 'ekf':
            p_c, v_c = p_hat_kf, v_hat_kf
        else:
            p_c, v_c = p_hat_mhe, v_hat_mhe

        # ---- consensus on the centre ----
        for i in range(Na):
            c_local[i] = center_ref + rng.standard_normal(3)*0.05
        c_agree = G.consensus_center(c_local, rounds=3)

        # ---- per-agent control ----
        a_nom = np.zeros((Na, 3))
        for i in range(Na):
            p_slot = c_agree[i] + offs[i]
            if use_c_mpc:
                a_nom[i] = cc.command(i, np.concatenate([p_c[i], v_c[i]]), p_slot)
            else:
                a_nom[i] = mpc.command(p_c[i], v_c[i], np.tile(p_slot, (mpc.N, 1)))
        a_safe = np.zeros((Na, 3))
        for i in range(Na):
            a_safe[i] = cbf_safety_filter(i, p_c, v_c, a_nom[i],
                                          G.graph[i], D_safe=D_SAFE,
                                          k1=3.5, k2=4.0, a_max=6.0)
            if not np.all(np.isfinite(a_safe[i])):
                qp_fail += 1
                a_safe[i] = np.clip(a_nom[i], -6, 6)

        # ---- flatness + inner attitude loop on the nonlinear plant ----
        rotor_rms = np.zeros(Na); effort = 0.0
        for i in range(Na):
            Tmag, q_des, R_des = flatness(a_safe[i], m, yaw_des=0.0)
            dti = dt/inner
            v_before = X[i, 3:6].copy()
            for _ in range(inner):
                tau = geometric_attitude(X[i, 6:10], X[i, 10:13], R_des, hexas[i].J)
                Om = hexas[i].wrench_to_rotor_speeds(np.array([Tmag, *tau]))
                X[i] = hexas[i].step(X[i], hexas[i].rotor_speeds_to_wrench(Om), dti)
            a_meas[i] = (X[i, 3:6] - v_before)/dt + rng.standard_normal(3)*0.08
            rotor_rms[i] = np.sqrt(np.mean(Om**2)); effort += Tmag
        if np.max(np.abs(X[:, 0:3])) > 60 or not np.all(np.isfinite(X)):
            diverged = True

        # ---- logging ----
        form_err = np.mean([np.linalg.norm(X[i, 0:3]-(center_ref+offs[i]))
                            for i in range(Na)])
        dmin = np.min([np.linalg.norm(X[i, 0:3]-X[j, 0:3])
                       for i in range(Na) for j in range(i+1, Na)])
        log['t'].append(t); log['Xtrue'].append(X.copy())
        log['p_hat_mhe'].append(p_hat_mhe.copy()); log['p_hat_kf'].append(p_hat_kf.copy())
        log['form_err'].append(form_err); log['min_dist'].append(dmin)
        log['effort'].append(effort/Na); log['step_ms'].append(step_ms)
        log['center'].append(center_ref.copy()); log['rotor'].append(rotor_rms.copy())
        if verbose and os.environ.get('DBG') and k % 40 == 0:
            print(f"  k={k:3d} t={t:5.1f} max|p|={np.max(np.abs(X[:,0:3])):8.2f} "
                  f"form_err={form_err:7.2f} dmin={dmin:5.2f}")

    out = {kk: np.array(v) for kk, v in log.items()}
    tt = out['t']
    den = sorted(denied)
    win = (tt >= T0d) & (tt < T1d)
    e_mhe = out['p_hat_mhe'] - out['Xtrue'][:, :, 0:3]
    e_kf = out['p_hat_kf'] - out['Xtrue'][:, :, 0:3]

    def rmse(e, mask, agents):
        if mask.sum() == 0 or len(agents) == 0:
            return float('nan')
        return float(np.sqrt(np.mean(e[mask][:, agents, :]**2)))

    hold = (tt >= 8.0) & (tt <= 20.0)          # assembled, pre-reconfig
    # shape error: deviation from the formation geometry about the achieved
    # centroid (isolates formation-keeping from uniform centroid tracking lag)
    P = out['Xtrue'][:, :, 0:3]
    centroid = P.mean(axis=1, keepdims=True)
    shape_err_t = np.mean(np.linalg.norm(P - centroid - off0[None, :, :], axis=2), axis=1)
    metrics = dict(
        seed=seed, Na=Na, lam2=float(lam2), topology=topology,
        denied=den, ctrl_est=ctrl_est, impl=impl, multihop=bool(multihop),
        rmse_mhe_denied=rmse(e_mhe, win, den),
        rmse_kf_denied=rmse(e_kf, win, den),
        rmse_mhe_all=float(np.sqrt(np.mean(e_mhe**2))),
        rmse_kf_all=float(np.sqrt(np.mean(e_kf**2))),
        form_err_hold=float(np.mean(out['form_err'][hold])),
        shape_err_hold=float(np.mean(shape_err_t[hold])),
        form_err_final=float(out['form_err'][-1]),
        form_err_max_blackout=float(np.max(out['form_err'][win])) if win.sum() else float('nan'),
        min_sep=float(out['min_dist'].min()),
        sep_violations=int(np.sum(out['min_dist'] < D_SAFE)),
        step_ms_med=float(np.median(out['step_ms'])),
        step_ms_p95=float(np.percentile(out['step_ms'], 95)),
        step_ms_max=float(out['step_ms'].max()),
        qp_failures=int(qp_fail),
        diverged=bool(diverged),
    )
    if use_c_mhe and cm.times:
        tm = np.array(cm.times)
        metrics['c_mhe_us'] = [float(v) for v in np.median(tm, 0)]
    if use_c_mpc and cc.times:
        tc = np.array(cc.times)
        metrics['c_mpc_us'] = [float(v) for v in np.median(tc, 0)]
    if save:
        out['lam2'] = lam2; out['Na'] = Na
        out['gps_denied'] = np.array(den); out['deny_window'] = np.array([T0d, T1d])
        np.savez('results_hexa.npz', **out)
    if verbose:
        print(f"[hexa] seed={seed} denied={den} topo={topology} ctrl={ctrl_est}: "
              f"DMHE_den={metrics['rmse_mhe_denied']:.3f} KF_den={metrics['rmse_kf_denied']:.3f} "
              f"minsep={metrics['min_sep']:.2f} viol={metrics['sep_violations']} "
              f"ferr_hold={metrics['form_err_hold']:.2f} diverged={diverged}")
    return metrics


if __name__ == '__main__':
    mtr = run(save=True, verbose=True)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in mtr.items()})
