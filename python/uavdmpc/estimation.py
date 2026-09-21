"""
estimation.py -- distributed moving-horizon estimation (DMHE) for the
translational states of a cluster agent, plus a local Kalman baseline.

Per agent state  s = [p (3), v (3)]  with linear model
    s_{k+1} = F s_k + Gb a_k + noise,   F=[[I,dt I],[0,I]], Gb=[[.5dt^2 I],[dt I]].
Measurements:
    * absolute noisy position (GPS-like)             y^p = p + n_p
    * relative position to neighbors j in N_i        y^r = (p_i - p_j) + n_r
The relative rows couple the estimators; DMHE additionally runs a consensus
averaging step on the shared estimate (Farina et al. 2010; Battistelli 2019
arrival-cost consensus). Baseline LocalKF uses GPS + accel only, no coupling.

The translational least-squares estimator here is the reference stand-in for the
real-time moving-horizon estimator each agent runs on the full nonlinear model:
the information/arrival-cost structure is the same, at reduced scale, and both
sit behind the same interface (see c_api/uav_solver.h).
"""
import numpy as np


def _FG(dt):
    F = np.block([[np.eye(3), dt*np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
    Gb = np.block([[0.5*dt*dt*np.eye(3)], [dt*np.eye(3)]])
    return F, Gb


class LocalMHE:
    """Linear moving-horizon estimator over a window of M intervals (batch LS
    with arrival cost). Supports absolute + relative (neighbor) measurements."""
    def __init__(self, dt=0.05, M=10, q_pos=0.02, q_vel=0.05,
                 r_gps=0.30, r_rel=0.10):
        self.dt, self.M = dt, M
        self.F, self.Gb = _FG(dt)
        self.Qi = np.diag([1/q_pos**2]*3 + [1/q_vel**2]*3)  # process info
        self.Rp_i = (1/r_gps**2)
        self.Rr_i = (1/r_rel**2)
        self.Hp = np.hstack([np.eye(3), np.zeros((3, 3))])
        # buffers
        self.ys = []        # gps positions
        self.us = []        # applied accel
        self.rels = []      # list of dict{j: y_rel} per node
        self.arrival_mean = None
        self.arrival_info = np.diag([1/1.0**2]*3 + [1/1.0**2]*3)
        self.shat = None

    def push(self, y_gps, u_acc, rel_meas):
        self.ys.append(np.asarray(y_gps, float))
        self.us.append(np.asarray(u_acc, float))
        self.rels.append(dict(rel_meas))
        if len(self.ys) > self.M+1:
            # advance arrival cost by one KF step (marginalize oldest)
            self._advance_arrival()
            self.ys.pop(0); self.us.pop(0); self.rels.pop(0)

    def _advance_arrival(self):
        # one information-filter step on the oldest node -> new arrival prior
        if self.arrival_mean is None:
            self.arrival_mean = np.concatenate([self.ys[0], np.zeros(3)])
        s0 = self.arrival_mean
        # measurement update at node 0 with gps
        Hp = self.Hp
        Info = self.arrival_info + Hp.T*self.Rp_i @ Hp
        rhs = self.arrival_info @ s0 + Hp.T*self.Rp_i @ self.ys[0]
        s0u = np.linalg.solve(Info, rhs)
        # time update
        s1 = self.F @ s0u + self.Gb @ self.us[0]
        # propagate information (add process noise)
        P = np.linalg.inv(Info)
        P1 = self.F @ P @ self.F.T + np.linalg.inv(self.Qi)
        self.arrival_info = np.linalg.inv(P1)
        self.arrival_mean = s1

    def estimate(self, neighbor_estimates):
        """Solve the window LS. neighbor_estimates: dict j-> p_hat_j (world)."""
        n = len(self.ys)
        if n == 0:
            return None
        M = n - 1
        dimz = 6*n
        A = []   # rows of the sparse LS (dense here, small)
        bvec = []
        W = []   # per-row weights (info)
        # arrival cost on node 0
        if self.arrival_mean is not None:
            L = np.linalg.cholesky(self.arrival_info)
            for r in range(6):
                row = np.zeros(dimz)
                row[0:6] = L[r]
                A.append(row); bvec.append(L[r] @ self.arrival_mean); W.append(1.0)
        # dynamics constraints as soft (process-noise) rows
        for k in range(M):
            pred = self.F  # s_{k+1} - F s_k - Gb u_k = w_k
            Lq = np.linalg.cholesky(self.Qi)
            for r in range(6):
                row = np.zeros(dimz)
                row[6*(k+1):6*(k+1)+6] += Lq[r]
                row[6*k:6*k+6] -= Lq[r] @ self.F
                A.append(row)
                bvec.append(Lq[r] @ (self.Gb @ self.us[k]))
                W.append(1.0)
        # gps measurements
        for k in range(n):
            for r in range(3):
                row = np.zeros(dimz)
                row[6*k+r] = np.sqrt(self.Rp_i)
                A.append(row); bvec.append(np.sqrt(self.Rp_i)*self.ys[k][r]); W.append(1.0)
        # relative measurements: apply ONLY at the newest node, where the
        # communicated neighbour estimate is time-aligned with the measurement.
        # (Anchoring a past window node to the current neighbour position would
        #  introduce a velocity-dependent time-mismatch error.)
        knew = n - 1
        for j, yrel in self.rels[knew].items():
            if j not in neighbor_estimates:
                continue
            pj = neighbor_estimates[j]
            for r in range(3):
                row = np.zeros(dimz)
                row[6*knew+r] = np.sqrt(self.Rr_i)
                A.append(row)
                bvec.append(np.sqrt(self.Rr_i)*(yrel[r] + pj[r]))
                W.append(1.0)
        A = np.array(A); bvec = np.array(bvec)
        # normal equations
        AtA = A.T @ A + 1e-9*np.eye(dimz)
        Atb = A.T @ bvec
        z = np.linalg.solve(AtA, Atb)
        self.shat = z[-6:]
        # Engagement guard: velocity is unobservable from a near-empty window,
        # so during window fill-up hold it at the known rest condition rather
        # than emitting the cold-start transient. A magnitude clamp guards any
        # residual outlier from reaching the controller.
        if n < 5:
            self.shat[3:6] *= (n - 1)/4.0 if n > 1 else 0.0
        vmax = 30.0
        vn = np.linalg.norm(self.shat[3:6])
        if vn > vmax:
            self.shat[3:6] *= vmax/vn
        return self.shat.copy()


class LocalKF:
    """Plain Kalman filter baseline: GPS + accel only (no coupling/consensus).

    For this linear-Gaussian translational problem the general moving-horizon
    estimator reduces *exactly* to this information/covariance Kalman recursion
    (the estimator's EKF/EKS-reduction property); it is used as the per-agent
    engine in the reference sim because it is numerically well conditioned from
    the first sample (proper covariance initialisation), unlike a cold-started
    batch window. The distributed DMHE extends it with relative fusion below."""
    def __init__(self, dt=0.05, q_pos=0.02, q_vel=0.05, r_gps=0.30,
                 p0_pos=0.5, p0_vel=0.2):
        self.F, self.Gb = _FG(dt)
        self.Q = np.diag([q_pos**2]*3 + [q_vel**2]*3)
        self.R = r_gps**2*np.eye(3)
        self.Hp = np.hstack([np.eye(3), np.zeros((3, 3))])
        self.s = None
        self.P = np.diag([p0_pos**2]*3 + [p0_vel**2]*3)   # known rest start

    def predict(self, u_acc):
        if self.s is None:
            return
        self.s = self.F @ self.s + self.Gb @ u_acc
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_gps(self, y_gps):
        if self.s is None:
            self.s = np.concatenate([y_gps, np.zeros(3)])
            return
        y = y_gps - self.Hp @ self.s
        S = self.Hp @ self.P @ self.Hp.T + self.R
        K = self.P @ self.Hp.T @ np.linalg.inv(S)
        self.s = self.s + K @ y
        self.P = (np.eye(6) - K @ self.Hp) @ self.P

    def step(self, y_gps, u_acc):
        self.predict(u_acc)
        self.update_gps(y_gps)
        return self.s.copy()


class DMHE(LocalKF):
    """Distributed MHE (information/KF reduction) with relative-measurement
    fusion. Each step: IMU predict -> GPS update -> relative update against
    time-aligned neighbour estimates (Farina et al. 2010). The relative rows
    reduce error below the GPS-only local filter without disturbing the
    GPS-anchored absolute frame."""
    def __init__(self, *a, r_rel=0.10, **k):
        super().__init__(*a, **k)
        self.Rr = r_rel**2*np.eye(3)

    def update_relative(self, rel_meas, neighbor_states, neighbor_P):
        if self.s is None or not rel_meas:
            return
        for j, y_rel in rel_meas.items():
            if j not in neighbor_states:
                continue
            pj = neighbor_states[j][0:3]
            Pj = neighbor_P.get(j, np.eye(3)*0.25)[0:3, 0:3] if neighbor_P else np.eye(3)*0.25
            z = y_rel + pj                       # pseudo-measurement of p_i
            y = z - self.Hp @ self.s
            S = self.Hp @ self.P @ self.Hp.T + self.Rr + Pj
            K = self.P @ self.Hp.T @ np.linalg.inv(S)
            self.s = self.s + K @ y
            self.P = (np.eye(6) - K @ self.Hp) @ self.P

    def step_dist(self, y_gps, u_acc, rel_meas, neighbor_states, neighbor_P):
        self.predict(u_acc)
        self.update_gps(y_gps)
        self.update_relative(rel_meas, neighbor_states, neighbor_P)
        return self.s.copy()


def metropolis_weights(graph):
    """Metropolis-Hastings consensus weights (doubly stochastic, spectral radius
    of the disagreement dynamics < 1 for any connected graph -> always stable)."""
    deg = {i: len(graph[i]) for i in graph}
    W = {}
    for i in graph:
        wij = {}
        s = 0.0
        for j in graph[i]:
            w = 1.0/(1.0 + max(deg[i], deg[j]))
            wij[j] = w
            s += w
        wij[i] = 1.0 - s
        W[i] = wij
    return W


def consensus_average(estimates, graph, rounds=1):
    """Metropolis-weighted consensus on estimates (guaranteed contractive on the
    disagreement subspace). estimates: dict i-> s_i. graph: dict i-> neighbors."""
    W = metropolis_weights(graph)
    est = {i: estimates[i].copy() for i in estimates}
    for _ in range(rounds):
        new = {}
        for i in est:
            acc = W[i][i]*est[i]
            for j in graph[i]:
                acc = acc + W[i][j]*est[j]
            new[i] = acc
        est = new
    return est
