"""
qpsolve.py -- tiny dependency-free dense QP solver for the small problems that
arise in the coordination layer (CBF safety filter, local ADMM steps).

Solves    min 1/2 z^T H z + g^T z   s.t.   A z <= b
with a primal active-set method. H must be SPD. Dimensions are small (<= ~12),
so this is more than fast enough and keeps the reference sim dependency-free.
This mirrors, at toy scale, the role the compiled C solver core plays per agent.
"""
import numpy as np


def solve_qp(H, g, A=None, b=None, max_iter=100):
    n = H.shape[0]
    Hinv = np.linalg.inv(H)
    z = -Hinv @ g                      # unconstrained minimizer
    if A is None or len(A) == 0:
        return z, np.array([], dtype=int)
    A = np.atleast_2d(A); b = np.atleast_1d(b)
    m = A.shape[0]
    active = []
    for _ in range(max_iter):
        viol = A @ z - b
        if np.all(viol <= 1e-9):
            # check dual feasibility on active set
            if not active:
                return z, np.array(active, dtype=int)
        # add most violated inactive constraint
        cand = [i for i in range(m) if i not in active]
        if cand:
            vio = A[cand] @ z - b[cand]
            j = int(np.argmax(vio))
            if vio[j] > 1e-9:
                active.append(cand[j])
        # solve equality-constrained QP on the active set (KKT)
        if active:
            Aa = A[active]
            KKT = np.block([[H, Aa.T],
                            [Aa, np.zeros((len(active), len(active)))]])
            rhs = np.concatenate([-g, b[active]])
            try:
                sol = np.linalg.solve(KKT, rhs)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
            z = sol[:n]
            lam = sol[n:]
            # drop constraints with negative multiplier
            if np.all(lam >= -1e-9):
                if np.all(A @ z - b <= 1e-8):
                    return z, np.array(active, dtype=int)
            else:
                drop = int(np.argmin(lam))
                active.pop(drop)
        else:
            z = -Hinv @ g
    return z, np.array(active, dtype=int)
