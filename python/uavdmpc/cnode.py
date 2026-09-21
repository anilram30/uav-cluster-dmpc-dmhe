"""
cnode.py -- ctypes bridge from the Python research layer to a compiled
per-agent solver node library.

WHAT THIS IS
------------
The distributed layers of this framework never call an optimisation solver
directly: they call the small C interface in ``c_api/uav_solver.h``.  A *node
library* is any shared object that implements a per-agent controller and
estimator behind that interface and exposes the thin C entry points below
(``nmpc_*`` / ``nmhe_*``), which hold one solver instance per agent and report
their own execution times.

This module loads such a library, if one is present, and lets the closed-loop
simulations run the compiled controller and estimator in the loop instead of
their Python reference counterparts (``impl='c_mpc'``, ``'c_mhe'``,
``'c_both'``).  When no library is found, everything runs in Python
(``impl='py'``), which is the default and needs nothing but NumPy.

NOT INCLUDED IN THIS REPOSITORY
-------------------------------
The real-time-iteration NMPC and MHE solvers used for the in-the-loop and
timing results reported here are the author's separate, unpublished research
work (an extension of an MSc thesis on SLQP-MPC), and their sources are not
part of this repository.  Nothing in this framework depends on them: the
reference implementations in ``c_api/uav_solver_ref.c`` and in ``uavdmpc``
satisfy the same interface, and any solver that does can be dropped in here.

Expected symbols (all arrays are caller-owned, row-major, double):

    void nmpc_reset_all(void);
    void nmpc_step (int agent, const double *xhat, const double *ref,
                    double *u0, double *t_us);
    void nmhe_reset_all(void);
    void nmhe_init (int agent, const double *p0);
    void nmhe_prepare (int agent, const double *a_prev, double *t_us);
    void nmhe_pred_pos(int agent, double *p_pred);
    void nmhe_update  (int agent, const double *y, double *xhat, double *t_us);

Set ``UAV_SOLVER_NODE_LIB`` to the library path, or place it next to the
``uavdmpc`` package as ``libuavnode.so``.
"""
import ctypes
import os
import numpy as np

_LIB = None
_LIB_NAME = 'libuavnode.so'


def _find_lib():
    here = os.path.dirname(__file__)
    cands = [
        os.environ.get('UAV_SOLVER_NODE_LIB', ''),
        os.path.join(here, '..', _LIB_NAME),
        os.path.join(here, '..', '..', _LIB_NAME),
    ]
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def available():
    """True if a node library can be loaded (else run with impl='py')."""
    return _find_lib() is not None


def lib():
    global _LIB
    if _LIB is None:
        path = _find_lib()
        if path is None:
            raise RuntimeError(
                "no solver node library found (%s). Build one against "
                "c_api/uav_solver.h and point UAV_SOLVER_NODE_LIB at it, or "
                "run with impl='py'." % _LIB_NAME)
        L = ctypes.CDLL(path)
        d = ctypes.POINTER(ctypes.c_double)
        L.nmpc_reset_all.argtypes = []
        L.nmpc_step.argtypes = [ctypes.c_int, d, d, d, d]
        L.nmhe_reset_all.argtypes = []
        L.nmhe_init.argtypes = [ctypes.c_int, d]
        L.nmhe_prepare.argtypes = [ctypes.c_int, d, d]
        L.nmhe_pred_pos.argtypes = [ctypes.c_int, d]
        L.nmhe_update.argtypes = [ctypes.c_int, d, d, d]
        _LIB = L
    return _LIB


def _p(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class CNMPC:
    """Compiled MPC nodes (one solver instance per agent) behind the C API."""

    def __init__(self, n_agents):
        self.L = lib()
        self.L.nmpc_reset_all()
        self.n = n_agents
        self.t_us = np.zeros(2)
        self.times = []          # (prepare_us, feedback_us) per call

    def command(self, i, xhat6, slot3):
        """Return the input agent *i* applies, given its estimate and slot."""
        u0 = np.zeros(3)
        xh = np.ascontiguousarray(xhat6, dtype=float)
        sl = np.ascontiguousarray(slot3, dtype=float)
        self.L.nmpc_step(i, _p(xh), _p(sl), _p(u0), _p(self.t_us))
        self.times.append(tuple(self.t_us))
        return u0


class CMHE:
    """Compiled MHE nodes (one solver instance per agent) behind the C API."""

    def __init__(self, n_agents):
        self.L = lib()
        self.L.nmhe_reset_all()
        self.n = n_agents
        self.t1 = np.zeros(1)
        self.t2 = np.zeros(2)
        self.times = []          # (prepare_us, latency_us, feedback_us)

    def init(self, i, p0):
        p = np.ascontiguousarray(p0, dtype=float)
        self.L.nmhe_init(i, _p(p))

    def prepare(self, i, a_prev):
        """Everything that does not need the arriving measurement."""
        a = np.ascontiguousarray(a_prev, dtype=float)
        self.L.nmhe_prepare(i, _p(a), _p(self.t1))

    def pred_pos(self, i):
        """Predicted position before the update (used for relative fusion)."""
        p = np.zeros(3)
        self.L.nmhe_pred_pos(i, _p(p))
        return p

    def update(self, i, y3):
        """Fold in the measurement and return the fresh state estimate."""
        xh = np.zeros(6)
        y = np.ascontiguousarray(y3, dtype=float)
        self.L.nmhe_update(i, _p(y), _p(xh), _p(self.t2))
        self.times.append((float(self.t1[0]), float(self.t2[0]),
                           float(self.t2[1])))
        return xh
