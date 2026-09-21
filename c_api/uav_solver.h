/* ==========================================================================
 * uav_solver.h -- clean C application programming interface to the per-agent
 * real-time solvers used by the distributed UAV cluster framework.
 *
 * This is the interface described in the report, Ch. 5 ("Per-Agent Real-Time
 * Estimation and Control"). The coordination layer (C++) talks ONLY to this
 * interface; it never sees how the solvers are built. In a production build the
 * function bodies dispatch to a real-time-iteration MPC controller and MHE
 * estimator (separate work, not part of this repository); uav_solver_ref.c
 * provides a dependency-free reference implementation (finite-horizon LQR /
 * Kalman filter on the translational model) so the whole stack compiles, links
 * and runs on its own.
 *
 * All arrays are caller-owned, row-major, statically sized by the Dims structs.
 * No dynamic allocation on the hot path.
 * ========================================================================== */
#ifndef UAV_SOLVER_H
#define UAV_SOLVER_H

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------- controller */
typedef struct {
    int    nx;      /* state dimension                  */
    int    nu;      /* input dimension                  */
    int    nc;      /* stage constraint rows c(x,u)<=0  */
    int    ncN;     /* terminal constraint rows         */
    int    N;       /* prediction horizon               */
    double Ts;      /* sampling period [s]              */
} MPCDims;

typedef struct MPCProblem MPCProblem;

MPCProblem* mpc_create      (const MPCDims* d);
int         mpc_set_weights (MPCProblem* p, const double* Qdiag,
                             const double* Rdiag, const double* QNdiag);
/* inject the coordination-layer reference (formation slot) */
int         mpc_set_reference(MPCProblem* p, const double* xref, const double* uref);
/* inject/update coupling constraint bounds (e.g. collision-avoidance rows) */
int         mpc_set_constraint_bounds(MPCProblem* p, const double* c_ub);
/* RTI preparation phase: matrix pass, before the measurement arrives */
int         mpc_prepare     (MPCProblem* p);
/* RTI feedback phase: applied input u0 from the fresh estimate xhat */
int         mpc_feedback    (MPCProblem* p, const double* xhat, double* u0);
void        mpc_destroy     (MPCProblem* p);

/* --------------------------------------------------------------- estimator */
typedef struct {
    int    nx;      /* state dimension                    */
    int    nu;      /* known input dimension              */
    int    nw;      /* estimated disturbance dimension    */
    int    ny;      /* measurement dimension              */
    int    nc;      /* stage constraint rows              */
    int    ncM;     /* newest-node constraint rows        */
    int    M;       /* estimation window length           */
    double Ts;      /* sampling period [s]                */
} MHEDims;

typedef struct MHEProblem MHEProblem;

MHEProblem* mhe_create   (const MHEDims* d);
/* Vdiag: measurement info; Wdiag: disturbance info; La0/xa0: arrival prior */
int         mhe_init     (MHEProblem* e, const double* Vdiag, const double* Wdiag,
                          const double* La0diag, const double* xa0);
/* RTI preparation: everything not needing the new measurement y_t */
int         mhe_prepare  (MHEProblem* e, const double* u_prev, const double* u_t);
/* instantaneous output: estimate xhat from the arriving measurement y_t */
int         mhe_feedback (MHEProblem* e, const double* y_t, double* xhat);
/* fold in a relative pseudo-measurement z of the position block, with the
 * neighbour position covariance Pj (covariance-aware DMHE fusion, Ch. 6) */
int         mhe_fuse_relative(MHEProblem* e, const double* z, const double* Pj);
void        mhe_destroy  (MHEProblem* e);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* UAV_SOLVER_H */
