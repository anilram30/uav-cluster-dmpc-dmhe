/* ==========================================================================
 * uav_solver_ref.c -- dependency-free REFERENCE implementation of uav_solver.h.
 *
 * The controller is a finite-/infinite-horizon LQR on the translational
 * double-integrator model (per axis); the estimator is an information/Kalman
 * filter with covariance-aware relative fusion. These are exactly the linear
 * reductions the report uses (MPC of Ch. 5; Kalman reduction of the MHE, Ch. 5-6)
 * and mirror the Python reference sim. In a production build these bodies are
 * replaced by calls into a compiled real-time-iteration MPC/MHE core behind
 * the same interface (uav_solver.h), which is unchanged.  <<< PLUG-IN POINT >>>
 * ========================================================================== */
#include "uav_solver.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------- MPC */
struct MPCProblem {
    MPCDims d;
    double  Ts;
    double  qp, qv, ru;         /* scalar weights (double-integrator LQR)   */
    double  kp, kv;             /* LQR gains (prepared)                     */
    double  xref[64], uref[32];
    double  a_max;
    int     prepared;
};

MPCProblem* mpc_create(const MPCDims* d) {
    MPCProblem* p = (MPCProblem*)calloc(1, sizeof(MPCProblem));
    if (!p) return NULL;
    p->d = *d; p->Ts = d->Ts;
    p->qp = 3.0; p->qv = 3.5; p->ru = 1.2; p->a_max = 4.0;
    p->prepared = 0;
    return p;
}

int mpc_set_weights(MPCProblem* p, const double* Qdiag,
                    const double* Rdiag, const double* QNdiag) {
    (void)QNdiag;
    if (!p) return -1;
    if (Qdiag) { p->qp = Qdiag[0]; p->qv = (p->d.nx > 3) ? Qdiag[3] : Qdiag[0]; }
    if (Rdiag) p->ru = Rdiag[0];
    p->prepared = 0;
    return 0;
}

int mpc_set_reference(MPCProblem* p, const double* xref, const double* uref) {
    if (!p) return -1;
    if (xref) memcpy(p->xref, xref, sizeof(double) * (size_t)p->d.nx);
    if (uref) memcpy(p->uref, uref, sizeof(double) * (size_t)p->d.nu);
    return 0;
}

int mpc_set_constraint_bounds(MPCProblem* p, const double* c_ub) {
    /* coordination layer injects collision-avoidance / actuator bounds here;
     * the reference LQR clips at a_max, the real solver applies them exactly. */
    (void)p; (void)c_ub; return 0;
}

/* RTI preparation: solve the scalar double-integrator DARE for the LQR gains */
int mpc_prepare(MPCProblem* p) {
    if (!p) return -1;
    const double dt = p->Ts;
    /* A=[[1,dt],[0,1]], B=[[.5dt^2],[dt]], Q=diag(qp,qv), R=ru */
    double s11 = p->qp, s12 = 0.0, s22 = p->qv;   /* cost-to-go P */
    for (int it = 0; it < 200; ++it) {
        /* K = (R + B'PB)^-1 B'PA ; standard 2-state DARE fixed point */
        double b0 = 0.5 * dt * dt, b1 = dt;
        double PB0 = s11 * b0 + s12 * b1, PB1 = s12 * b0 + s22 * b1;
        double BPB = b0 * PB0 + b1 * PB1 + p->ru;
        /* B'PA */
        double PA00 = s11, PA01 = s11 * dt + s12;
        double PA10 = s12, PA11 = s12 * dt + s22;
        double BPA0 = b0 * PA00 + b1 * PA10;
        double BPA1 = b0 * PA01 + b1 * PA11;
        double k0 = BPA0 / BPB, k1 = BPA1 / BPB;
        /* P+ = Q + A'PA - (A'PB) K */
        double APA00 = s11;
        double APA01 = s11 * dt + s12;
        double APA11 = dt * (s11 * dt + s12) + (s12 * dt + s22);
        double APB0 = PA00 * b0 + PA10 * b1;   /* = A'PB, symmetric use */
        double APB1 = PA01 * b0 + PA11 * b1;
        double n11 = p->qp + APA00 - APB0 * k0;
        double n12 = APA01 - APB0 * k1;
        double n22 = p->qv + APA11 - APB1 * k1;
        double d = fabs(n11 - s11) + fabs(n12 - s12) + fabs(n22 - s22);
        s11 = n11; s12 = n12; s22 = n22;
        p->kp = k0; p->kv = k1;
        if (d < 1e-12) break;
    }
    p->prepared = 1;
    return 0;
}

/* RTI feedback: u0 = a_ref - kp (p - p_ref) - kv (v - v_ref), clipped */
int mpc_feedback(MPCProblem* p, const double* xhat, double* u0) {
    if (!p || !xhat || !u0) return -1;
    if (!p->prepared) mpc_prepare(p);
    for (int ax = 0; ax < p->d.nu && ax < 3; ++ax) {
        double perr = xhat[ax] - p->xref[ax];
        double verr = xhat[3 + ax] - (p->d.nx > 3 ? p->xref[3 + ax] : 0.0);
        double a = (p->d.nu <= (int)0 ? 0.0 : p->uref[ax]) - p->kp * perr - p->kv * verr;
        if (a > p->a_max) a = p->a_max;
        if (a < -p->a_max) a = -p->a_max;
        u0[ax] = a;
    }
    return 0;
}

void mpc_destroy(MPCProblem* p) { free(p); }

/* ------------------------------------------------------------------- MHE */
struct MHEProblem {
    MHEDims d;
    double  Ts;
    double  s[6];               /* [p(3), v(3)]                     */
    double  P[36];              /* covariance                       */
    double  Q[6], r_gps;        /* process var (diag), gps var      */
    double  r_rel;              /* relative-measurement var         */
    int     have_gps;           /* set by prepare via u/measurement */
    int     init;
};

static void mat6_predict(double* P, double dt, const double* Q) {
    /* P <- F P F' + diag(Q), F=[[I,dt I],[0,I]] (block form) */
    double Pp[36];
    /* F P : rows 0..2 get row_i + dt*row_{i+3}; rows 3..5 unchanged */
    double FP[36];
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) {
            double v = P[i * 6 + j];
            if (i < 3) v += dt * P[(i + 3) * 6 + j];
            FP[i * 6 + j] = v;
        }
    /* (F P) F' : cols 0..2 get col_j + dt*col_{j+3} */
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) {
            double v = FP[i * 6 + j];
            if (j < 3) v += dt * FP[i * 6 + (j + 3)];
            Pp[i * 6 + j] = v;
        }
    for (int i = 0; i < 6; ++i) Pp[i * 6 + i] += Q[i];
    memcpy(P, Pp, sizeof(Pp));
}

/* Kalman update of the 3 position components with 3x3 measurement cov R */
static void kf_pos_update(double* s, double* P, const double* z, const double* R) {
    /* S = P_pp + R ; K = P_:p S^-1 ; standard 3-block update            */
    double S[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            S[i * 3 + j] = P[i * 6 + j] + R[i * 3 + j];
    /* invert 3x3 S */
    double a = S[0], b = S[1], c = S[2], d = S[3], e = S[4], f = S[5],
           g = S[6], h = S[7], k = S[8];
    double A = e * k - f * h, B = -(d * k - f * g), C = d * h - e * g;
    double det = a * A + b * B + c * C;
    if (fabs(det) < 1e-18) return;
    double id = 1.0 / det;
    double Si[9] = { A * id, (c * h - b * k) * id, (b * f - c * e) * id,
                     B * id, (a * k - c * g) * id, (c * d - a * f) * id,
                     C * id, (b * g - a * h) * id, (a * e - b * d) * id };
    /* innovation y = z - s_p */
    double y[3] = { z[0] - s[0], z[1] - s[1], z[2] - s[2] };
    /* K = P_:p Si  (6x3),  s += K y,  P -= K S K'  (use P -= P_:p Si P_p:) */
    double K[18];
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 3; ++j) {
            double v = 0;
            for (int l = 0; l < 3; ++l) v += P[i * 6 + l] * Si[l * 3 + j];
            K[i * 3 + j] = v;
        }
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 3; ++j) s[i] += K[i * 3 + j] * y[j];
    double Pn[36];
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) {
            double v = P[i * 6 + j];
            for (int l = 0; l < 3; ++l) v -= K[i * 3 + l] * P[l * 6 + j];
            Pn[i * 6 + j] = v;
        }
    memcpy(P, Pn, sizeof(Pn));
}

MHEProblem* mhe_create(const MHEDims* d) {
    MHEProblem* e = (MHEProblem*)calloc(1, sizeof(MHEProblem));
    if (!e) return NULL;
    e->d = *d; e->Ts = d->Ts;
    for (int i = 0; i < 3; ++i) { e->Q[i] = 0.03 * 0.03; e->Q[3 + i] = 0.10 * 0.10; }
    e->r_gps = 0.30 * 0.30; e->r_rel = 0.12 * 0.12; e->init = 0;
    return e;
}

int mhe_init(MHEProblem* e, const double* Vdiag, const double* Wdiag,
             const double* La0diag, const double* xa0) {
    (void)Wdiag;
    if (!e) return -1;
    if (Vdiag && Vdiag[0] > 0) e->r_gps = 1.0 / Vdiag[0];
    memset(e->P, 0, sizeof(e->P));
    for (int i = 0; i < 3; ++i) e->P[i * 6 + i] = La0diag ? La0diag[i] : 0.09;
    for (int i = 3; i < 6; ++i) e->P[i * 6 + i] = 0.05 * 0.05;
    for (int i = 0; i < 6; ++i) e->s[i] = xa0 ? xa0[i] : 0.0;
    e->init = 1;
    return 0;
}

/* prepare: IMU predict step with u_t interpreted as measured accel */
int mhe_prepare(MHEProblem* e, const double* u_prev, const double* u_t) {
    (void)u_prev;
    if (!e) return -1;
    double a[3] = { u_t ? u_t[0] : 0, u_t ? u_t[1] : 0, u_t ? u_t[2] : 0 };
    const double dt = e->Ts;
    for (int i = 0; i < 3; ++i) {
        e->s[i] += dt * e->s[3 + i] + 0.5 * dt * dt * a[i];
        e->s[3 + i] += dt * a[i];
    }
    mat6_predict(e->P, dt, e->Q);
    return 0;
}

/* feedback: GPS update with measurement y_t (position); output estimate */
int mhe_feedback(MHEProblem* e, const double* y_t, double* xhat) {
    if (!e) return -1;
    if (y_t) {
        double R[9] = { e->r_gps, 0, 0, 0, e->r_gps, 0, 0, 0, e->r_gps };
        kf_pos_update(e->s, e->P, y_t, R);
    }
    if (xhat) memcpy(xhat, e->s, sizeof(double) * 6);
    return 0;
}

/* covariance-aware relative fusion: z = y_rel + p_neighbour, cov = r_rel I + Pj */
int mhe_fuse_relative(MHEProblem* e, const double* z, const double* Pj) {
    if (!e || !z) return -1;
    double R[9] = { e->r_rel, 0, 0, 0, e->r_rel, 0, 0, 0, e->r_rel };
    if (Pj) for (int i = 0; i < 9; ++i) R[i] += Pj[i];
    kf_pos_update(e->s, e->P, z, R);
    return 0;
}

void mhe_destroy(MHEProblem* e) { free(e); }
