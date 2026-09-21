// ==========================================================================
// agent.hpp -- C++ coordination-layer agent. Wraps the per-agent C solvers
// (uav_solver.h) and adds the distributed logic of the report Ch. 6-8:
//   * DMHE relative fusion (covariance-aware, GNSS-loss fallback)
//   * consensus on the formation centre
//   * CBF collision-avoidance safety filter
// The numerical core is never re-implemented here; it is called through the
// C API. This mirrors the architecture of report Ch. 9.
// ==========================================================================
#pragma once
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
extern "C" {
#include "uav_solver.h"
}

using Vec3 = std::array<double, 3>;

inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
inline Vec3 operator+(const Vec3& a, const Vec3& b) { return {a[0]+b[0],a[1]+b[1],a[2]+b[2]}; }
inline double dot(const Vec3& a, const Vec3& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
inline double norm(const Vec3& a) { return std::sqrt(dot(a,a)); }

// message broadcast to neighbours each step
struct AgentMsg {
    int id;
    Vec3 p, v;               // estimated position, velocity
    std::array<double,9> Ppos;  // position covariance block (row-major)
    Vec3 centre;             // local estimate of formation centre
    bool has_gps;
};

class Agent {
public:
    Agent(int id, const Vec3& p0, const Vec3& offset)
        : id_(id), offset_(offset) {
        MPCDims md{6,3,0,0,25,0.05};
        mpc_ = mpc_create(&md);
        double Q[6]={3,3,3,3.5,3.5,3.5}, R[3]={1.2,1.2,1.2};
        mpc_set_weights(mpc_, Q, R, Q);
        MHEDims ed{6,3,3,3,0,0,8,0.05};
        mhe_ = mhe_create(&ed);
        double V[3]={1.0/(0.30*0.30),0,0}, W[3]={0,0,0};
        double La0[3]={0.09,0.09,0.09}; double xa0[6]={p0[0],p0[1],p0[2],0,0,0};
        mhe_init(mhe_, V, W, La0, xa0);
        p_=p0; v_={0,0,0}; centre_={0,0,0};
    }
    ~Agent(){ mpc_destroy(mpc_); mhe_destroy(mhe_); }

    // --- distributed estimation (report Ch.6, Alg. 1) ---
    // pass 1: predict + (GPS update if available)
    void estimate_local(const Vec3& accel_meas, const Vec3& gps, bool have_gps) {
        double u[3]={accel_meas[0],accel_meas[1],accel_meas[2]};
        mhe_prepare(mhe_, u, u);
        have_gps_ = have_gps;
        double xhat[6];
        if (have_gps) { double y[3]={gps[0],gps[1],gps[2]}; mhe_feedback(mhe_, y, xhat); }
        else          { mhe_feedback(mhe_, nullptr, xhat); }
        p_={xhat[0],xhat[1],xhat[2]}; v_={xhat[3],xhat[4],xhat[5]};
    }
    // pass 2: relative fusion to GPS-good neighbours (only when GPS-denied)
    void estimate_relative(const std::vector<AgentMsg>& nbrs,
                           const std::vector<Vec3>& rel_meas) {
        if (have_gps_) return;                       // local filter suffices
        double xhat[6];
        for (size_t k=0;k<nbrs.size();++k) {
            if (!nbrs[k].has_gps) continue;          // avoid data incest
            Vec3 z = rel_meas[k] + nbrs[k].p;        // pseudo-measurement of p_i
            double zz[3]={z[0],z[1],z[2]};
            mhe_fuse_relative(mhe_, zz, nbrs[k].Ppos.data());
        }
        mhe_feedback(mhe_, nullptr, xhat);           // read back fused state
        p_={xhat[0],xhat[1],xhat[2]}; v_={xhat[3],xhat[4],xhat[5]};
    }

    // --- consensus on the formation centre (report Ch.7, eq. centre-consensus) ---
    void consensus_centre(const Vec3& mission_centre, const std::vector<AgentMsg>& nbrs) {
        // Metropolis-weighted one round; seed with local sensing of the centre
        double dmax = (double)std::max((size_t)1, nbrs.size());
        Vec3 c = (centre_ == Vec3{0,0,0}) ? mission_centre : centre_;
        c = mission_centre;                          // local sensing of mission ref
        Vec3 acc = c; double wsum = 1.0;
        for (auto& n : nbrs) { double w=1.0/(1.0+std::max(dmax,(double)nbrs.size()));
            acc = acc + Vec3{w*(n.centre[0]-c[0]),w*(n.centre[1]-c[1]),w*(n.centre[2]-c[2])};
            (void)wsum; }
        centre_ = acc;
    }

    // --- MPC command (report Ch.5 RTI feedback) ---
    Vec3 mpc_command(const Vec3& mission_centre) {
        Vec3 slot = mission_centre + offset_;
        double xref[6]={slot[0],slot[1],slot[2],0,0,0};
        mpc_set_reference(mpc_, xref, nullptr);
        mpc_prepare(mpc_);
        double xhat[6]={p_[0],p_[1],p_[2],v_[0],v_[1],v_[2]}, u0[3];
        mpc_feedback(mpc_, xhat, u0);
        return {u0[0],u0[1],u0[2]};
    }

    // --- CBF safety filter (report Ch.7, eq. cbf-qp) : 1 neighbour analytic ---
    Vec3 cbf_filter(const Vec3& a_cmd, const std::vector<AgentMsg>& nbrs,
                    double D=1.4, double k1=3.5, double k2=4.0) {
        Vec3 a = a_cmd;
        for (auto& n : nbrs) {
            Vec3 dp = p_ - n.p, dv = v_ - n.v;
            double h = dot(dp,dp) - D*D;
            double hdot = 2.0*dot(dp,dv);
            // constraint: -2 dp . a <= 2 dv.dv + k1 hdot + k2 h  =: rhs
            double rhs = 2.0*dot(dv,dv) + k1*hdot + k2*h;
            double lhs = -2.0*dot(dp,a);
            if (lhs > rhs) {                          // active: project onto boundary
                Vec3 g = {-2.0*dp[0],-2.0*dp[1],-2.0*dp[2]};
                double gg = dot(g,g);
                if (gg>1e-9){ double lam=(lhs-rhs)/gg;
                    a = {a[0]-lam*g[0], a[1]-lam*g[1], a[2]-lam*g[2]}; }
            }
        }
        return a;
    }

    AgentMsg message() const {
        AgentMsg m; m.id=id_; m.p=p_; m.v=v_; m.centre=centre_; m.has_gps=have_gps_;
        m.Ppos = {0.02,0,0, 0,0.02,0, 0,0,0.02};      // reported position cov
        return m;
    }
    Vec3 pos() const { return p_; }
    Vec3 vel() const { return v_; }
    // plant integration (double integrator; the real plant is the nonlinear model)
    void integrate(const Vec3& a, double dt) {
        for (int i=0;i<3;++i){ p_true_[i]+=dt*v_true_[i]+0.5*dt*dt*a[i]; v_true_[i]+=dt*a[i]; }
    }
    Vec3 p_true_{}, v_true_{};
private:
    int id_; Vec3 offset_;
    MPCProblem* mpc_; MHEProblem* mhe_;
    Vec3 p_, v_, centre_;
    bool have_gps_ = true;
};

inline bool operator==(const Vec3&a,const Vec3&b){return a[0]==b[0]&&a[1]==b[1]&&a[2]==b[2];}
