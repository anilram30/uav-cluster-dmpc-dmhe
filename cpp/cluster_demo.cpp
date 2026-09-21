// ==========================================================================
// cluster_demo.cpp -- end-to-end smoke test of the C-core + C++-coordination
// stack. Four agents assemble a ring formation on a double-integrator plant,
// coordinated by consensus + MPC + CBF, with distributed MHE. Agent 3 suffers a
// GNSS blackout mid-run and stays localised by relative fusion to neighbours.
//
// This exercises the SAME control/estimation/coordination path the report's
// Python study uses, through the production C API and C++ layer. Build: make.
// ==========================================================================
#include "agent.hpp"
#include <vector>
#include <random>
#include <cstdio>
#include <cmath>

int main() {
    const int Na = 4;
    const double dt = 0.05;
    std::mt19937 rng(7);
    std::normal_distribution<double> ngps(0.0, 0.30), nrel(0.0, 0.10), nacc(0.0, 0.08);

    // ring formation offsets (radius 3 m)
    std::vector<Vec3> off(Na);
    for (int i=0;i<Na;++i){ double a=2*M_PI*i/Na; off[i]={3*std::cos(a),3*std::sin(a),0}; }
    // ring communication graph
    std::vector<std::vector<int>> nbr(Na);
    for (int i=0;i<Na;++i){ nbr[i]={(i+1)%Na,(i+Na-1)%Na}; }

    std::vector<Agent*> ag;
    for (int i=0;i<Na;++i){ Vec3 p0={off[i][0]*0.7,off[i][1]*0.7,0};
        Agent* a=new Agent(i,p0,off[i]); a->p_true_=p0; a->v_true_={0,0,0}; ag.push_back(a); }

    auto gps_denied=[&](int i,double t){ return i==3 && t>=6.0 && t<12.0; };

    double form_err_final=0, rmse_mhe=0; int nwin=0;
    Vec3 accel_prev[8]={};
    for (int k=0;k<400;++k){
        double t=k*dt;
        Vec3 centre = {6.0*std::sin(0.15*t), 3.0*std::sin(0.25*t), 6.0};
        // gather previous messages
        std::vector<AgentMsg> msg(Na);
        for (int i=0;i<Na;++i) msg[i]=ag[i]->message();

        // estimation pass 1
        for (int i=0;i<Na;++i){
            Vec3 gps={ag[i]->p_true_[0]+ngps(rng),ag[i]->p_true_[1]+ngps(rng),ag[i]->p_true_[2]+ngps(rng)};
            Vec3 am ={accel_prev[i][0]+nacc(rng),accel_prev[i][1]+nacc(rng),accel_prev[i][2]+nacc(rng)};
            ag[i]->estimate_local(am, gps, !gps_denied(i,t));
        }
        for (int i=0;i<Na;++i) msg[i]=ag[i]->message();
        // estimation pass 2: relative fusion for denied agents
        for (int i=0;i<Na;++i){
            std::vector<AgentMsg> nb; std::vector<Vec3> rel;
            for (int j:nbr[i]){ nb.push_back(msg[j]);
                rel.push_back(Vec3{ag[i]->p_true_[0]-ag[j]->p_true_[0]+nrel(rng),
                                   ag[i]->p_true_[1]-ag[j]->p_true_[1]+nrel(rng),
                                   ag[i]->p_true_[2]-ag[j]->p_true_[2]+nrel(rng)}); }
            ag[i]->estimate_relative(nb, rel);
        }
        for (int i=0;i<Na;++i) msg[i]=ag[i]->message();

        // control
        double ferr=0;
        for (int i=0;i<Na;++i){
            std::vector<AgentMsg> nb; for (int j:nbr[i]) nb.push_back(msg[j]);
            ag[i]->consensus_centre(centre, nb);
            Vec3 a = ag[i]->mpc_command(centre);
            a = ag[i]->cbf_filter(a, nb);
            accel_prev[i]=a;
            ag[i]->integrate(a, dt);
            Vec3 slot = centre + off[i];
            ferr += norm(ag[i]->p_true_ - slot);
        }
        ferr/=Na;
        // estimation error (denied agent 3 during window)
        if (t>=6.0 && t<12.0){ Vec3 e = ag[3]->pos() - ag[3]->p_true_;
            rmse_mhe += dot(e,e); nwin++; }
        if (k==399) form_err_final=ferr;
    }
    rmse_mhe = std::sqrt(rmse_mhe/std::max(1,nwin));
    std::printf("[C++ cluster demo] agents=%d, steps=400\n", Na);
    std::printf("  final formation error = %.3f m\n", form_err_final);
    std::printf("  agent-3 position error during GNSS blackout (DMHE) = %.3f m\n", rmse_mhe);
    std::printf("  -> C numerical core + C++ coordination layer ran end-to-end.\n");
    for (auto a:ag) delete a;
    return 0;
}
