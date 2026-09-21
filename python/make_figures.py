"""
make_figures.py -- generate all report figures from the saved simulation
results. Outputs PDF (for LaTeX) + PNG into ../figures/ (override with UAV_FIGDIR).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

FIGDIR = os.environ.get('UAV_FIGDIR',
                       os.path.join(os.path.dirname(__file__), '..', 'figures'))
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'mathtext.fontset': 'cm',
    'axes.grid': True, 'grid.alpha': 0.25, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 130, 'savefig.bbox': 'tight',
})
C = ['#2b6cb0', '#c05621', '#2f855a', '#6b46c1', '#b83280', '#718096']


def save(fig, name):
    fig.savefig(os.path.join(FIGDIR, name+'.pdf'))
    fig.savefig(os.path.join(FIGDIR, name+'.png'), dpi=140)
    plt.close(fig)
    print('wrote', name)


def hexa_figures():
    d = np.load('results_hexa.npz')
    t = d['t']; X = d['Xtrue']; Na = int(d['Na'])
    denied = d['gps_denied']; t0, t1 = d['deny_window']

    # --- 3D trajectories ---
    fig = plt.figure(figsize=(6.2, 5.0))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(Na):
        ax.plot(X[:, i, 0], X[:, i, 1], X[:, i, 2], color=C[i % 6], lw=1.3,
                label=f'agent {i}' + (' (GNSS-denied)' if i in denied else ''))
        ax.scatter(X[0, i, 0], X[0, i, 1], X[0, i, 2], color=C[i % 6], marker='o', s=18)
        ax.scatter(X[-1, i, 0], X[-1, i, 1], X[-1, i, 2], color=C[i % 6], marker='^', s=30)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z (up) [m]')
    ax.set_title('Hexacopter cluster: closed-loop trajectories\n'
                 '(o start, $\\triangle$ end; take-off, formation, sweep, reconfig)')
    ax.legend(fontsize=6.5, loc='upper left', ncol=2)
    ax.view_init(elev=22, azim=-58)
    save(fig, 'fig_hexa_traj3d')

    # --- formation error + min separation ---
    fig, ax = plt.subplots(2, 1, figsize=(6.2, 4.4), sharex=True)
    ax[0].plot(t, d['form_err'], color=C[0], lw=1.4)
    ax[0].axvspan(t0, t1, color='0.85', alpha=0.6, lw=0)
    ax[0].set_ylabel('formation error [m]')
    ax[0].set_title('Formation tracking and inter-agent safety')
    ax[0].text(0.5*(t0+t1), ax[0].get_ylim()[1]*0.85, 'GNSS blackout',
               ha='center', fontsize=7, color='0.35')
    ax[1].plot(t, d['min_dist'], color=C[2], lw=1.4, label='min pairwise distance')
    ax[1].axhline(1.4, color=C[1], ls='--', lw=1.1, label='safety radius $D=1.4$ m')
    ax[1].axvspan(t0, t1, color='0.85', alpha=0.6, lw=0)
    ax[1].set_ylabel('separation [m]'); ax[1].set_xlabel('time [s]')
    ax[1].legend(fontsize=7.5, loc='lower right')
    ax[1].set_ylim(0, None)
    save(fig, 'fig_hexa_formation')

    # --- DMHE vs EKF estimation error (denied agents) ---
    e_m = np.sqrt(np.mean((d['p_hat_mhe'] - X[:, :, 0:3])**2, axis=2))   # (T,Na)
    e_k = np.sqrt(np.mean((d['p_hat_kf'] - X[:, :, 0:3])**2, axis=2))
    den = list(denied)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(t, e_k[:, den].mean(1), color=C[1], lw=1.6,
            label='local EKF (GNSS+IMU, dead-reckons in blackout)')
    ax.plot(t, e_m[:, den].mean(1), color=C[0], lw=1.6,
            label='DMHE (relative fusion to GNSS-good neighbours)')
    ax.axvspan(t0, t1, color='0.85', alpha=0.6, lw=0)
    ymax = max(e_k[:, den].mean(1).max(), 0.1)
    ax.annotate('GNSS blackout\n(agents 3,4,5)', xy=(0.5*(t0+t1), 0.42*ymax),
                ha='center', va='center', fontsize=7, color='0.3')
    ax.set_xlabel('time [s]'); ax.set_ylabel('position RMSE [m]')
    ax.set_ylim(0, 1.12*ymax)
    ax.set_title('Distributed MHE keeps GNSS-denied agents localised')
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
    save(fig, 'fig_hexa_dmhe')

    # --- solve time histogram ---
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ax.hist(d['step_ms']*1e3, bins=40, color=C[3], alpha=0.85)
    ax.axvline(np.median(d['step_ms'])*1e3, color=C[1], ls='--',
               label=f"median {np.median(d['step_ms'])*1e3:.0f} $\\mu$s")
    ax.set_xlabel('per-step estimation+coordination wall time [$\\mu$s]')
    ax.set_ylabel('count'); ax.legend(fontsize=8)
    ax.set_title('Reference-layer compute per control step (6 agents, 1 core)')
    save(fig, 'fig_hexa_timing')


def fw_figures():
    if not os.path.exists('results_fw.npz'):
        print('(fixed-wing results not present yet)')
        return
    d = np.load('results_fw.npz')
    t = d['t']; X = d['Xtrue']; Na = int(d['Na'])
    # top-down + altitude
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.2))
    for i in range(Na):
        ax[0].plot(X[:, i, 1], X[:, i, 0], color=C[i % 6], lw=1.1, label=f'UAV {i}')
        ax[1].plot(t, -X[:, i, 2], color=C[i % 6], lw=1.1)
    ax[0].set_xlabel('East $p_e$ [m]'); ax[0].set_ylabel('North $p_n$ [m]')
    ax[0].set_title('Fixed-wing formation (top view)'); ax[0].axis('equal')
    ax[0].legend(fontsize=6.5, ncol=2, loc='best')
    ax[1].set_xlabel('time [s]'); ax[1].set_ylabel('altitude $h$ [m]')
    ax[1].set_title('Altitude')
    save(fig, 'fig_fw_traj')

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.plot(t, d['form_err'], color=C[0], lw=1.5)
    ax.set_xlabel('time [s]'); ax.set_ylabel('formation error [m]')
    ax.set_title('Fixed-wing coordinated-turn formation error')
    save(fig, 'fig_fw_formation')


if __name__ == '__main__':
    hexa_figures()
    fw_figures()
    print('figures ->', os.path.abspath(FIGDIR))
