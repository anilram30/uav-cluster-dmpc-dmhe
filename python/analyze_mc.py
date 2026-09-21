"""
analyze_mc.py -- aggregate mc_results.json into robustness statistics and the
GNSS-denial sweep figure for the report. Run after mc_study.py completes.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGDIR = os.environ.get('UAV_FIGDIR',
                       os.path.join(os.path.dirname(__file__), '..', 'figures'))
os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'mathtext.fontset': 'cm',
    'axes.grid': True, 'grid.alpha': 0.25, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 130, 'savefig.bbox': 'tight',
})
C = ['#2b6cb0', '#c05621', '#2f855a', '#6b46c1', '#b83280', '#718096']

R = json.load(open('mc_results.json'))


def agg(scn, key):
    vals = np.array([r[key] for r in R[scn] if not r.get('crashed')])
    return dict(mean=float(np.mean(vals)), med=float(np.median(vals)),
                p95=float(np.percentile(vals, 95)), min=float(np.min(vals)),
                max=float(np.max(vals)), n=len(vals))


def line(scn):
    a = {k: agg(scn, k) for k in
         ['rmse_mhe_denied', 'rmse_kf_denied', 'shape_err_hold', 'form_err_hold',
          'form_err_max_blackout', 'min_sep', 'sep_violations', 'step_ms_med',
          'qp_failures']}
    div = sum(1 for r in R[scn] if r.get('diverged') or r.get('crashed'))
    return a, div


summary = {}
print(f"{'scenario':<10} {'n':>3} {'DMHE_den mean/med/p95':>24} {'KF_den':>8} "
      f"{'shape':>6} {'minsep':>7} {'viol':>4} {'div':>3} {'ms med':>7}")
for scn in sorted(R):
    a, div = line(scn)
    summary[scn] = dict(agg={k: v for k, v in a.items()}, diverged=div,
                        n=a['min_sep']['n'])
    print(f"{scn:<10} {a['min_sep']['n']:>3} "
          f"{a['rmse_mhe_denied']['mean']:>7.3f}/{a['rmse_mhe_denied']['med']:.3f}"
          f"/{a['rmse_mhe_denied']['p95']:.3f} "
          f"{a['rmse_kf_denied']['mean']:>8.3f} "
          f"{a['shape_err_hold']['mean']:>6.3f} {a['min_sep']['min']:>7.3f} "
          f"{int(sum(r['sep_violations'] for r in R[scn])):>4d} {div:>3d} "
          f"{a['step_ms_med']['med']:>7.3f}")
json.dump(summary, open('mc_summary.json', 'w'), indent=1)

# ---------------- GNSS-denial sweep figure ----------------
ks = [1, 2, 3, 4, 5]
mhe_m, mhe_lo, mhe_hi, kf_m, kf_lo, kf_hi = [], [], [], [], [], []
for k in ks:
    a = np.array([r['rmse_mhe_denied'] for r in R[f'sweep_{k}']])
    b = np.array([r['rmse_kf_denied'] for r in R[f'sweep_{k}']])
    mhe_m.append(a.mean()); mhe_lo.append(a.min()); mhe_hi.append(a.max())
    kf_m.append(b.mean()); kf_lo.append(b.min()); kf_hi.append(b.max())

fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.fill_between(ks, kf_lo, kf_hi, color=C[1], alpha=0.18, lw=0)
ax.plot(ks, kf_m, 'o-', color=C[1], lw=1.6,
        label='local filter (dead-reckoning)')
ax.fill_between(ks, mhe_lo, mhe_hi, color=C[0], alpha=0.18, lw=0)
ax.plot(ks, mhe_m, 's-', color=C[0], lw=1.6,
        label='DMHE (relative fusion)')
ax.annotate('agent 3 has no GNSS-good\nneighbour at 5/6 denied\n(anchor unreachable)',
            xy=(5, mhe_m[-1]), xytext=(3.55, mhe_m[-1]*0.72 + 0.32),
            fontsize=7, color='0.25',
            arrowprops=dict(arrowstyle='->', color='0.45', lw=0.8))
ax.set_xticks(ks)
ax.set_xlabel('number of GNSS-denied agents (of 6, contiguous)')
ax.set_ylabel('denied-agent position RMSE [m]')
ax.set_title('GNSS-denial severity sweep (12 Monte-Carlo seeds each; band = min--max)')
ax.legend(fontsize=8, loc='upper left')
fig.savefig(os.path.join(FIGDIR, 'fig_denial_sweep.pdf'))
fig.savefig(os.path.join(FIGDIR, 'fig_denial_sweep.png'), dpi=140)
print('wrote fig_denial_sweep')

# ---------------- headline aggregates for the report text ----------------
tot_runs = sum(len(v) for v in R.values())
tot_viol = sum(r['sep_violations'] for v in R.values() for r in v)
tot_div = sum(1 for v in R.values() for r in v if r.get('diverged') or r.get('crashed'))
tot_qp = sum(r.get('qp_failures', 0) for v in R.values() for r in v)
print(f"\nTOTAL runs={tot_runs} separation_violations={tot_viol} "
      f"divergences={tot_div} qp_failures={tot_qp}")
for scn in ['main', 'ring', 'edgedrop', 'ekfctrl']:
    if scn in R:
        a, _ = line(scn)
        print(f"{scn}: blackout max form err mean={a['form_err_max_blackout']['mean']:.2f} "
              f"p95={a['form_err_max_blackout']['p95']:.2f} | "
              f"min_sep min={a['min_sep']['min']:.2f}")
