"""
analyze_mc2.py -- aggregate campaign 2 (C-in-the-loop equivalence + multi-hop
recovery) and regenerate the GNSS-denial sweep figure with the recovery curve.
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


def stats(scn, key):
    v = np.array([r[key] for r in R[scn] if not r.get('crashed')])
    return v.mean(), np.median(v), np.percentile(v, 95), v.min(), v.max()


print("=== C-in-the-loop equivalence (15 paired seeds per configuration) ===")
print(f"{'config':<12} {'DMHE_den mean/med/p95':>24} {'shape':>7} {'minsep(min)':>11} "
      f"{'viol':>5} {'div':>4} {'in-loop solver med us':>24}")
for scn, lab in [('equiv_py', 'py'), ('equiv_cmhe', 'C-MHE'),
                 ('equiv_cmpc', 'C-NMPC'), ('equiv_cboth', 'C-both')]:
    a = stats(scn, 'rmse_mhe_denied'); sh = stats(scn, 'shape_err_hold')
    ms = stats(scn, 'min_sep')
    viol = sum(r['sep_violations'] for r in R[scn])
    div = sum(1 for r in R[scn] if r.get('diverged') or r.get('crashed'))
    tail = ''
    tm = [r['c_mhe_us'] for r in R[scn] if 'c_mhe_us' in r]
    tc = [r['c_mpc_us'] for r in R[scn] if 'c_mpc_us' in r]
    if tm:
        m = np.median(np.array(tm), 0)
        tail += f" MHE({m[0]:.1f},{m[1]:.2f},{m[2]:.1f})"
    if tc:
        m = np.median(np.array(tc), 0)
        tail += f" MPC({m[0]:.1f},{m[1]:.1f})"
    print(f"{lab:<12} {a[0]:>8.3f}/{a[1]:.3f}/{a[2]:.3f} {sh[0]:>7.3f} "
          f"{ms[3]:>11.3f} {viol:>5d} {div:>4d} {tail:>24}")

print()
print("=== multi-hop anchoring recovery (12 seeds each) ===")
for scn, base in [('mh_sweep5', 'sweep_5'), ('mh_ring', 'ring')]:
    a = stats(scn, 'rmse_mhe_denied'); b = stats(base, 'rmse_mhe_denied')
    ms = stats(scn, 'min_sep'); mb = stats(base, 'min_sep')
    sh = stats(scn, 'shape_err_hold')
    viol = sum(r['sep_violations'] for r in R[scn])
    print(f"{scn:<10} DMHE_den {b[0]:.3f} -> {a[0]:.3f} (p95 {a[2]:.3f}) | "
          f"minsep {mb[3]:.2f} -> {ms[3]:.2f} | shape {sh[0]:.3f} | viol {viol}")

# ------------- updated denial sweep figure with recovery curve -------------
ks = [1, 2, 3, 4, 5]
mhe_m, mhe_lo, mhe_hi, kf_m, kf_lo, kf_hi = [], [], [], [], [], []
for k in ks:
    a = np.array([r['rmse_mhe_denied'] for r in R[f'sweep_{k}']])
    b = np.array([r['rmse_kf_denied'] for r in R[f'sweep_{k}']])
    mhe_m.append(a.mean()); mhe_lo.append(a.min()); mhe_hi.append(a.max())
    kf_m.append(b.mean()); kf_lo.append(b.min()); kf_hi.append(b.max())
mh5 = np.array([r['rmse_mhe_denied'] for r in R['mh_sweep5']])
mh_m = mhe_m[:4] + [mh5.mean()]
mh_lo = mhe_lo[:4] + [mh5.min()]
mh_hi = mhe_hi[:4] + [mh5.max()]

fig, ax = plt.subplots(figsize=(6.4, 3.7))
ax.fill_between(ks, kf_lo, kf_hi, color=C[1], alpha=0.18, lw=0)
ax.plot(ks, kf_m, 'o-', color=C[1], lw=1.6, label='local filter (dead-reckoning)')
ax.fill_between(ks, mhe_lo, mhe_hi, color=C[0], alpha=0.18, lw=0)
ax.plot(ks, mhe_m, 's-', color=C[0], lw=1.6, label='DMHE, one-hop anchoring')
ax.fill_between(ks, mh_lo, mh_hi, color=C[2], alpha=0.18, lw=0)
ax.plot(ks, mh_m, 'D--', color=C[2], lw=1.6, ms=5,
        label='DMHE, multi-hop anchoring')
ax.annotate('one-hop anchor unreachable\nfor agent 3 at 5/6',
            xy=(5, mhe_m[-1]), xytext=(3.35, mhe_m[-1]*0.80),
            fontsize=7, color='0.25',
            arrowprops=dict(arrowstyle='->', color='0.45', lw=0.8))
ax.annotate('recovered by\nmulti-hop', xy=(5, mh_m[-1]),
            xytext=(4.35, 0.30), fontsize=7, color=C[2],
            arrowprops=dict(arrowstyle='->', color=C[2], lw=0.8))
ax.set_xticks(ks)
ax.set_xlabel('number of GNSS-denied agents (of 6, contiguous)')
ax.set_ylabel('denied-agent position RMSE [m]')
ax.set_title('GNSS-denial severity sweep (12 seeds per point; bands = min--max)')
ax.legend(fontsize=8, loc='upper left')
fig.savefig(os.path.join(FIGDIR, 'fig_denial_sweep.pdf'))
fig.savefig(os.path.join(FIGDIR, 'fig_denial_sweep.png'), dpi=140)
print('\nwrote fig_denial_sweep (with multi-hop recovery curve)')
