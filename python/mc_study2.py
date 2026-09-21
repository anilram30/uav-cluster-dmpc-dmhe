"""
mc_study2.py -- second Monte-Carlo campaign:
  equiv_py / equiv_cmhe / equiv_cmpc / equiv_cboth :
      4-configuration C-in-the-loop equivalence study, 15 paired seeds each
      (identical noise streams across configurations).
  mh_sweep5 / mh_ring :
      multi-hop leveled-anchoring recovery of the two anchor-coverage-broken
      scenarios, 12 seeds each.
Appends to mc_results.json (resumable).
"""
import json
import os
from multiprocessing import Pool
from sim_hexacluster import run

OUT = 'mc_results.json'
SEEDS_EQ = list(range(100, 115))            # 15 paired seeds
SEEDS_VAR = list(range(200, 212))           # 12


def jobs():
    J = []
    for name, impl in [('equiv_py', 'py'), ('equiv_cmhe', 'c_mhe'),
                       ('equiv_cmpc', 'c_mpc'), ('equiv_cboth', 'c_both')]:
        for s in SEEDS_EQ:
            J.append((name, s, dict(seed=s, impl=impl)))
    for s in SEEDS_VAR:
        J.append(('mh_sweep5', s, dict(seed=s, denied=(1, 2, 3, 4, 5), multihop=True)))
    for s in SEEDS_VAR:
        J.append(('mh_ring', s, dict(seed=s, topology='ring', multihop=True)))
    return J


def worker(arg):
    name, seed, kw = arg
    try:
        m = run(**kw)
        m['scenario'] = name
        return m
    except Exception as e:
        return dict(scenario=name, seed=seed, crashed=True, error=str(e))


def main():
    results = {}
    if os.path.exists(OUT):
        results = json.load(open(OUT))
    done = {(r['scenario'], r['seed']) for rs in results.values() for r in rs}
    todo = [j for j in jobs() if (j[0], j[1]) not in done]
    print(f"todo: {len(todo)} runs")
    with Pool(2) as pool:
        for i, m in enumerate(pool.imap_unordered(worker, todo)):
            results.setdefault(m['scenario'], []).append(m)
            json.dump(results, open(OUT, 'w'), indent=1)
            print(f"[{i+1}/{len(todo)}] {m['scenario']} seed={m['seed']} "
                  f"DMHE_den={m.get('rmse_mhe_denied', float('nan')):.3f} "
                  f"minsep={m.get('min_sep', float('nan')):.2f} "
                  f"div={m.get('diverged', m.get('crashed'))}", flush=True)
    print("campaign 2 complete ->", OUT)


if __name__ == '__main__':
    main()
