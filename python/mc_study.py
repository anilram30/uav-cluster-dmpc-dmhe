"""
mc_study.py -- Monte-Carlo robustness campaign for the hexacopter cluster.

Scenarios
  main      : 30 noise seeds, nominal configuration (3/6 GNSS-denied)
  sweep_k   : GNSS-denial severity sweep, contiguous denied sets of size 1..5
  ring      : sparser communication graph (plain ring, degree 2, lambda2=1)
  edgedrop  : temporary loss of comm edge (3,2) during the blackout
  ekfctrl   : closed-loop baseline -- controller fed by the dead-reckoning
              local filter instead of the DMHE

Writes/updates mc_results.json (resumable: completed (scenario,seed) pairs are
skipped). Run:  python3 mc_study.py
"""
import json
import os
from multiprocessing import Pool
from sim_hexacluster import run

OUT = 'mc_results.json'
SEEDS_MAIN = list(range(100, 130))          # 30
SEEDS_VAR = list(range(200, 212))           # 12

DENY_SETS = {
    1: (3,),
    2: (3, 4),
    3: (3, 4, 5),
    4: (2, 3, 4, 5),
    5: (1, 2, 3, 4, 5),
}


def jobs():
    J = []
    for s in SEEDS_MAIN:
        J.append(('main', s, dict(seed=s)))
    for k, dset in DENY_SETS.items():
        for s in SEEDS_VAR:
            J.append((f'sweep_{k}', s, dict(seed=s, denied=dset)))
    for s in SEEDS_VAR:
        J.append(('ring', s, dict(seed=s, topology='ring')))
    for s in SEEDS_VAR:
        J.append(('edgedrop', s, dict(seed=s, edge_drop=(3, 2, 14.0, 18.0))))
    for s in SEEDS_VAR:
        J.append(('ekfctrl', s, dict(seed=s, ctrl_est='ekf')))
    return J


def worker(arg):
    name, seed, kw = arg
    try:
        m = run(**kw)
        m['scenario'] = name
        return m
    except Exception as e:                    # a crash is itself a data point
        return dict(scenario=name, seed=seed, crashed=True, error=str(e))


def main():
    results = {}
    if os.path.exists(OUT):
        results = json.load(open(OUT))
    done = {(r['scenario'], r['seed']) for rs in results.values() for r in rs}
    todo = [j for j in jobs() if (j[0], j[1]) not in done]
    print(f"todo: {len(todo)} runs ({len(done)} already done)")
    with Pool(2) as pool:
        for i, m in enumerate(pool.imap_unordered(worker, todo)):
            results.setdefault(m['scenario'], []).append(m)
            json.dump(results, open(OUT, 'w'), indent=1)
            print(f"[{i+1}/{len(todo)}] {m['scenario']} seed={m['seed']} "
                  f"DMHE_den={m.get('rmse_mhe_denied', float('nan')):.3f} "
                  f"minsep={m.get('min_sep', float('nan')):.2f} "
                  f"div={m.get('diverged', m.get('crashed'))}", flush=True)
    print("campaign complete ->", OUT)


if __name__ == '__main__':
    main()
