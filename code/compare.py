"""Is setting B really better than setting A? Paired bootstrap over the labelled vehicles.

Each truth vehicle contributes the rows it shares with the system ID matched to it (IDTP
part) and its own rows; resampling the vehicles with replacement, with the matching of the
full data kept fixed, gives a 95% interval for IDF1(B) - IDF1(A).

    python compare.py lam_l lam_f lam_c T  [lam_l lam_f lam_c T]
The second setting defaults to the setting used in the paper (0.816/0.122/0.061, T 0.816).
"""
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

from grid import (load_window_blocks, params, CUR, T_CUR, load_labels, truth_ids, window_rows,
                  replay, labels_at_rows)


def per_vehicle(sys_ids, truth):
    s_u, s_i = np.unique(sys_ids, return_inverse=True)
    nt = int(truth.max()) + 1
    C = np.bincount(s_i * nt + truth, minlength=len(s_u) * nt).reshape(len(s_u), nt)
    r, c = linear_sum_assignment(-C)
    contrib = np.zeros(nt)
    contrib[c] = C[r, c]
    return contrib


def main():
    v = [float(x) for x in sys.argv[1:]]
    B = (tuple(v[0:3]), v[3])
    A = (tuple(v[4:7]), v[7]) if len(v) >= 8 else (CUR, T_CUR)
    meta, qs, latest = load_labels()
    tid = truth_ids(meta, qs, latest)
    rows = window_rows().reset_index(drop=True)
    truth = rows['seg'].map(tid).to_numpy().astype(np.int64)
    blocks = load_window_blocks()
    n_t = np.bincount(truth)
    c = {}
    for name, (w, t) in (('A', A), ('B', B)):
        ev, _, _ = replay(blocks, params=params(w, t))
        c[name] = per_vehicle(labels_at_rows(rows[['frame', 'uid']], ev), truth)
    full = (c['B'].sum() - c['A'].sum()) / n_t.sum()
    rng = np.random.RandomState(0)
    k = len(n_t)
    diffs = []
    for _ in range(5000):
        s = rng.randint(0, k, k)
        diffs.append((c['B'][s].sum() - c['A'][s].sum()) / n_t[s].sum())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print('A %s T %.3f: IDF1 %.2f%%' % (np.round(A[0], 3), A[1], 100 * c['A'].sum() / n_t.sum()))
    print('B %s T %.3f: IDF1 %.2f%%' % (np.round(B[0], 3), B[1], 100 * c['B'].sum() / n_t.sum()))
    print('B - A = %+.2f points, 95%% interval %+.2f to %+.2f (bootstrap over %d vehicles); '
          'vehicles that differ: %d' % (100 * full, 100 * lo, 100 * hi, k, int((c['A'] != c['B']).sum())))


if __name__ == '__main__':
    main()
