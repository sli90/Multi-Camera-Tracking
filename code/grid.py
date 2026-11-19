"""Grid search over the association weights and threshold, scored by IDF1 against the
labels of the 2-minute validation clip (tracks frames 1..1800).

Normalized scale (a pure rescale of the cost used in the live code):
    cost = lam_l * (distance / 5 m) + lam_f * appearance + lam_c * (0 same class, 1 different)
    lam_l + lam_f + lam_c = 1; a pair is matched if cost <= T
    (when the track is an oversize truck: cost <= 10 / 2.45 = 4.08, kept fixed)
The live code used 0.4 / 0.3 / 0.3 with class mismatch 0.5 and threshold 2 (oversize 10);
on this scale that is lam = (0.816, 0.122, 0.061), T = 0.816.

IDF1: every row of tracks.csv in the window carries the replayed ID (system) and the
labelled vehicle of its segment (truth). IDF1 = IDTP / rows, where IDTP is the number of
rows shared under the one-to-one matching of system IDs to truth vehicles that maximizes
it. Both sides have the same rows, so IDP = IDR = IDF1.

    python grid.py --check    checks only (exact replay, rescale, scorer)
    python grid.py            checks, then every setting on all CPUs but two
Writes results/grid_idf1.csv.
"""
import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from truth import window_rows, load_labels, truth_ids, DATA, RES, F_LAST   # noqa: E402
from replay_association import replay, labels_at_rows                       # noqa: E402

K = 2.45                                  # live code: 0.4 * 5 m + 0.3 + 0.3 * 0.5
D0 = 5.0                                  # metres per unit of location cost
THR_OVERSIZE = 10.0 / K
CUR, T_CUR = (2.0 / K, 0.3 / K, 0.15 / K), 2.0 / K
TAB1, T_TAB1 = (2.25 / 2.8, 0.35 / 2.8, 0.20 / 2.8), 1.6 / 2.8     # former Table 1 values, rescaled
STEP = 0.05
T_GRID = [round(STEP * k, 2) for k in range(1, 31)]                 # 0.05 .. 1.50


def params(lam, thr):
    ll, lf, lc = lam
    return {'w_loc': ll / D0, 'w_feat': lf, 'w_cls': lc, 'cls_mismatch': 1.0,
            'thr': thr, 'thr_oversize': THR_OVERSIZE}


def weight_grid():
    n = int(round(1 / STEP))
    return [(i / n, j / n, (n - i - j) / n) for i in range(n + 1) for j in range(n + 1 - i)]


def settings():
    S = [('grid', w, t) for w in weight_grid() for t in T_GRID]
    S += [('current', CUR, T_CUR), ('table1', TAB1, T_TAB1)]
    S += [('at_current_T', w, T_CUR) for w in weight_grid()]
    S += [('current_weights', CUR, t) for t in T_GRID]
    for d in [(.1, -.1, 0), (-.1, .1, 0), (.1, 0, -.1), (-.1, 0, .1), (0, .1, -.1), (0, -.1, .1)]:
        w = tuple(a + b for a, b in zip(CUR, d))
        if min(w) >= 0:
            S.append(('current_pm0.1', w, T_CUR))
    return S


def load_window_blocks(last=F_LAST):
    """Recorded matching steps with frame <= last, in step order."""
    with open(os.path.join(DATA, 'steps.pkl'), 'rb') as f:
        blocks = pickle.load(f)
    return [b for b in blocks if b['frame'] <= last]


def idtp(sys_ids, truth):
    """Rows shared by the best one-to-one matching of system IDs to truth vehicles."""
    s_u, s_i = np.unique(sys_ids, return_inverse=True)
    nt = int(truth.max()) + 1
    C = np.bincount(s_i * nt + truth, minlength=len(s_u) * nt).reshape(len(s_u), nt)
    r, c = linear_sum_assignment(-C)
    return int(C[r, c].sum())


_W = {}


def _init(payload):
    _W.update(payload)


def run_one(item):
    kind, w, t = item
    ev, _, rep = replay(_W['blocks'], params=params(w, t))
    lab = labels_at_rows(_W['rows'], ev)
    tp = idtp(lab, _W['truth'])
    return {'kind': kind, 'lam_l': w[0], 'lam_f': w[1], 'lam_c': w[2], 'T': t,
            'idf1': tp / len(lab), 'idtp': tp, 'rows': len(lab), 'relabels': len(ev),
            'near_ties': len(rep['near_tie_seqs'])}


def prepare():
    t0 = time.time()
    meta, qs, latest = load_labels()
    tid = truth_ids(meta, qs, latest)
    rows = window_rows(usecols=('frame', 'cam', 'uid', 'tsu', 'label')).reset_index(drop=True)
    truth = rows['seg'].map(tid).to_numpy().astype(np.int64)
    blocks = load_window_blocks()
    print('window: %d steps (%d live calls), %d rows, %d segments, %d truth vehicles, load %.0f s' % (
        len(blocks), sum(b['live'] is not None for b in blocks), len(rows), len(tid),
        truth.max() + 1, time.time() - t0))

    # 1. replay at the live code's values reproduces the recorded IDs
    t1 = time.time()
    ev0, _, rep0 = replay(blocks, verify=True)
    one = time.time() - t1
    lab0 = labels_at_rows(rows, ev0)
    bad = {k: rep0[k] for k in ('state_mismatch', 'call_mismatch', 'class_matrix_mismatch', 'match_mismatch')}
    print('check 1, live values: step mismatches %s, rows with another ID than recorded %d of %d' % (
        bad, int((lab0 != rows['label'].to_numpy()).sum()), len(rows)))
    # 2. the normalized form of the live setting gives exactly the same IDs
    ev1, _, _ = replay(blocks, params=params(CUR, T_CUR))
    lab1 = labels_at_rows(rows, ev1)
    print('check 2, normalized live setting: rows with another ID %d' % int((lab1 != lab0).sum()))
    # 3. scorer: truth scores 1; no hand-over at all (every track keeps its own ID) as a floor
    print('check 3, IDF1 of the truth itself %.4f, of no hand-over %.4f, of the live setting %.4f' % (
        idtp(truth, truth) / len(rows), idtp(rows['uid'].to_numpy(), truth) / len(rows),
        idtp(lab1, truth) / len(rows)))
    print('one replay %.2f s' % one)
    return {'blocks': blocks, 'rows': rows[['frame', 'uid']], 'truth': truth}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 2))
    args = ap.parse_args()
    payload = prepare()
    if args.check:
        return
    S = settings()
    print('settings: %d, workers %d' % (len(S), args.workers), flush=True)
    t0, out = time.time(), []
    with mp.Pool(args.workers, initializer=_init, initargs=(payload,)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, S, chunksize=4), 1):
            out.append(r)
            if i % 500 == 0 or i == len(S):
                print('  %d of %d done, %.0f s' % (i, len(S), time.time() - t0), flush=True)
    os.makedirs(RES, exist_ok=True)
    df = pd.DataFrame(out).sort_values(['kind', 'lam_l', 'lam_f', 'T']).reset_index(drop=True)
    df.to_csv(os.path.join(RES, 'grid_idf1.csv'), index=False)
    print('wrote results/grid_idf1.csv; near ties in any setting:', int(df['near_ties'].sum()))


if __name__ == '__main__':
    main()
