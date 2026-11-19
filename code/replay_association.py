"""Offline replay of the cross-camera ID handover.

The single-camera tracks are fixed by a recording of the live run; this re-runs only the
cross-camera matching, with any weights or thresholds. The recorded steps of the first
2 minutes are in data/validation_clip/steps.pkl (the recording of the full hour is not
included). With the live values the replay reproduces the recorded IDs exactly; grid.py
--check verifies this.
"""
import os
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# Values hard-coded in the live code:
#   deep_sort/new_matching_metric.py  associate_detections_to_trackers(): 0.4 / 0.3 / 0.3
#   deep_sort/new_matching_metric.py  new_match(): dist_threshold=2, oversize_threshold=10
#   deep_sort/nn_matching.py          class_distance(): 0 for the same class, 0.5 otherwise
LIVE_PARAMS = {'w_loc': 0.4, 'w_feat': 0.3, 'w_cls': 0.3,
               'cls_mismatch': 0.5, 'thr': 2.0, 'thr_oversize': 10.0}

# The live code computes appearance costs in float32, and their last digits
# depend on how many candidates of the other camera are in the call. A recorded
# cost is therefore bit-exact only for the candidate set it was computed with
# (and the same machine and MKL thread count). FEAT_EPS is a margin above the
# largest rounding difference measured, not a proven bound; --verify reports the
# largest difference actually seen in a recording and fails if it exceeds this.
FEAT_EPS = 1e-6


def load_blocks(record_dir):
    blocks = []
    with open(os.path.join(record_dir, 'blocks.pkl'), 'rb') as f:
        while True:
            try:
                blocks.append(pickle.load(f))
            except EOFError:
                break
    return blocks


def net_cam_of(b):
    """Camera ID of the other side of a step (older recordings lack 'net_cam';
    zone codes start with the camera ID, e.g. 194141 -> 194)."""
    return b['net_cam'] if 'net_cam' in b else b['zone_net'][1] // 1000


def _distance(a, b):
    """Same arithmetic as new_matching_metric.distance (rows: a, columns: b)."""
    b = np.expand_dims(b, 0)
    a = np.expand_dims(a, 1)
    x = np.subtract(a[..., 0], b[..., 0])
    y = np.subtract(a[..., 1], b[..., 1])
    return np.sqrt(np.power(x, 2) + np.power(y, 2))


def _appearance(b, ni, ci, at_live, prefer_live):
    """Appearance costs for the candidates (rows ni and columns ci of the
    recorded superset), and for each column whether it is bit-exact.

    The live code computes each column (one track of this camera) in its own
    call, against the stacked features of the other camera's candidates. So a
    column is bit-exact if it was recorded with this same set of other-camera
    candidates: the live matrix (the live candidates) or the superset matrix
    (every track in the zone).
    """
    net, cur, live, sup = b['net'], b['cur'], b['live'], b['C_feat_sup']
    if prefer_live and at_live and live is not None:
        return live['C_feat'], np.ones(len(ci), dtype=bool)
    if sup is None:
        raise ValueError('step %d has no superset appearance matrix' % b['seq'])
    C_feat = sup[np.ix_(ni, ci)]
    exact = np.full(len(ci), len(ni) == len(net['uid']))
    if prefer_live and live is not None:
        live_ni = np.flatnonzero(net['tsu'] < b['live_win'][1])
        if np.array_equal(ni, live_ni):
            live_ci = np.flatnonzero(cur['tsu'] < b['live_win'][0])
            in_live = np.isin(ci, live_ci)
            C_feat[:, in_live] = live['C_feat'][:, np.searchsorted(live_ci, ci[in_live])]
            exact |= in_live
    return C_feat, exact


def _near_tie(D, rows, cols, thr, col_exact, tol):
    """Could rounding of up to tol per cost change this step's decision? Yes if
    a chosen pair sits on its threshold or another assignment is about as cheap."""
    for r, c in zip(rows, cols):
        if not col_exact[c] and abs(D[r, c] - thr[c]) <= tol:
            return True
    if D.size == 1:
        return False
    # any other assignment leaves out at least one chosen pair
    best = D[rows, cols].sum()
    forbidden = np.abs(D).sum() + 1.0
    for r, c in zip(rows, cols):
        D2 = D.copy()
        D2[r, c] = forbidden
        r2, c2 = linear_sum_assignment(D2)
        if D[r2, c2].sum() - best <= 2 * len(rows) * tol:
            return True
    return False


def replay(blocks, params=None, windows=None, verify=False, prefer_live=True):
    """Re-run the cross-camera matching over the recorded steps, in order.

    params      : overrides for LIVE_PARAMS
    windows     : optional {(cam, other_cam): (w_cam, w_other)} replacing the
                  live time windows (a candidate's time_since_update must be
                  below the value, as in the live code)
    verify      : compare the steps with what the live run saw and decided.
                  This only counts and never changes the result; the counts
                  are meaningful only at the live settings
    prefer_live : use the live appearance matrix wherever it is bit-exact
                  (False is only for testing the superset matrix)

    Returns (label_events, class_events, report).
    label_events : (seq, frame, uid, new_label, partner_uid) in the order the
                   relabels happened. partner_uid is the other camera's track
                   that passed its label on; new_label is often not its uid,
                   since labels travel along chains of handovers.
    class_events : (seq, frame, uid, new_class, partner_uid)
    report       : counters. With time windows that change the other camera's
                   candidates, some costs are exact only up to float32
                   rounding: 'inexact_calls' counts those calls, and
                   'near_tie_seqs' lists the steps whose own decision rounding
                   up to FEAT_EPS could change. If it is empty, such rounding
                   cannot change the result. Otherwise any step after the
                   earliest listed one may differ, listed or not.
    """
    p = dict(LIVE_PARAMS, **(params or {}))
    label, cls = {}, {}
    label_events, class_events = [], []
    rep = {'steps': 0, 'calls': 0, 'relabels': 0,
           'state_mismatch': 0, 'call_mismatch': 0, 'feat_mismatch': 0,
           'class_matrix_mismatch': 0, 'match_mismatch': 0,
           'superset_feat_differs': 0, 'superset_feat_maxdiff': 0.0,
           'inexact_calls': 0, 'near_tie_seqs': [], 'first_problem': None}

    def problem(kind, b, detail=''):
        rep[kind] += 1
        if rep['first_problem'] is None:
            rep['first_problem'] = '%s at step %d (frame %d, cam %d vs %d) %s' % (
                kind, b['seq'], b['frame'], b['cam'], net_cam_of(b), detail)

    for b in blocks:
        rep['steps'] += 1
        cur, net, live = b['cur'], b['net'], b['live']

        # replay state of every candidate (label and class it carries now)
        for side in (cur, net):
            for i, u in enumerate(side['uid'].tolist()):
                if u not in label:
                    label[u] = u
                    cls[u] = side['init_cls'][i]
                if verify and (label[u] != side['label'][i] or cls[u] != side['cls'][i]):
                    problem('state_mismatch', b, 'uid %d: replay %s/%s, live %s/%s' % (
                        u, label[u], cls[u], side['label'][i], side['cls'][i]))

        # candidates inside the time windows
        live_win = tuple(b['live_win'])
        win = live_win if windows is None else tuple(windows.get((b['cam'], net_cam_of(b)), live_win))
        at_live = win == live_win
        ci = np.flatnonzero(cur['tsu'] < win[0])
        ni = np.flatnonzero(net['tsu'] < win[1])
        called = len(ci) > 0 and len(ni) > 0
        if verify and at_live and called != (live is not None):
            problem('call_mismatch', b)
        if not called:
            continue
        rep['calls'] += 1
        net_uid = net['uid'][ni].tolist()
        cur_uid = cur['uid'][ci].tolist()

        # appearance; verify also measures the superset matrix against the live one
        C_feat, col_exact = _appearance(b, ni, ci, at_live, prefer_live)
        if verify and at_live and live is not None:
            sup = b['C_feat_sup']
            sup_live = None if sup is None else sup[np.ix_(ni, ci)]
            if sup_live is None:
                problem('feat_mismatch', b, '(no superset matrix recorded)')
            elif sup_live.shape != live['C_feat'].shape:
                problem('feat_mismatch', b, 'shape %s vs %s' % (sup_live.shape, live['C_feat'].shape))
            else:
                diff = float(np.abs(sup_live - live['C_feat']).max())
                if diff > 0:
                    rep['superset_feat_differs'] += 1
                    rep['superset_feat_maxdiff'] = max(rep['superset_feat_maxdiff'], diff)

        # location and class, then the same cost and assignment as the live code
        dist = _distance(net['xy'][ni], cur['xy'][ci])
        net_cls = [cls[u] for u in net_uid]
        cur_cls = [cls[u] for u in cur_uid]
        C_class = np.zeros((len(ni), len(ci)))
        for r, a in enumerate(net_cls):
            for c, bcls in enumerate(cur_cls):
                if a != bcls:
                    C_class[r, c] = p['cls_mismatch']
        if verify and live is not None and not np.array_equal(C_class, live['C_class']):
            problem('class_matrix_mismatch', b)

        D = (p['w_loc'] * 1) * dist + (p['w_feat'] * 1) * C_feat + (p['w_cls'] * 1) * C_class
        rows, cols = linear_sum_assignment(D)
        thr = [p['thr_oversize'] if k == 'oversize' else p['thr'] for k in cur_cls]
        matches = []
        for r, c in zip(rows, cols):
            if not D[r, c] > thr[c]:         # live code rejects cost > threshold
                matches.append((net_uid[r], cur_uid[c]))

        # flag decisions that rounding of a not bit-exact cost could change
        if p['w_feat'] != 0 and not col_exact.all():
            rep['inexact_calls'] += 1
            if _near_tie(D, rows, cols, thr, col_exact, p['w_feat'] * FEAT_EPS):
                rep['near_tie_seqs'].append(b['seq'])

        if verify and live is not None:
            got = sorted((label[a], label[c]) for a, c in matches)
            exp = sorted(tuple(m) for m in live['matched'].tolist())
            if got != exp:
                problem('match_mismatch', b, 'replay %s, live %s' % (got, exp))

        # hand over IDs, unless this camera already uses that ID (live rule)
        taken = set(label.get(u, u) for u in b['all_conf_cur'].tolist())
        for nu, cu in matches:
            nl = label[nu]
            if nl in taken:
                continue
            label[cu] = nl
            label_events.append((b['seq'], b['frame'], cu, nl, nu))
            rep['relabels'] += 1
            if b['copy_class']:
                cls[cu] = cls[nu]
                class_events.append((b['seq'], b['frame'], cu, cls[nu], nu))
    return label_events, class_events, rep


def labels_at_rows(tracks, label_events):
    """ID each row of tracks carries under the replayed relabels, as an array
    aligned with the rows by position, whatever their order or index
    (a relabel during frame f is visible from frame f on)."""
    t = tracks[['frame', 'uid']].reset_index(drop=True).reset_index()
    if not label_events:
        return t.set_index('index')['uid'].sort_index().to_numpy()
    ev = pd.DataFrame([e[:4] for e in label_events], columns=['seq', 'frame', 'uid', 'new_label'])
    ev = (ev.sort_values(['frame', 'seq'])
            .drop_duplicates(['uid', 'frame'], keep='last')[['frame', 'uid', 'new_label']])
    t = t.sort_values('frame', kind='mergesort')
    m = pd.merge_asof(t, ev, on='frame', by='uid', direction='backward')
    lab = m['new_label'].fillna(m['uid']).astype(np.int64)
    return pd.Series(lab.to_numpy(), index=m['index'].to_numpy()).sort_index().to_numpy()
