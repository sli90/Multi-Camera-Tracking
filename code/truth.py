"""Ground truth from the labels of the 2-minute validation clip, and a check of the labels
for contradictions.

Segments: one track's rows, cut where two consecutive rows are >= 8 frames apart; key
'uid_firstframe'. Every segment with rows in the window (frames 1..1800) starts as its own
vehicle. A match answer 'k' joins Q's segment with candidate box k's segment, and a gap
answer 'same' joins the segments before and after the gap (union-find). 'none' and
'different' join nothing.

Contradictions (reported only; nothing is changed):
  none_linked       a match question answered 'none', but other answers put Q together
                    with one of its candidate boxes
  different_linked  a gap check answered 'different', but other answers put A and B together
  same_time         two segments of one camera, both detected in the same frame, end up as
                    one vehicle (can be right, e.g. two boxes on one long truck)

Reads data/validation_clip/{questions.json, answers.csv, tracks.csv};
writes results/truth_segments.csv and results/contradictions.csv.
"""
import collections
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'validation_clip')
RES = os.path.join(ROOT, 'results')
F_FIRST, F_LAST = 1, 1800      # tracks frame numbers; video frame index = frame - 1
SPLIT = 8                      # gaps >= SPLIT frames end a segment
SEGM = 1000                    # segment id = uid * SEGM + k
FPS = 15


def segments(T):
    Ts = T.sort_values(['uid', 'frame'], kind='mergesort').reset_index(drop=True)
    u = Ts['uid'].to_numpy()
    f = Ts['frame'].to_numpy()
    newu = np.r_[True, u[1:] != u[:-1]]
    d = np.r_[0, f[1:] - f[:-1]]
    c = np.cumsum((~newu) & (d >= SPLIT))
    starts = np.flatnonzero(newu)
    k = c - np.repeat(c[starts], np.diff(np.r_[starts, len(u)]))
    Ts['sid'] = u.astype(np.int64) * SEGM + k
    g = Ts.groupby('sid', sort=True)
    SEG = pd.DataFrame({'uid': g['uid'].first(), 'cam': g['cam'].first(),
                        'first': g['frame'].min(), 'last': g['frame'].max()}).reset_index()
    gi = np.flatnonzero((~newu) & (d >= SPLIT))
    GAPS = pd.DataFrame({'uid': u[gi], 'cam': Ts['cam'].to_numpy()[gi], 'f_before': f[gi - 1],
                         'f_after': f[gi], 'gap': d[gi], 'sid_before': Ts['sid'].to_numpy()[gi - 1],
                         'sid_after': Ts['sid'].to_numpy()[gi]})
    return Ts, SEG, GAPS


def video_time(frame):
    """Tracks frame f is video index f-1; returns m:ss.s from the clip start."""
    s = (frame - 1) / FPS
    return '%d:%04.1f' % (int(s // 60), s % 60)


def window_rows(usecols=('frame', 'cam', 'uid', 'tsu')):
    """Rows of tracks.csv inside the window, each with its segment key ('uid_firstframe')."""
    T = pd.read_csv(os.path.join(DATA, 'tracks.csv'), usecols=list(usecols))
    T = T[(T['frame'] >= F_FIRST) & (T['frame'] <= F_LAST)].copy()
    Ts, SEG, _ = segments(T)
    first = SEG.set_index('sid')['first']
    Ts['seg'] = (Ts['sid'] // SEGM).astype(str) + '_' + Ts['sid'].map(first).astype(str)
    return Ts


def load_labels():
    with open(os.path.join(DATA, 'questions.json'), encoding='utf-8') as fh:
        Q = json.load(fh)
    A = pd.read_csv(os.path.join(DATA, 'answers.csv'), dtype={'answer': str})
    latest = {r.qid: {'answer': r.answer} for r in A.itertuples(index=False)}
    return Q['meta'], Q['questions'], latest


class DSU:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_truth(meta, qs, latest):
    """Returns (dsu, joins, apart): joins = [(seg_a, seg_b, n)] from the answers,
    apart = [(seg_a, seg_b, n, kind)] pairs the answers say are different vehicles."""
    dsu = DSU(list(meta['segments_in_window']))
    joins, apart = [], []
    for q in qs:
        a = latest[q['qid']]['answer']
        p0, p1 = q['panels']
        left = p0['boxes'][0]['seg']
        if q['type'] == 'match':
            cands = [b['seg'] for b in p1['boxes']]
            if a == 'none':
                apart += [(left, c, q['n'], 'none_linked') for c in cands]
            else:
                joins.append((left, cands[int(a) - 1], q['n']))
        else:
            right = p1['boxes'][0]['seg']
            if a == 'same':
                joins.append((left, right, q['n']))
            else:
                apart.append((left, right, q['n'], 'different_linked'))
    for a, b, _ in joins:
        dsu.union(a, b)
    return dsu, joins, apart


def path_items(joins, s, t):
    """Question numbers of the answers that join s to t (shortest chain)."""
    adj = collections.defaultdict(list)
    for a, b, n in joins:
        adj[a].append((b, n))
        adj[b].append((a, n))
    prev, todo = {s: None}, collections.deque([s])
    while todo:
        x = todo.popleft()
        if x == t:
            break
        for y, n in adj[x]:
            if y not in prev:
                prev[y] = (x, n)
                todo.append(y)
    items, x = [], t
    while prev.get(x) is not None:
        x, n = prev[x]
        items.append(n)
    return items[::-1]


def truth_ids(meta, qs, latest):
    """Truth vehicle number for every segment key in the window."""
    dsu, _, _ = build_truth(meta, qs, latest)
    roots = {}
    return {s: roots.setdefault(dsu.find(s), len(roots)) for s in meta['segments_in_window']}


def main():
    meta, qs, latest = load_labels()
    assert all(q['qid'] in latest for q in qs), 'some questions have no answer'
    byn = {q['n']: q for q in qs}
    dsu, joins, apart = build_truth(meta, qs, latest)

    Ts = window_rows()
    segs = set(meta['segments_in_window'])
    assert set(Ts['seg']) == segs, 'segments differ from the question set'
    info = Ts.groupby('seg').agg(cam=('cam', 'first'), first=('frame', 'min'),
                                 last=('frame', 'max'), rows=('frame', 'size'))
    det = Ts[Ts['tsu'] == 0].groupby('seg')['frame'].apply(set).to_dict()
    roots = {}
    tid = {s: roots.setdefault(dsu.find(s), len(roots)) for s in sorted(segs, key=lambda s: (info.loc[s, 'first'], s))}

    out = []
    for a, b, n, kind in apart:
        if dsu.find(a) == dsu.find(b):
            out.append({'kind': kind, 'items': [n] + path_items(joins, a, b), 'detail': '%s and %s' % (a, b)})
    members = collections.defaultdict(list)
    for s in segs:
        members[dsu.find(s)].append(s)
    for ss in members.values():
        by_cam = collections.defaultdict(list)
        for s in ss:
            by_cam[info.loc[s, 'cam']].append(s)
        for cam, cs in by_cam.items():
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    both = det.get(cs[i], set()) & det.get(cs[j], set())
                    if both:
                        out.append({'kind': 'same_time', 'items': path_items(joins, cs[i], cs[j]),
                                    'detail': 'cam %d: %s and %s both detected in %d frames from %s' % (
                                        cam, cs[i], cs[j], len(both), video_time(min(both)))})

    os.makedirs(RES, exist_ok=True)
    seg_tab = info.copy()
    seg_tab['truth_id'] = [tid[s] for s in seg_tab.index]
    seg_tab.sort_values(['truth_id', 'first']).to_csv(os.path.join(RES, 'truth_segments.csv'))
    rows = [{'contradiction': k, 'kind': c['kind'], 'detail': c['detail'], 'n': n, 'type': byn[n]['type'],
             'answer': latest[byn[n]['qid']]['answer']} for k, c in enumerate(out, 1) for n in c['items']]
    pd.DataFrame(rows, columns=['contradiction', 'kind', 'detail', 'n', 'type', 'answer']).to_csv(
        os.path.join(RES, 'contradictions.csv'), index=False)

    cams = collections.defaultdict(set)
    for s, t in tid.items():
        cams[t].add(info.loc[s, 'cam'])
    print('segments %d, rows %d, joins %d, apart pairs %d' % (len(segs), len(Ts), len(joins), len(apart)))
    print('truth vehicles %d: seen in 1 camera %d, 2 cameras %d, 3 %d, 4 %d' % (
        len(cams), *[sum(len(v) == k for v in cams.values()) for k in (1, 2, 3, 4)]))
    print('contradictions: %d' % len(out), dict(collections.Counter(c['kind'] for c in out)))


if __name__ == '__main__':
    main()
