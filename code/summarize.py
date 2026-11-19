"""Key numbers from results/grid_idf1.csv.
Writes results/summary.json."""
import os
import json

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), 'results')


def fmt(r):
    d = r._asdict() if hasattr(r, '_asdict') else dict(r)      # itertuples row or Series
    return '%.3f/%.3f/%.3f T %.3f: IDF1 %.2f%%' % (d['lam_l'], d['lam_f'], d['lam_c'], d['T'], 100 * d['idf1'])


def main():
    df = pd.read_csv(os.path.join(RES, 'grid_idf1.csv'))
    G = df[df.kind == 'grid']
    cur = df[df.kind == 'current'].iloc[0]
    tab = df[df.kind == 'table1'].iloc[0]
    best = G.idf1.max()
    top = G[G.idf1 >= best - 1e-12].sort_values(['T', 'lam_l'])
    print('grid: %d settings; best IDF1 %.2f%%, reached by %d settings:' % (len(G), 100 * best, len(top)))
    for r in top.head(12).itertuples():
        print('   ', fmt(r))
    print('code setting     ', fmt(cur), '-> %.2f points below the best' % (100 * (best - cur.idf1)))
    print('paper Table 1    ', fmt(tab), '-> %.2f points below the best' % (100 * (best - tab.idf1)))
    out = {'best_idf1': best, 'n_best': len(top), 'current_idf1': cur.idf1, 'table1_idf1': tab.idf1}

    for d in (0.5, 1, 2):
        m = G.idf1 >= best - d / 100
        print('grid settings within %.1f points of the best: %d (%.1f%%)' % (d, m.sum(), 100 * m.mean()))

    perT = G.groupby('T').idf1.max()
    print('best IDF1 at each threshold (over all 231 weight combinations):')
    print('   ' + '  '.join('%.2f:%.1f' % (t, 100 * v) for t, v in perT.items()))

    A = df[df.kind == 'at_current_T']
    print('at the code threshold T = %.3f (231 weight combinations): best %.2f%%, worst %.2f%%' % (
        cur['T'], 100 * A.idf1.max(), 100 * A.idf1.min()))
    for d in (0.5, 1, 2):
        m1 = A.idf1 >= A.idf1.max() - d / 100
        m2 = A.idf1 >= cur.idf1 - d / 100
        print('   within %.1f points of that best: %.1f%%; within %.1f points of the code setting or above: %.1f%%' % (
            d, 100 * m1.mean(), d, 100 * m2.mean()))
        out['plateau_share_%g' % d] = m1.mean()

    P = df[df.kind == 'current_pm0.1']
    print('one weight +-0.1 around the code setting (another weight -+0.1), T fixed:')
    for r in P.itertuples():
        print('   %s  (%+.2f points)' % (fmt(r), 100 * (r.idf1 - cur.idf1)))
    out['pm01_max_change'] = float((P.idf1 - cur.idf1).abs().max())

    C = df[df.kind == 'current_weights'].sort_values('T')
    print('threshold curve at the code weights:')
    print('   ' + '  '.join('%.2f:%.1f' % (t, 100 * v) for t, v in zip(C['T'], C.idf1)))
    out['t_curve'] = {'%.2f' % t: v for t, v in zip(C['T'], C.idf1)}

    variants = [('location only', G.lam_l == 1), ('appearance only', G.lam_f == 1),
                ('class only', G.lam_c == 1), ('no class', G.lam_c == 0),
                ('no appearance', G.lam_f == 0), ('no location', G.lam_l == 0)]
    print('cue variants (best over the weights and thresholds allowed):')
    for name, m in variants:
        r = G[m].sort_values('idf1', ascending=False).iloc[0]
        print('   %-16s %s' % (name, fmt(r)))
        out['variant_' + name.replace(' ', '_')] = r.idf1
    print('near ties in any setting:', int(df.near_ties.sum()))
    with open(os.path.join(RES, 'summary.json'), 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
