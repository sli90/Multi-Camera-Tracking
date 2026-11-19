"""Figure of the parameter study.

(a) IDF1 for every weight combination (lam_l + lam_f + lam_c = 1) at the selected threshold;
    one hexagonal cell per grid setting, dashed line = 1 point below the best, star = selected.
(b) IDF1 versus the threshold T_dis at the selected weights.

Reads results/grid_idf1.csv; writes results/fig_sensitivity.png (600 dpi)
and fig_sensitivity.pdf (vector). Colors: one-hue blue ramp for magnitude (light = low).
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                        # noqa: E402
import matplotlib.tri as mtri                          # noqa: E402
from matplotlib import font_manager                    # noqa: E402
from matplotlib.collections import PolyCollection      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize   # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402
from matplotlib.patches import Polygon                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), 'results')

BLUE_RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
             '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
SERIES, ACCENT = '#2a78d6', '#eb6834'
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
H = np.sqrt(3) / 2
STEP = 0.05


def xy(l, f, c):
    """Weights -> plane: location corner bottom-left, appearance bottom-right, class on top."""
    return np.asarray(f) + 0.5 * np.asarray(c), H * np.asarray(c)


def hexagon(cx, cy, s):
    """Cell of one setting: the hexagon around a point of a triangular grid with spacing s."""
    r = s / np.sqrt(3)
    a = np.deg2rad(90 + 60 * np.arange(6))
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]


def setup_fonts():
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        'font.family': 'Arial' if 'Arial' in names else 'DejaVu Sans',
        'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
        'mathtext.default': 'regular'})


def panel_a(fig, ax, cax, A, cur, best):
    z = 100 * A['idf1'].to_numpy()
    px, py = xy(A['lam_l'], A['lam_f'], A['lam_c'])
    norm = Normalize(vmin=np.floor(z.min()), vmax=np.ceil(z.max()))
    cmap = LinearSegmentedColormap.from_list('blue', BLUE_RAMP)
    tri = Polygon([(0, 0), (1, 0), (0.5, H)], closed=True, transform=ax.transData)
    cells = PolyCollection([hexagon(x, y, STEP) for x, y in zip(px, py)], array=z, cmap=cmap,
                           norm=norm, edgecolors='white', linewidths=0.4)
    cells.set_clip_path(tri)
    ax.add_collection(cells)
    ax.add_patch(Polygon([(0, 0), (1, 0), (0.5, H)], closed=True, fill=False, edgecolor=AXIS, lw=0.8))
    ax.tricontour(mtri.Triangulation(px, py), z, levels=[100 * best - 1.0], colors=INK,
                  linewidths=0.9, linestyles=[(0, (3, 2))])
    sx, sy = xy(cur['lam_l'], cur['lam_f'], cur['lam_c'])
    ax.plot([sx], [sy], marker='*', ms=10, color=ACCENT, mec='white', mew=0.7, zorder=5)

    # ticks on the three edges (values 0.2 .. 0.8)
    for v in (0.2, 0.4, 0.6, 0.8):
        for (x, y), (nx, ny), ha, va in (
                (xy(1 - v, v, 0), (0, -1), 'center', 'top'),            # bottom: lam_f
                (xy(0, 1 - v, v), (H, 0.5), 'left', 'center'),          # right: lam_c
                (xy(v, 0, 1 - v), (-H, 0.5), 'right', 'center')):       # left: lam_l
            ax.plot([x, x + 0.02 * nx], [y, y + 0.02 * ny], color=AXIS, lw=0.6)
            ax.text(x + 0.035 * nx, y + 0.035 * ny, '%.1f' % v, ha=ha, va=va, fontsize=6.5, color=INK2)
    ax.text(0.5, -0.115, r'$\lambda_f$', ha='center', va='top', fontsize=8, color=INK)
    ax.text(0.86, 0.47, r'$\lambda_c$', ha='left', va='center', fontsize=8, color=INK)
    ax.text(0.14, 0.47, r'$\lambda_l$', ha='right', va='center', fontsize=8, color=INK)
    ax.text(-0.03, -0.04, 'Location', ha='right', va='top', fontsize=7.5, color=INK)
    ax.text(1.03, -0.04, 'Appearance', ha='left', va='top', fontsize=7.5, color=INK)
    ax.text(0.5, H + 0.035, 'Class', ha='center', va='bottom', fontsize=7.5, color=INK)
    ax.text(-0.17, H + 0.10, r'$T_{dis}$ = %.2f' % cur['T'], ha='left', va='top', fontsize=7.5, color=INK2)
    handles = [Line2D([], [], marker='*', ls='none', ms=8, color=ACCENT, mec='white', label='selected setting'),
               Line2D([], [], color=INK, lw=0.9, ls=(0, (3, 2)), label='1 point below the best')]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.17, 1.08), frameon=False,
              handlelength=2.2, borderaxespad=0)
    ax.set_xlim(-0.24, 1.2)
    ax.set_ylim(-0.15, H + 0.10)
    ax.set_aspect('equal')
    ax.axis('off')
    cb = fig.colorbar(cells, cax=cax)
    cb.outline.set_visible(False)
    cb.set_label('IDF1 (%)', color=INK)
    cb.ax.tick_params(length=2, width=0.5, colors=INK2)


def panel_b(ax, C, cur, best):
    t, v = C['T'].to_numpy(), 100 * C['idf1'].to_numpy()
    ax.plot(t, v, color=SERIES, lw=1.5, marker='o', ms=3.2, mfc=SERIES, mec='white', mew=0.5, zorder=3)
    ax.axvline(cur['T'], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.axhline(100 * best, color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=1)
    ax.plot([cur['T']], [100 * cur['idf1']], marker='*', ms=10, color=ACCENT, mec='white', mew=0.7, zorder=4)
    ax.text(cur['T'] + 0.03, 72.0, r'selected $T_{dis}$ = %.2f' % cur['T'], ha='left', va='bottom',
            fontsize=7, color=INK2)
    ax.text(1.52, 100 * best + 0.4, 'best in grid, %.1f%%' % (100 * best), ha='right', va='bottom',
            fontsize=7, color=INK2)
    ax.set_xlim(0, 1.55)
    ax.set_ylim(70, 95)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    ax.set_yticks([70, 75, 80, 85, 90, 95])
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(length=2.5, width=0.5)
    ax.set_xlabel(r'Distance threshold $T_{dis}$')
    ax.set_ylabel('IDF1 (%)')


def main():
    setup_fonts()
    df = pd.read_csv(os.path.join(RES, 'grid_idf1.csv'))
    A = df[df['kind'] == 'at_current_T']
    C = df[df['kind'] == 'current_weights'].sort_values('T')
    cur = df[df['kind'] == 'current'].iloc[0]
    best = df[df['kind'] == 'grid']['idf1'].max()

    fig = plt.figure(figsize=(174 / 25.4, 72 / 25.4))
    axa = fig.add_axes([0.01, 0.06, 0.45, 0.86])
    cax = fig.add_axes([0.465, 0.17, 0.012, 0.62])
    axb = fig.add_axes([0.60, 0.17, 0.385, 0.76])
    panel_a(fig, axa, cax, A, cur, best)
    panel_b(axb, C, cur, best)
    fig.text(0.01, 0.97, '(a)', fontsize=9, fontweight='bold', va='top', color=INK)
    fig.text(0.54, 0.97, '(b)', fontsize=9, fontweight='bold', va='top', color=INK)
    for ext, kw in (('png', {'dpi': 600}), ('pdf', {})):
        fig.savefig(os.path.join(RES, 'fig_sensitivity.' + ext), facecolor='white', **kw)
    print('wrote', os.path.join(RES, 'fig_sensitivity.png'), 'and .pdf; font', plt.rcParams['font.family'],
          '; matplotlib', matplotlib.__version__)


if __name__ == '__main__':
    main()
