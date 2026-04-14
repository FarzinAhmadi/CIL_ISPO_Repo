"""
Regenerate all three new figures with clean formatting:
 - Value labels placed above error-bar caps (not overlapping)
 - Consistent font sizes and spacing
 - Tighter layouts
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUT = "/sessions/trusting-cool-ramanujan/mnt/Operations Research Letters SPO plus paper/output"

# ── Shared style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

def annotate_bar(ax, bar, mean, se, fmt='{:.1f}%', pad_frac=0.06, fontsize=8.5):
    """Place label just above the error-bar cap, never overlapping it."""
    top = mean + se          # top of error bar
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    y_label = top + pad_frac * y_range
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        y_label,
        fmt.format(mean),
        ha='center', va='bottom',
        fontsize=fontsize, clip_on=False
    )

# ══════════════════════════════════════════════════════════════════════════════
# 1. fig_ispo_warmstart  (bar chart, 4 bars)
# ══════════════════════════════════════════════════════════════════════════════
means_ws  = [26.73, 52.40, 53.19, 51.51]
sds_ws    = [9.69,   8.61,  8.14,  8.48]
n_trials  = 8
ses_ws    = [s / np.sqrt(n_trials) for s in sds_ws]

labels_ws = ['CIL-MSE\n(Linear)', 'CIL-ISPO+\n(WarmStart)',
             'CIL-ISPO+\n(NoWarm)',  'CIL-ISPO+\n(Curriculum)']
colors_ws = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

fig, ax = plt.subplots(figsize=(6.5, 4.2))
bars = ax.bar(labels_ws, means_ws, yerr=ses_ws,
              capsize=5, color=colors_ws, alpha=0.88,
              edgecolor='k', linewidth=0.7,
              error_kw=dict(elinewidth=1.2, ecolor='#333333', capthick=1.2))

# Set y-limit with headroom for labels
ax.set_ylim(0, max(m + s for m, s in zip(means_ws, ses_ws)) * 1.30)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
ax.set_ylabel('Mean SPO Loss (%)')
ax.set_title('Effect of Initialization: ISPO+ Variants')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar, m, se in zip(bars, means_ws, ses_ws):
    annotate_bar(ax, bar, m, se, pad_frac=0.04)

plt.tight_layout()
fig.savefig(f'{OUT}/fig_ispo_warmstart.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig_ispo_warmstart.png', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fig_ispo_warmstart')

# ══════════════════════════════════════════════════════════════════════════════
# 2. fig_hypothesis_class  (line chart, 3 series)
# ══════════════════════════════════════════════════════════════════════════════
K_vals = [50, 200, 500]
data_hyp = {
    'Linear': {'means': [36.14, 32.75, 32.43], 'sds': [9.45, 9.75, 9.37]},
    'RF':     {'means': [52.37, 35.31, 27.28], 'sds': [10.54, 7.73, 6.91]},
    'NN':     {'means': [34.06, 28.82, 17.52], 'sds': [7.01,  6.35, 6.33]},
}
n_h = 8
styles_hyp = {
    'Linear': dict(fmt='o-',  color='#4C72B0', label='Linear'),
    'RF':     dict(fmt='s--', color='#DD8452', label='RF'),
    'NN':     dict(fmt='^:',  color='#2ca02c', label='NN'),
}

fig, ax = plt.subplots(figsize=(6.0, 4.0))
for name, d in data_hyp.items():
    ses = [s / np.sqrt(n_h) for s in d['sds']]
    s = styles_hyp[name]
    ax.errorbar(K_vals, d['means'], yerr=ses,
                fmt=s['fmt'], color=s['color'], label=s['label'],
                capsize=4, linewidth=1.8, markersize=7,
                markerfacecolor=s['color'], markeredgecolor='white',
                markeredgewidth=0.5)

ax.set_xlabel('Training set size $K$')
ax.set_ylabel('Mean SPO Loss (%)')
ax.set_title('CIL-MSE: Hypothesis Class Comparison')
ax.set_xscale('log')
ax.set_xticks(K_vals)
ax.set_xticklabels([str(k) for k in K_vals])
ax.set_ylim(10, 62)
ax.legend(loc='upper right', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.6)

# Annotate endpoints for readability
for name, d in data_hyp.items():
    col = styles_hyp[name]['color']
    # label the K=500 endpoint
    ax.annotate(f"{d['means'][-1]:.1f}%",
                xy=(500, d['means'][-1]),
                xytext=(8, 0), textcoords='offset points',
                va='center', fontsize=8, color=col)

plt.tight_layout()
fig.savefig(f'{OUT}/fig_hypothesis_class.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig_hypothesis_class.png', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fig_hypothesis_class')

# ══════════════════════════════════════════════════════════════════════════════
# 3. fig_shortest_path  (2-panel bar chart)
# ══════════════════════════════════════════════════════════════════════════════
data_sp = {
    0.1: {'IO+LS (Linear)':    (34.00, 7.86),
          'CIL-MSE (RF)':      (30.65, 6.35),
          'Uniform baseline':  (41.29, 8.23)},
    0.5: {'IO+LS (Linear)':    (39.57, 7.30),
          'CIL-MSE (RF)':      (35.88, 5.73),
          'Uniform baseline':  (41.61, 7.04)},
}
methods_sp = ['IO+LS (Linear)', 'CIL-MSE (RF)', 'Uniform baseline']
colors_sp  = ['#4C72B0',        '#55A868',       '#999999']
n_sp = 8

fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=False)
for ax, sigma in zip(axes, [0.1, 0.5]):
    d = data_sp[sigma]
    ms  = [d[m][0] for m in methods_sp]
    ses = [d[m][1] / np.sqrt(n_sp) for m in methods_sp]

    xpos = np.arange(len(methods_sp))
    bars = ax.bar(xpos, ms, yerr=ses,
                  capsize=5, color=colors_sp, alpha=0.88,
                  edgecolor='k', linewidth=0.7,
                  error_kw=dict(elinewidth=1.2, ecolor='#333333', capthick=1.2))

    ax.set_xticks(xpos)
    ax.set_xticklabels(methods_sp, fontsize=8.5)
    ax.set_title(f'Shortest Path  ($\\sigma = {sigma}$)')
    ax.set_ylabel('Mean SPO Loss (%)')
    ax.set_ylim(0, max(m + s for m, s in zip(ms, ses)) * 1.32)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, m, se in zip(bars, ms, ses):
        annotate_bar(ax, bar, m, se, pad_frac=0.04)

fig.suptitle('Shortest-Path Case Study: 5×5 Directed Grid', fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(f'{OUT}/fig_shortest_path.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig_shortest_path.png', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fig_shortest_path')

print('\n✓ All figures regenerated.')
