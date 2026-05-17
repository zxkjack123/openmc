#!/usr/bin/env python3
"""Generate paper-quality figures for OpenMC DCU performance optimization.

Reads scaling_full.csv and baseline_v2.csv to produce:
1. Scaling curves (rate vs particle count)
2. Speedup vs CPU-Event
3. Transport time breakdown (stacked bar)
4. Optimization comparison (v1 vs v2)
"""

import csv
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Configuration ────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(OUT_DIR, exist_ok=True)
DPI = 300
FIG_W, FIG_H = 7.0, 4.5  # single-column width in inches

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
})

# ── Load data ────────────────────────────────────────────────────────
def load_csv(path):
    data = defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row['particles']), row['mode'])
            data[key].append(float(row['rate_active']))
    return data

here = os.path.dirname(os.path.abspath(__file__))
scaling = load_csv(os.path.join(here, 'scaling_full.csv'))

# Compute medians
def get_medians(data, mode, particles):
    return [statistics.median(data.get((p, mode), [0])) for p in particles]

particles = [1000, 5000, 10000, 50000, 100000, 200000, 500000, 1000000]
p_labels = ['1K', '5K', '10K', '50K', '100K', '200K', '500K', '1M']

cpu_hist = get_medians(scaling, 'cpu_history_64T', particles)
cpu_event = get_medians(scaling, 'cpu_event_64T', particles)
dcu_2gpu = get_medians(scaling, 'dcu_v2_2gpu', particles)
dcu_1gpu_full = get_medians(scaling, 'dcu_v2_1gpu', particles)
# DCU-1GPU has no 1M data
dcu_1gpu_particles = particles[:7]
dcu_1gpu = dcu_1gpu_full[:7]

# ── Figure 1: Scaling curves ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.plot(particles, cpu_hist, 'k-o', label='CPU History (64T)', zorder=3)
ax.plot(particles, cpu_event, 'b-s', label='CPU Event (64T)', zorder=3)
ax.plot(particles, dcu_2gpu, 'r-^', label='DCU-v2 2-GPU', zorder=3)
ax.plot(dcu_1gpu_particles, dcu_1gpu, 'g-d', label='DCU-v2 1-GPU', zorder=3)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Particles per Batch')
ax.set_ylabel('Calculation Rate (particles/s)')
ax.set_title('OpenMC Event-Mode Performance: CPU vs DCU')
ax.legend(loc='upper left')
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(800, 1200000)

# Custom x-tick labels
ax.set_xticks(particles)
ax.set_xticklabels(p_labels)
ax.xaxis.set_minor_formatter(ticker.NullFormatter())

out1 = os.path.join(OUT_DIR, 'fig1_scaling_curves.png')
fig.savefig(out1)
plt.close(fig)
print(f'  [1/4] {out1}')

# ── Figure 2: Speedup vs CPU-Event ──────────────────────────────────
speedup_2gpu = [d / c if c > 0 else 0 for d, c in zip(dcu_2gpu, cpu_event)]
speedup_1gpu = [d / c if c > 0 else 0 for d, c in zip(dcu_1gpu, cpu_event[:7])]

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.plot(particles, speedup_2gpu, 'r-^', label='DCU-v2 2-GPU / CPU-Event', linewidth=2)
ax.plot(dcu_1gpu_particles, speedup_1gpu, 'g-d', label='DCU-v2 1-GPU / CPU-Event', linewidth=2)
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Parity (1.0x)')

ax.set_xscale('log')
ax.set_xlabel('Particles per Batch')
ax.set_ylabel('Speedup vs CPU-Event (64T)')
ax.set_title('DCU Speedup Relative to CPU Event-Based Transport')
ax.legend(loc='best')
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(800, 1200000)
ax.set_ylim(0, 2.2)

ax.set_xticks(particles)
ax.set_xticklabels(p_labels)
ax.xaxis.set_minor_formatter(ticker.NullFormatter())

# Annotate peak
peak_idx = speedup_2gpu.index(max(speedup_2gpu))
ax.annotate(f'{speedup_2gpu[peak_idx]:.2f}x',
            xy=(particles[peak_idx], speedup_2gpu[peak_idx]),
            xytext=(particles[peak_idx]*2, speedup_2gpu[peak_idx]+0.15),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')

out2 = os.path.join(OUT_DIR, 'fig2_speedup_vs_cpu_event.png')
fig.savefig(out2)
plt.close(fig)
print(f'  [2/4] {out2}')

# ── Figure 3: XS time breakdown ─────────────────────────────────────
# Using scaling data XS times at representative particle counts
xs_data = defaultdict(dict)
with open(os.path.join(here, 'scaling_full.csv')) as f:
    reader = csv.DictReader(f)
    for row in reader:
        p = int(row['particles'])
        mode = row['mode']
        xs_val = row.get('xs', '')
        if xs_val and p == 50000:  # representative particle count
            key = (p, mode)
            if key not in xs_data:
                xs_data[key] = []
            xs_data[key].append(float(xs_val))

# For breakdown, use wall time - xs time as "other"
breakdown_modes = ['cpu_event_64T', 'dcu_v2_1gpu', 'dcu_v2_2gpu']
breakdown_labels = ['CPU-Event\n(64T)', 'DCU-v2\n1-GPU', 'DCU-v2\n2-GPU']
p_ref = 50000

xs_times = []
other_times = []
for mode in breakdown_modes:
    key = (p_ref, mode)
    rates = scaling.get(key, [0])
    xs_vals = xs_data.get(key, [0])
    xs_med = statistics.median(xs_vals) if xs_vals else 0

    # Compute total active transport time from rate
    # active_time = particles * active_batches / rate
    rate_med = statistics.median(rates) if rates else 1
    active_batches = 10  # from our benchmark config
    total_active = p_ref * active_batches / rate_med
    other = max(0, total_active - xs_med)

    xs_times.append(xs_med)
    other_times.append(other)

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
x = np.arange(len(breakdown_modes))
width = 0.5

bars_other = ax.bar(x, other_times, width, label='Other Transport', color='#4472C4')
bars_xs = ax.bar(x, xs_times, width, bottom=other_times, label='XS Lookup (GPU)', color='#ED7D31')

ax.set_xlabel('Execution Mode')
ax.set_ylabel('Active Transport Time (s)')
ax.set_title(f'Transport Time Breakdown at {p_ref//1000}K Particles')
ax.set_xticks(x)
ax.set_xticklabels(breakdown_labels)
ax.legend()
ax.grid(True, axis='y', alpha=0.3)

# Add value labels
for i, (xs, oth) in enumerate(zip(xs_times, other_times)):
    total = xs + oth
    ax.text(i, total + 2, f'{total:.0f}s', ha='center', fontsize=9)
    if xs > 5:
        ax.text(i, oth + xs/2, f'XS: {xs:.0f}s', ha='center', va='center', fontsize=8, color='white')

out3 = os.path.join(OUT_DIR, 'fig3_transport_breakdown.png')
fig.savefig(out3)
plt.close(fig)
print(f'  [3/4] {out3}')

# ── Figure 4: 2-GPU scaling efficiency ──────────────────────────────
scaling_ratio = [d2 / d1 if d1 > 0 else 0
                 for d2, d1 in zip(dcu_2gpu[:7], dcu_1gpu)]

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.bar(range(7), scaling_ratio, color='#5B9BD5', edgecolor='black', linewidth=0.5)
ax.axhline(y=2.0, color='red', linestyle='--', linewidth=1, label='Ideal 2x')
ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1)

ax.set_xlabel('Particles per Batch')
ax.set_ylabel('2-GPU / 1-GPU Rate Ratio')
ax.set_title('Dual-DCU Parallel Scaling Efficiency')
ax.set_xticks(range(7))
ax.set_xticklabels(p_labels[:7])
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(0, 3.2)

# Add value labels
for i, v in enumerate(scaling_ratio):
    ax.text(i, v + 0.05, f'{v:.2f}x', ha='center', fontsize=9)

out4 = os.path.join(OUT_DIR, 'fig4_2gpu_scaling.png')
fig.savefig(out4)
plt.close(fig)
print(f'  [4/4] {out4}')

print(f'\nAll figures saved to {OUT_DIR}/')
