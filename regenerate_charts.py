#!/usr/bin/env python3
"""Regenerate benchmark 04 charts in ATOM-style (white background, clean theme).
Matches: blue/green/orange palette, bold titles with em dash, bold labels, 1480x730.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
OUT_DIR = os.path.join(RESULTS_DIR, 'cross_comparison')

# Full 500-step runs
RUNS = {
    'LoRA': os.path.join(RESULTS_DIR, 'gx10_lora_20260405_213220', 'benchmark_metrics.csv'),
    'QLoRA': os.path.join(RESULTS_DIR, 'gx10_qlora_20260404_001959', 'benchmark_metrics.csv'),
    'Full Fine-Tune': os.path.join(RESULTS_DIR, 'gx10_fullft_20260404_225659', 'benchmark_metrics.csv'),
}

COLORS = {
    'LoRA': '#4A90D9',        # blue
    'QLoRA': '#4CAF50',       # green
    'Full Fine-Tune': '#FF9800',  # orange
}

MARKERS = {
    'LoRA': 'o',
    'QLoRA': 's',
    'Full Fine-Tune': 'D',
}

def load_csv(path):
    steps, loss, step_time, gpu_mem = [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            loss.append(float(row['loss']))
            step_time.append(float(row['step_time_sec']))
            gpu_mem.append(float(row['gpu_memory_mb']) / 1024)  # Convert to GB
    return steps, loss, step_time, gpu_mem

def setup_style():
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': '#cccccc',
        'axes.edgecolor': '#cccccc',
        'font.family': 'sans-serif',
        'font.size': 14,
        'axes.titlesize': 20,
        'axes.titleweight': 'bold',
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

setup_style()

# Load all data
data = {}
for name, path in RUNS.items():
    data[name] = load_csv(path)

# --- Chart 1: Training Loss Curves ---
fig, ax = plt.subplots(figsize=(14.8, 7.3))
for name in ['LoRA', 'QLoRA', 'Full Fine-Tune']:
    steps, loss, _, _ = data[name]
    # Subsample for cleaner plot (every 5th point + markers every 50)
    ax.plot(steps, loss, color=COLORS[name], linewidth=2, alpha=0.8, label=name)
    marker_idx = list(range(0, len(steps), 50))
    ax.plot([steps[i] for i in marker_idx], [loss[i] for i in marker_idx],
            color=COLORS[name], marker=MARKERS[name], linestyle='none',
            markersize=8, markeredgecolor='white', markeredgewidth=1.5)

ax.set_xlabel('Training Step')
ax.set_ylabel('Loss')
ax.set_title('Training Loss Curves \u2014 Llama 3.1 8B Instruct')
ax.legend(fontsize=13, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'loss_curves.png'), dpi=100, bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, 'loss_curves.svg'), bbox_inches='tight')
plt.close()
print('Saved loss_curves.png/svg')

# --- Chart 2: GPU Memory Usage (bar chart) ---
fig, ax = plt.subplots(figsize=(14.8, 7.3))
names = ['LoRA', 'QLoRA', 'Full Fine-Tune']
peak_mem = [max(data[n][3]) for n in names]  # gpu_mem is index 3
bar_colors = [COLORS[n] for n in names]

bars = ax.bar(names, peak_mem, color=bar_colors, width=0.5, edgecolor='white', linewidth=1.5)
ax.axhline(y=122, color='#FF4444', linestyle='--', linewidth=2, label='Total GPU Memory (122 GB)')

for bar, val in zip(bars, peak_mem):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val:.1f} GB', ha='center', va='bottom', fontsize=16, fontweight='bold')

ax.set_ylabel('GPU Memory (GB)')
ax.set_title('Peak GPU Memory Usage \u2014 Llama 3.1 8B Instruct')
ax.set_ylim(0, 140)
ax.legend(loc='upper left', fontsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'gpu_memory.png'), dpi=100, bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, 'gpu_memory.svg'), bbox_inches='tight')
plt.close()
print('Saved gpu_memory.png/svg')

# --- Chart 3: Step Time ---
fig, ax = plt.subplots(figsize=(14.8, 7.3))
for name in ['LoRA', 'QLoRA', 'Full Fine-Tune']:
    steps, _, step_time, _ = data[name]
    avg_time = np.mean(step_time[3:])  # skip warmup
    ax.plot(steps, step_time, color=COLORS[name], linewidth=2, alpha=0.8,
            label=f'{name} ({avg_time:.1f}s avg)', marker=MARKERS[name],
            markevery=50, markersize=8, markeredgecolor='white', markeredgewidth=1.5)

ax.set_xlabel('Training Step')
ax.set_ylabel('Step Time (seconds)')
ax.set_title('Training Step Time \u2014 Llama 3.1 8B Instruct')
ax.legend(fontsize=13, loc='upper right')
ax.set_ylim(0, max(max(data[n][2]) for n in names) * 1.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'step_time.png'), dpi=100, bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, 'step_time.svg'), bbox_inches='tight')
plt.close()
print('Saved step_time.png/svg')

print('\nAll charts regenerated in ATOM style.')
