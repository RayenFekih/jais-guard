"""Reproduce JaisGuard figures: python plot_benchmarks.py metrics.json --out figures"""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

COLORS = {'Untuned Jais 3 0.9B': '#B8B4AC', 'Frozen Backbone + Head': '#6489AA',
          'LoRA + Head': '#D77B5D', 'Jev 1.14.0': '#DCD9D0'}
ORDER = list(COLORS)
LABELS = ['Untuned Jais 3 0.9B', 'Frozen + Head', 'LoRA + Head', 'Jev 1.14.0']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14,
                     'text.color': '#202020', 'axes.labelcolor': '#202020',
                     'xtick.color': '#202020', 'ytick.color': '#202020',
                     'svg.fonttype': 'path', 'savefig.facecolor': 'white'})

def figure(title, subtitle, labels, values, texts, colors, ylabel, ymax, step, caption, output, callout=None):
    fig = plt.figure(figsize=(12.8, 7.4), facecolor='white')
    ax = fig.add_axes([.105, .245, .85, .52])
    fig.text(.53, .925, title, ha='center', fontsize=28, weight='medium')
    fig.text(.53, .867, subtitle, ha='center', fontsize=17, fontfamily='DejaVu Serif', color='#50504C')
    bars = ax.bar(range(len(values)), values, width=.68, color=colors, zorder=3)
    ax.set_ylim(0, ymax)
    ax.set_xlim(-.7, len(values)-.3)
    ax.set_xticks(range(len(values)), labels, fontsize=14)
    ax.tick_params(axis='x', length=0, pad=16)
    ax.tick_params(axis='y', length=0, pad=9, labelsize=12)
    ax.set_ylabel(ylabel, fontsize=15, labelpad=17)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.grid(axis='y', color='#DCDCD5', linewidth=.85, zorder=0)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#292929')
        ax.spines[side].set_linewidth(1.25)
    for bar, label in zip(bars, texts):
        ax.annotate(label, (bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0, 9), textcoords='offset points', ha='center', va='bottom',
                    fontsize=12 if len(label)>10 else 15)
    if 'LoRA + Head' in labels:
        ax.get_xticklabels()[labels.index('LoRA + Head')].set_weight('bold')
    if callout:
        ax.text(.035, .87, callout, transform=ax.transAxes, fontsize=20, color='#B96348', weight='medium')
    fig.text(.105, .085, caption, fontsize=11.5, color='#5A5A55', linespacing=1.65, va='bottom')
    fig.savefig(output.with_suffix('.png'), dpi=300)
    fig.savefig(output.with_suffix('.svg'))
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('metrics', type=Path)
    parser.add_argument('--out', type=Path, default=Path('figures'))
    args = parser.parse_args()
    raw = args.metrics.read_bytes()
    data = json.loads(raw)
    args.out.mkdir(parents=True, exist_ok=True)
    quality = {r['model']: r for r in data['quality']['rows']}
    latency = {r['model']: r for r in data['latency']['rows']}
    figure('Arabic Gibberish Detection', 'Macro F1 on 400 historical evaluation examples',
           LABELS, [quality[m]['macro_f1'] for m in ORDER],
           [str(quality[m]['macro_f1']) for m in ORDER], [COLORS[m] for m in ORDER],
           'Macro F1 (%)', 105, 20,
           '188 VALID / 212 GIBBERISH. Historical, previously examined inputs; not a blind test.\nSame inputs across approaches; inference protocols differ.', args.out/'classification_quality')
    figure('Inference Latency', 'Mean latency per input, lower is better', LABELS,
           [latency[m]['mean_ms'] for m in ORDER], [f"{latency[m]['mean_ms']:.2f}" for m in ORDER],
           [COLORS[m] for m in ORDER], 'Mean latency (ms)', 250, 50,
           'Jais models: local RTX 4050 Laptop GPU, batch 1, no HTTP. Jev: hosted API, including network.\nDifferent deployment conditions; not a controlled same-hardware comparison.', args.out/'inference_latency')
    serving = data['serving']
    rows = serving['rows']
    figure('Optimized GPU Serving', 'FP16, concurrency 16, RTX 4050 Laptop GPU',
           [r['backend'] for r in rows], [r['requests_per_second'] for r in rows],
           [f"{r['requests_per_second']:.2f}" for r in rows], ['#6489AA', '#D77B5D'],
           'Requests per second', 280, 50,
           f"400 warm HTTP requests/backend. Peak allocated GPU memory: {rows[0]['peak_gpu_allocated_mib']:.2f} → {rows[1]['peak_gpu_allocated_mib']:.2f} MiB.\nHistorical full-backbone exports; HF serialized, vLLM batched. Allocator peak, not total board usage.",
           args.out/'serving_throughput', f"{serving['throughput_speedup']:.2f}× throughput")
    notes = ['# JaisGuard benchmark provenance',
             f"Source: `{args.metrics.name}`. SHA-256: `{hashlib.sha256(raw).hexdigest()}`.",
             'All bar heights use JSON values directly. F1 labels retain all supplied digits; latency and throughput retain two decimal places. All axes start at zero. No uncertainty estimates were supplied, so no error bars are plotted.',
             '## Classification quality', data['quality']['evaluation']]
    notes += [f"- {m}: Macro F1 {quality[m]['macro_f1']}%; {quality[m]['protocol']}." for m in ORDER]
    notes += ['## Inference latency', data['latency']['evaluation']]
    notes += [f"- {m}: {latency[m]['mean_ms']:.2f} ms; {latency[m]['mode']}." for m in ORDER]
    notes += ['## Optimized serving throughput', serving['evaluation']]
    notes += [f"- {r['backend']}: {r['requests_per_second']:.2f} requests/s; peak allocated GPU memory {r['peak_gpu_allocated_mib']:.2f} MiB." for r in rows]
    notes += [f"Reported throughput improvement: {serving['throughput_speedup']:.2f}×, as supplied in JSON. The ratio of the displayed rounded rates is approximately {rows[1]['requests_per_second']/rows[0]['requests_per_second']:.5f}×.", serving['limitations'],
              'Quality/latency and serving experiments differ in precision and batching. Serving results do not describe native vLLM compatibility of the compact published HF packages.',
              '## Reproduction', 'Python 3.12; Matplotlib 3.10.8. Install: `python -m pip install matplotlib==3.10.8`.',
              'Run: `python plot_benchmarks.py jaisguard_readme_metrics.json --out figures`.',
              'PNG: 3840 × 2220 pixels, 300 dpi. SVG: vector paths, including text, for portable typography.']
    (args.out/'metrics_provenance.md').write_text('\n\n'.join(notes)+'\n')

if __name__ == '__main__':
    main()
