"""Regenerate the portable learning figure: python plot.py (requires matplotlib)."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
read_lines = lambda name: [json.loads(line) for line in (folder / name).read_text().splitlines()]
summary = json.loads((folder / 'summary.json').read_text())
updates, probes = read_lines('updates.jsonl'), read_lines('probes.jsonl')
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': '#334b47', 'text.color': '#19352f',
                     'axes.titleweight': 'semibold', 'figure.facecolor': '#f7f8f3',
                     'axes.facecolor': '#ffffff', 'savefig.facecolor': '#f7f8f3'})
fig, axs = plt.subplots(2, 2, figsize=(11, 7.6))
fig.suptitle('First inspectable training experiment', x=.07, ha='left', fontsize=20, fontweight='bold')
fig.text(.07, .925, '12 self-play games · 64 MCTS simulations/move · 200-ply cap · 22 optimizer updates', fontsize=11)
colors = ['#147665', '#4c67ad', '#be5c35', '#94629f']
counts = [summary['decisive_games'], summary['actual_draws'], summary['move_limit_games']]
ax = axs[0, 0]
bars=ax.bar(['Checkmate', 'Rule-defined draw', 'Move limit'], counts, color=[colors[0], colors[1], colors[2]], width=.55)
ax.bar_label(bars, padding=4)
ax.set_ylim(0, max(counts)+2)
ax.set_ylabel('Games')
ax.set_title('How games actually ended', loc='left')
steps=[row['step'] for row in updates]
ax=axs[0, 1]
ax.plot(steps, [row['policy_kl'] for row in updates], color=colors[0], lw=2)
ax.set_title('Policy fit to search targets', loc='left')
ax.set_xlabel('Optimizer step')
ax.set_ylabel('KL divergence, full action space (nats)')
ax.grid(alpha=.2)
ax=axs[1, 0]
ax.plot(steps, [row['value_loss'] for row in updates], color=colors[2], lw=2, label='Network')
ax.plot(steps, [row['target_win_fraction']+row['target_loss_fraction'] for row in updates], color='#71807a', lw=1.8, linestyle='--', label='Always predict zero')
ax.set_title('Value error on changing training batches', loc='left')
ax.set_xlabel('Optimizer step')
ax.set_ylabel('Mean squared error')
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=.2)
ax=axs[1, 1]
labels={'white_capture_queen':'White: capture queen', 'black_capture_queen':'Black: capture queen',
        'white_mate_in_one':'White: mate in one', 'black_mate_in_one':'Black: mate in one'}
for i,(identifier,label) in enumerate(labels.items()):
    values=[(snapshot['game'],next(row for row in snapshot['positions'] if row['id']==identifier)['policy_probability_expected']*100) for snapshot in probes]
    ax.plot([v[0] for v in values],[v[1] for v in values], marker='o', markersize=4, color=colors[i], label=label, lw=1.6)
ax.set_title('Raw policy on the same tactical positions', loc='left')
ax.set_xlabel('Completed self-play games')
ax.set_ylabel('Probability on expected move(s), %')
ax.set_ylim(0, 34)
ax.set_xticks([snapshot['game'] for snapshot in probes])
ax.legend(fontsize=8, frameon=False, ncols=2, loc='upper right')
ax.grid(alpha=.2)
fig.tight_layout(rect=[.04,.09,.99,.90], h_pad=2, w_pad=3)
fig.text(.07,.035, 'Cutoffs still receive zero value targets. Batch losses use changing samples; these diagnostics do not establish stronger play.', fontsize=9, color='#5c7066')
fig.savefig(folder / 'learning-summary.png', dpi=160)
fig.savefig(folder / 'learning-summary.svg')
print(folder / 'learning-summary.png')
