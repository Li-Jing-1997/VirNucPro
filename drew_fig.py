import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import sys
import os

infile = sys.argv[1]
directory = os.path.dirname(os.path.abspath(infile))

out_fig = os.path.join(directory, 'prediction.png')

node_data = {}

with open(infile, 'r') as file:
    for line in file:
        if 'Sequence_ID' in line:
            continue
        parts = line.strip().split('\t')
        sequence_id = parts[0]
        score1 = float(parts[2])
        score2 = float(parts[3])
        prediction = parts[1]

        node = sequence_id.split('_chunk_')[0]
        chunk = sequence_id.split('_chunk_')[1][:-2] if '_chunk_' in sequence_id else None  # Remove last two characters
        max_score = max(score1, score2)

        if node in node_data:

            node_data[node]['chunks'].append(chunk)
            node_data[node]['max_scores'].append(max_score)
            node_data[node]['predictions'].append(prediction)
        else:
            node_data[node] = {
                'chunks': [chunk],
                'max_scores': [max_score],
                'predictions': [prediction]
            }

unique_nodes = list(node_data.keys())

if len(unique_nodes) > 5:
    unique_nodes = unique_nodes[:5]

num_nodes = len(unique_nodes)
fig, axes = plt.subplots(nrows=num_nodes, ncols=1, figsize=(20, 2 * num_nodes))

if num_nodes == 1:
    axes = [axes]

for ax, node in zip(axes, unique_nodes):
    chunks = node_data[node]['chunks']
    max_scores = node_data[node]['max_scores']
    predictions = node_data[node]['predictions']

    chunk_numbers = []
    for chunk in chunks:
        try:
            chunk_num = int(chunk)
            chunk_numbers.append(chunk_num)
        except (IndexError, ValueError):
            print(f"Warning: Unexpected chunk format '{chunk}'. Skipping...")
            continue

    chunk_scaled = [chunk * 500 for chunk in chunk_numbers]

    colors = ['red' if pred == 'virus' else 'darkblue' for pred in predictions]

    for x, y, color in zip(chunk_scaled, max_scores, colors):
        ax.plot([x, x + 500], [y, y], color=color, linewidth=2)

    ax.set_title(f'{node}', fontsize=16, fontweight='bold', color='black')
    ax.set_xlabel('Length', fontsize=14, color='darkgray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.tick_params(colors='gray', which='both')

    ax.grid(False)

    ax.set_ylim(0.45, 1.05)

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False, useMathText=False))
    ax.ticklabel_format(style='plain', axis='x')

plt.tight_layout()

plt.savefig(out_fig, bbox_inches='tight')
plt.close()