import pandas as pd
import networkx as nx
import numpy as np
import json
from sklearn.model_selection import train_test_split
import random

### KOTOR DATASET (use .head(AMOUNT) to limit the scope of the network for debugging purposeses)
# df = pd.read_csv('KOTOR_dataset.csv').head(3000)
df = pd.read_csv('KOTOR_dataset.csv')
max_length = 500   # Maximum number of dialogue turns in an interaction

G = nx.DiGraph()
for index, row in df.iterrows():
    G.add_node(row['id'], text=row['text'], speaker=row['speaker'], listener=row.get('listener', ''), 
               animation=row.get('animation', ''), comment=row.get('comment', ''))
    if pd.notna(row['next']):
        next_ids = eval(row['next'])
        for next_id in next_ids:
            if next_id in df['id'].values:
                G.add_edge(row['id'], next_id)


def create_sequences(graph, max_length):
    sequences = []
    for node in list(graph.nodes):
        sequence = []
        edges = list(nx.dfs_edges(graph, source=node, depth_limit=max_length - 1))
        if edges:
            nodes = [node] + [v for u, v in edges]
            sequence = [
                {
                    'id': graph.nodes[n].get('id'),
                    'speaker': graph.nodes[n].get('speaker'),
                    'listener': graph.nodes[n].get('listener'),
                    'text': graph.nodes[n].get('text'),
                    'animation': graph.nodes[n].get('animation'),
                    'comment': graph.nodes[n].get('comment')
                }
                for n in nodes if 'text' in graph.nodes[n]
            ]
        if len(sequence) > 0:
            sequences.append(sequence)
        if len(sequence) >= max_length:
            break
    return sequences

def create_masked_examples(sequences):
    masked_examples = []
    for sequence in sequences:
        if len(sequence) > 2:
            mask_index = random.randint(1, len(sequence) - 2)       # DIALOGUE INDEX: CHOOSE RANDOMLY IN THE MIDDLE
            
            sequence_copy = sequence[:mask_index] + sequence[mask_index + 1:]
            masked_text = sequence[mask_index]['text']
            masked_speaker = sequence[mask_index]['speaker']
            masked_entry = f"{masked_speaker}: <MASK>"   # Replace masked speaker's dialogue with <MASK> token
            # Linear sequence has the speaker, text, animation, and developer context
            input_sequence = "\n".join([f"{item['speaker']}: {item['text']} (Animation: {item['animation']}, Comment: {item['comment']})" for item in sequence[:mask_index]]) + "\n" + masked_entry + "\n" + "\n".join([f"{item['speaker']}: {item['text']} (Animation: {item['animation']}, Comment: {item['comment']})" for item in sequence[mask_index+1:]])
            example = {
                "input": input_sequence,
                "target": masked_text
            }
            masked_examples.append(example)
    return masked_examples


sequences = create_sequences(G, max_length)
masked_examples = create_masked_examples(sequences)

with open('masked_examples_large.json', 'w') as f:
    json.dump(masked_examples, f, indent=4)
