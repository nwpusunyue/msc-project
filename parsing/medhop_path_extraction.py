import json
import networkx as nx
import numpy as np
import pandas as pd
from parsing.special_tokens import *
from tqdm import tqdm


class EntityNotFoundException(Exception):
    pass


def get_paths(graph_info, source, target, cutoff=None):
    G = nx.from_numpy_matrix(np.array(graph_info[1]))

    if source not in graph_info[0] or target not in graph_info[0]:
        raise EntityNotFoundException('Source or target not in adjacency matrix')
    else:
        # TODO!: Figure a way of setting the cutoff point for path lengths.
        if cutoff is None:
            paths = nx.all_shortest_paths(G, graph_info[0][source], graph_info[0][target])
        else:
            paths = nx.all_simple_paths(G, graph_info[0][source], graph_info[0][target], cutoff=cutoff)
        paths = list(paths)

        entity_paths = []
        rel_paths = []
        for p in paths:
            entity_path = []
            rel_path = []
            for e1, e2 in zip(p[:-1], p[1:]):
                e1_str = graph_info[3][e1]
                e2_str = graph_info[3][e2]

                rel = graph_info[2][str((e1, e2))]
                entity_path.append(e1_str)
                # TODO!: Need to deal with multiple relations between the same entities!
                rel_path.append(rel[0])

            entity_path.append(e2_str)
            rel_path.append([END])

            entity_paths.append(entity_path)
            rel_paths.append(rel_path)

    return entity_paths, rel_paths


def extract_paths_dataset(df, cutoff=None):
    dataset = {
        'id': [],
        'source': [],
        'target': [],
        'relation': [],
        'entity_paths': [],
        'relation_paths': [],
        'label': []
    }
    for index, row in tqdm(df.iterrows()):
        question_id = row['id']
        target = row['target']
        relation = row['relation']
        graph = row['graph']

        answer = row['answer']
        non_answer = [c for c in row['candidates'] if c != answer]

        try:
            # positive example
            entity_paths, rel_paths = get_paths(graph, answer, target, cutoff=cutoff)

            dataset['id'].append(question_id)
            dataset['source'].append(answer)
            dataset['target'].append(target)
            dataset['relation'].append(relation)
            dataset['entity_paths'].append(entity_paths)
            dataset['relation_paths'].append(rel_paths)
            dataset['label'].append(1)
        except EntityNotFoundException as e:
            print(e)
            pass

    # negative examples
    for src in non_answer:
        try:
            entity_paths, rel_paths = get_paths(graph, src, target, cutoff=cutoff)

            dataset['id'].append(question_id)
            dataset['source'].append(answer)
            dataset['target'].append(target)
            dataset['relation'].append(relation)
            dataset['entity_paths'].append(entity_paths)
            dataset['relation_paths'].append(rel_paths)
            dataset['label'].append(0)
        except Exception as e:
            print(e)
            pass

    return dataset


if __name__ == "__main__":
    df = pd.read_json('train_with_graph.json')
    train_dataset = extract_paths_dataset(df[:1])
    with open("dummy_dataset.json", "w") as text_file:
        text_file.write(json.dumps(train_dataset))
