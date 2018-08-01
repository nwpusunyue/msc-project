import json

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from parsing.path_search import (
    all_paths,
    shortest_paths,
    shortest_paths_plus_threshold,
)

path_search_method = {
    'shortest': shortest_paths,
    'shortest_plus': shortest_paths_plus_threshold,
    'all': all_paths
}

END_IDX = -1


class EntityNotFoundException(Exception):
    pass


def get_paths(graph_info, source, target, path_search_method_name, cutoff=None, limit=None):
    G = nx.from_numpy_matrix(np.array(graph_info[1]))

    if source not in graph_info[0] or target not in graph_info[0]:
        raise EntityNotFoundException('Source or target not in adjacency matrix')
    else:
        paths = path_search_method[path_search_method_name](G,
                                                            graph_info[0][source],
                                                            graph_info[0][target],
                                                            cutoff=cutoff
                                                            )
        paths = list(paths)
        if limit is not None and limit < len(paths):
            idxs = np.random.choice(np.arange(0, len(paths)), size=limit, replace=False)
        else:
            idxs = np.arange(0, len(paths))

        entity_paths = []
        rel_paths = []
        for idx in idxs:
            p = paths[idx]
            entity_path = []
            rel_path = []
            for e1, e2 in zip(p[:-1], p[1:]):
                e1_str = graph_info[3][e1]
                e2_str = graph_info[3][e2]

                rel = graph_info[2][(e1, e2)]
                entity_path.append(e1_str)
                # TODO!: Need to deal with multiple relations between the same entities!
                rel_path.append(rel[0])
            entity_path.append(e2_str)
            rel_path.append(END_IDX)
            entity_paths.append(entity_path)
            rel_paths.append(rel_path)

    return entity_paths, rel_paths


def extract_paths_dataset(df, path_search_method_name, cutoff=None, limit=None):
    dataset = {
        'id': [],
        'source': [],
        'target': [],
        'relation': [],
        'entity_paths': [],
        'relation_paths': []
    }
    if 'answer' in df.columns:
        dataset['label'] = []

    positive_ex_cnt = 0
    negative_ex_cnt = 0
    total = 0
    for index, row in tqdm(df.iterrows()):
        question_id = row['id']
        target = row['target']
        relation = row['relation']
        graph = row['graph']

        if 'answer' in row.axes[0].tolist():
            answer = row['answer']
            non_answer = [c for c in row['candidates'] if c != answer]

            try:
                total += 1
                # positive example
                entity_paths, rel_paths = get_paths(graph,
                                                    answer,
                                                    target,
                                                    path_search_method_name=path_search_method_name,
                                                    cutoff=cutoff,
                                                    limit=limit)

                dataset['id'].append(question_id)
                dataset['source'].append(answer)
                dataset['target'].append(target)
                dataset['relation'].append(relation)
                dataset['entity_paths'].append(entity_paths)
                dataset['relation_paths'].append(rel_paths)
                dataset['label'].append(1)
            except (nx.NetworkXNoPath, EntityNotFoundException) as e:
                print(type(e))
                positive_ex_cnt += 1
                pass

            # negative examples
            for src in non_answer:
                total += 1
                try:
                    entity_paths, rel_paths = get_paths(graph,
                                                        src,
                                                        target,
                                                        cutoff=cutoff,
                                                        limit=limit,
                                                        path_search_method_name=path_search_method_name)

                    dataset['id'].append(question_id)
                    dataset['source'].append(src)
                    dataset['target'].append(target)
                    dataset['relation'].append(relation)
                    dataset['entity_paths'].append(entity_paths)
                    dataset['relation_paths'].append(rel_paths)
                    dataset['label'].append(0)
                except Exception as e:
                    print(type(e))
                    negative_ex_cnt += 1
                    pass
        else:
            candidates = row['candidates']
            for c in candidates:
                total += 1
                try:
                    entity_paths, rel_paths = get_paths(graph,
                                                        c,
                                                        target,
                                                        cutoff=cutoff,
                                                        limit=limit,
                                                        path_search_method_name=path_search_method_name)

                    dataset['id'].append(question_id)
                    dataset['source'].append(c)
                    dataset['target'].append(target)
                    dataset['relation'].append(relation)
                    dataset['entity_paths'].append(entity_paths)
                    dataset['relation_paths'].append(rel_paths)
                except Exception as e:
                    print(type(e))
                    pass
    print('Total examples: {}\n'
          'Total with path: {}'.format(total,
                                       len(dataset['id'])))

    return dataset


def extract_source_target_dataset(df):
    dataset = {
        'id': [],
        'source': [],
        'target': [],
        'relation': [],
        'entity_paths': [],
        'relation_paths': []
    }
    if 'answer' in df.columns:
        dataset['label'] = []

    for index, row in tqdm(df.iterrows()):
        question_id = row['id']
        target = row['target']
        relation = row['relation']

        if 'answer' in row.axes[0].tolist():
            answer = row['answer']
            non_answer = [c for c in row['candidates'] if c != answer]

            try:
                # positive example
                entity_paths = [[answer, target]]
                rel_paths = [[-1, -1]]

                dataset['id'].append(question_id)
                dataset['source'].append(answer)
                dataset['target'].append(target)
                dataset['relation'].append(relation)
                dataset['entity_paths'].append(entity_paths)
                dataset['relation_paths'].append(rel_paths)
                dataset['label'].append(1)
            except (nx.NetworkXNoPath, EntityNotFoundException) as e:
                print(type(e))
                pass
            for src in non_answer:
                try:
                    # negative examples
                    entity_paths = [[src, target]]
                    rel_paths = [[-1, -1]]

                    dataset['id'].append(question_id)
                    dataset['source'].append(src)
                    dataset['target'].append(target)
                    dataset['relation'].append(relation)
                    dataset['entity_paths'].append(entity_paths)
                    dataset['relation_paths'].append(rel_paths)
                    dataset['label'].append(0)
                except Exception as e:
                    print(type(e))
                    pass
        else:
            candidates = row['candidates']
            for c in candidates:
                try:
                    entity_paths = [[c, target]]
                    rel_paths = [[-1]]

                    dataset['id'].append(question_id)
                    dataset['source'].append(c)
                    dataset['target'].append(target)
                    dataset['relation'].append(relation)
                    dataset['entity_paths'].append(entity_paths)
                    dataset['relation_paths'].append(rel_paths)
                except Exception as e:
                    print(type(e))
                    pass
    return dataset


if __name__ == "__main__":
    df = pd.read_json('train_with_graph.json')
    train_dataset = extract_paths_dataset(df, path_search_method_name='shortest_plus', cutoff=1)
    with open("dummy_dataset.json", "w") as text_file:
        text_file.write(json.dumps(train_dataset))
