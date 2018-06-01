import networkx as nx
import numpy as np
import operator
import pandas as pd
import re
from itertools import combinations
from nltk.tokenize import sent_tokenize
from nltk.tokenize import WordPunctTokenizer

word_tokenizer = WordPunctTokenizer()
target_regex = re.compile('^interacts_with (.+)\?$')
biomed_entity = re.compile("[ABCDEFGHIKMOPQS][0-9,A-Z]+[0-9]+")
ENT_1 = 'ent_1'
ENT_2 = 'ent_2'
ENT_X = 'ent_x'


def extract_target(query):
    return target_regex.match(query).group(1)


def extract_entities(sentence):
    unique_entities = {}
    entity_idxs = []

    tokenized_sentence = word_tokenizer.tokenize(sentence)
    for i, t in enumerate(tokenized_sentence):
        if biomed_entity.match(t):
            entity_idxs.append(i)
            if t not in unique_entities:
                unique_entities[t] = [i]
            else:
                unique_entities[t].append(i)
    for entity_idx in entity_idxs:
        tokenized_sentence[entity_idx] = ENT_X
    # return list of (entity, [list_of_occurences_indices]) tuples sorted by the index of the first occurence
    return (tokenized_sentence, sorted([(k, v) for k, v in unique_entities.items()], key=lambda x: x[0][1]))


def extract_medhop_instances(documents):
    instances = []
    for d in documents:
        documents_instances = list(filter(lambda x: len(x[1]) > 1,
                                          [extract_entities(s) for s in sent_tokenize(d)]))
        instances += documents_instances
    return instances


def replace_entities(tokens, list_ent1, list_ent2):
    tokens_cpy = tokens.copy()
    for idx in list_ent1:
        tokens_cpy[idx] = ENT_1
    for idx in list_ent2:
        tokens_cpy[idx] = ENT_2
    return tokens_cpy


def extract_binary_medhop_instances(medhop_instances):
    binary_instances = []
    for instance in medhop_instances:
        pairs = list(combinations(instance[1], 2))

        binary_instances += [(replace_entities(instance[0], pair[0][1], pair[1][1]),
                              pair) for pair in pairs]
    return binary_instances


def extract_graph_instances(binary_medhop_instances):
    relations = {}
    entity_id_mapping = {}
    current_id = 0
    for instance in binary_medhop_instances:
        ent1 = instance[1][0][0]
        ent2 = instance[1][1][0]
        relation = instance[0]
        if ent1 not in entity_id_mapping:
            entity_id_mapping[ent1] = current_id
            current_id += 1
        if ent2 not in entity_id_mapping:
            entity_id_mapping[ent2] = current_id
            current_id += 1
        if (entity_id_mapping[ent1], entity_id_mapping[ent2]) not in relations:
            relations[(entity_id_mapping[ent1], entity_id_mapping[ent2])] = [relation]
            ### adding the relation the other way around as well. To see if that's necessary.
            relations[(entity_id_mapping[ent2], entity_id_mapping[ent1])] = [relation]
        else:
            relations[(entity_id_mapping[ent1], entity_id_mapping[ent2])].append(relation)
            ### adding the relation the other way around as well. To see if that's necessary.
            relations[(entity_id_mapping[ent2], entity_id_mapping[ent1])].append(relation)
    # create adjacency matrix with entity_ids
    adj_matrix = np.zeros((len(entity_id_mapping), len(entity_id_mapping)))
    for (ent1_id, ent2_id) in relations.keys():
        adj_matrix[ent1_id, ent2_id] = 1.0
    labels = [i[0] for i in sorted(entity_id_mapping.items(), key=operator.itemgetter(1))]
    return (entity_id_mapping, adj_matrix, relations, labels)


def get_paths(graph_info, sources, target, cutoff=8):
    if type(sources) is not list:
        sources = [sources]

    G = nx.from_numpy_matrix(graph_info[1])
    all_paths = []
    for source in sources:
        if source not in graph_info[0] or target not in graph_info[0]:
            print('Source or target not in adjacency matrix')
        else:
            # TODO!: Figure a way of setting the cutoff point for path lengths.
            paths = nx.all_simple_paths(G, graph_info[0][source], graph_info[0][target], cutoff=cutoff)
            paths = list(paths)
            entity_rel_paths = []
            for p in paths:
                entity_rel_path = []
                for e1, e2 in zip(p[:-1], p[1:]):
                    e1_str = graph_info[3][e1]
                    e2_str = graph_info[3][e2]
                    rel = graph_info[2][(e1, e2)]
                    # TODO!: Need to deal with multiple relations between the same entities!
                    entity_rel_path.append((e1_str, rel[0]))
                entity_rel_path.append((e2_str, []))
                entity_rel_paths.append(entity_rel_path)
            all_paths += entity_rel_paths
    return all_paths


def preprocess_medhop(filename):
    df = pd.read_json(filename, orient='records')
    df['target'] = df['query'].apply(extract_target)
    df['medhop_instances'] = df['supports'].apply(extract_medhop_instances)
    print('Extracted medhop instances')
    df['binary_medhop_instances'] = df['medhop_instances'].apply(extract_binary_medhop_instances)
    print('Extracted binary medhop instances')
    df['graph'] = df['binary_medhop_instances'].apply(extract_graph_instances)
    print('Extracted graph data')
    df['positive_paths'] = df.apply(lambda row: get_paths(row['graph'], row['answer'], row['target']), axis=1)
    print('Extracted positive paths')
    df['negative_paths'] = df.apply(lambda row: get_paths(row['graph'], row['candidates'], row['target']), axis=1)
    print('Extracted negative paths')
    return df


if __name__ == "__main__":
    df = preprocess_medhop('./qangaroo_v1.1/medhop/train.json')
    print(df['graph'][0])
