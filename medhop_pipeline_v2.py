import networkx as nx
import numpy as np
import operator
import pandas as pd
import pickle
import re
from itertools import chain
from itertools import combinations
from nltk.tokenize import sent_tokenize
from nltk.tokenize import WordPunctTokenizer
from tqdm import tqdm

word_tokenizer = WordPunctTokenizer()
target_regex = re.compile('^interacts_with (.+)\?$')
biomed_entity = re.compile("[ABCDEFGHIKMOPQS][0-9,A-Z]+[0-9]+")
ENT_1 = 'ent_1'
ENT_2 = 'ent_2'
ENT_X = 'ent_x'


def extract_target_and_relation(df):
    regex = re.compile('^(.+) (.+)\?$')

    df['target'] = df['query'].apply(lambda q: regex.match(q).group(2))
    df['relation'] = df['query'].apply(lambda q: regex.match(q).group(1))


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


def extract_medhop_instances(document):
    document_instances = list(filter(lambda x: len(x[1]) > 1,
                                     [extract_entities(s) for s in sent_tokenize(document)]))
    return document_instances


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


def extract_document_binary_medhop_instances(documents):
    binary_medhop_instances = []
    for d in tqdm(documents):
        document_medhop_instances = extract_medhop_instances(d)
        document_binary_medhop_instances = extract_binary_medhop_instances(document_medhop_instances)
        binary_medhop_instances.append(document_binary_medhop_instances)
    return binary_medhop_instances


def extract_query_binary_medhop_instances(supports, documents, binary_medhop_instances):
    query_binary_medhop_instances = []
    for s in supports:
        idx = documents.index(s)
        query_binary_medhop_instances += binary_medhop_instances[idx]
    return query_binary_medhop_instances


def extract_graph(df, support_data_path=None):
    if support_data_path is not None:
        support_data = pickle.load(open(support_data_path, 'rb'))
        supports = support_data[0]
        binary_medhop_instances = support_data[1]
    else:
        # extract all unique supports
        supports = list(set(chain.from_iterable(df['supports'])))
        binary_medhop_instances = extract_document_binary_medhop_instances(supports)

    df['graph'] = df.apply(lambda row: _extract_graph(row['supports'],
                                                      supports,
                                                      binary_medhop_instances),
                           axis=1)


def _extract_graph(supports, documents, binary_medhop_instances):
    query_binary_medhop_instances = extract_query_binary_medhop_instances(supports,
                                                                          documents,
                                                                          binary_medhop_instances)

    relations = {}
    entity_id_mapping = {}
    current_id = 0
    for instance in query_binary_medhop_instances:
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


def extract_target_and_relation(df):
    regex = re.compile('^(.+) (.+)\?$')

    df['target'] = df['query'].apply(lambda q: regex.match(q).group(2))
    df['relation'] = df['query'].apply(lambda q: regex.match(q).group(1))


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


def preprocess_medhop(df):
    extract_graph(df)
    print('Extracted graph data')
    extract_target_and_relation(df)
    print('Extracted target and relation')
    return df


if __name__ == "__main__":
    df = pd.read_json('./qangaroo_v1.1/medhop/train.json', orient='records')
    df = preprocess_medhop(df)
    df.to_json('train_with_graph.json')
