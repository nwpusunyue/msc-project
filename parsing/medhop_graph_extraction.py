import numpy as np
import operator
import pandas as pd
import re
from itertools import chain
from itertools import combinations
from nltk.tokenize import sent_tokenize
from nltk.tokenize import WordPunctTokenizer
from parsing.special_tokens import *
from parsing.document_store import DocumentStore
from tqdm import tqdm

word_tokenizer = WordPunctTokenizer()
biomed_entity = re.compile("[ABCDEFGHIKMOPQS][0-9,A-Z]+[0-9]+")


def read_biomed_entity_list(filename):
    with open(filename, 'r') as f:
        content = f.readlines()
    biomed_entities = [x.strip() for x in content]
    return biomed_entities


def extract_target_and_relation(df):
    regex = re.compile('^(.+) (.+)\?$')

    df['target'] = df['query'].apply(lambda q: regex.match(q).group(2))
    df['relation'] = df['query'].apply(lambda q: regex.match(q).group(1))


def extract_entities(text, biomed_entities=None):
    unique_entities = {}
    entity_idxs = []

    tokenized_text = word_tokenizer.tokenize(text)
    for i, t in enumerate(tokenized_text):
        if (biomed_entities is None and t in biomed_entities) or biomed_entity.match(t):
            entity_idxs.append(i)
            if t not in unique_entities:
                unique_entities[t] = [i]
            else:
                unique_entities[t].append(i)
    for entity_idx in entity_idxs:
        tokenized_text[entity_idx] = ENT_X
    # return list of (entity, [list_of_occurences_indices]) tuples sorted by the index of the first occurence
    return (tokenized_text, sorted([(k, v) for k, v in unique_entities.items()], key=lambda x: x[0][1]))


def extract_medhop_instances(document, biomed_entities=None, sentence_wise=True):
    if sentence_wise:
        document_instances = list(filter(lambda x: len(x[1]) > 1,
                                         [extract_entities(s, biomed_entities) for s in sent_tokenize(document)]))
    else:
        document_instances = [extract_entities(document, biomed_entities)]
    return document_instances


def extract_binary_medhop_instances(doc_idx, medhop_instances):
    binary_instances = []
    for instance in medhop_instances:
        pairs = list(combinations(instance[1], 2))

        binary_instances += [(doc_idx,
                              (pair[0], pair[1])) for pair in pairs]
        binary_instances += [(doc_idx,
                              (pair[1], pair[0])) for pair in pairs]

    return binary_instances


def extract_document_binary_medhop_instances(documents, sentence_wise=True, entity_list_path=None):
    binary_medhop_instances = []
    medhop_instances = []
    biomed_entities = None

    if entity_list_path is not None:
        biomed_entities = read_biomed_entity_list(entity_list_path)
        print('{} possible entities'.format(len(biomed_entities)))

    for d_idx, d in tqdm(enumerate(documents)):
        document_medhop_instances = extract_medhop_instances(d, biomed_entities, sentence_wise)
        medhop_instances.append(document_medhop_instances)

        document_binary_medhop_instances = extract_binary_medhop_instances(d_idx, document_medhop_instances)
        binary_medhop_instances.append(document_binary_medhop_instances)

    document_store = DocumentStore(medhop_instances)
    return binary_medhop_instances, document_store


def extract_query_binary_medhop_instances(supports, documents, binary_medhop_instances):
    query_binary_medhop_instances = []
    for s in supports:
        idx = documents.index(s)
        query_binary_medhop_instances += binary_medhop_instances[idx]
    return query_binary_medhop_instances


def extract_graph(df, sentence_wise=True, entity_list_path=None):
    # extract all unique supports
    supports = sorted(list(set(chain.from_iterable(df['supports']))))
    print('Total documents: {}'.format(len(supports)))
    binary_medhop_instances, document_store = extract_document_binary_medhop_instances(supports,
                                                                                       sentence_wise,
                                                                                       entity_list_path)

    df['graph'] = df.apply(lambda row: _extract_graph(row['supports'],
                                                      supports,
                                                      binary_medhop_instances),
                           axis=1)
    return document_store


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
        else:
            relations[(entity_id_mapping[ent1], entity_id_mapping[ent2])].append(relation)
    # create adjacency matrix with entity_ids
    adj_matrix = np.zeros((len(entity_id_mapping), len(entity_id_mapping)))
    for (ent1_id, ent2_id) in relations.keys():
        adj_matrix[ent1_id, ent2_id] = 1.0
    labels = [i[0] for i in sorted(entity_id_mapping.items(), key=operator.itemgetter(1))]
    return entity_id_mapping, adj_matrix, relations, labels


def preprocess_medhop(df, sentence_wise=True, entity_list_path=None):
    document_store = extract_graph(df, sentence_wise, entity_list_path)
    print('Extracted graph data')
    extract_target_and_relation(df)
    print('Extracted target and relation')
    return df.drop('supports', axis=1), document_store


if __name__ == "__main__":
    df = pd.read_json('./qangaroo_v1.1/medhop/train.json', orient='records')
    df, document_store = preprocess_medhop(df,
                                           sentence_wise=False,
                                           entity_list_path='./parsing/entities.txt')
