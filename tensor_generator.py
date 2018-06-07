import numpy as np
from parsing.special_tokens import *


def get_indexed_paths(q_relation_paths,
                      q_entity_paths,
                      vocab,
                      max_paths,
                      max_path_length,
                      max_relation_length):
    '''
    eg: Hanging Gardens are located in Mumbai , capital of India

    :param q_relation_paths: list containing an entry for each query: the entry represents the list of relation paths
    corresponding to that query; eg: [['are located in', ', capital of', '#END'], ...]
    :param q_entity_paths: list containing an entry for each query: the entry represents the list of entity paths
    corresponding to that query; eg: [['Hanging_Gardens', 'Mumbai', 'India'], ...]
    :param vocab: dict with keys: words, values: indices of the words, corresponding to rows in the embedding matr
    :param max_paths: max. number of paths per query; queries with more than this, will have the extra paths truncated
    :param max_path_length: max. number of relations and entities per path (including final dummy relation); paths with
    more nodes will be discarded
    :param max_relation_length: max. number of tokens per relation; relations with more tokens will have the extra
    tokens truncated
    :return: indexed relation paths: [n_queries, max_paths, max_path_length, max_rel_length]
             indexed entity paths: [n_queries, max_paths, max_path_length]
             num_paths: [n_queries] - num of paths per query
             num_nodes: [n_queries, max_paths] - num of  nodes per path
             num_words: [n_queries, max_paths, max_path_length] - num of words for each relation in each path

    '''
    num_paths = np.array([len(relation_paths) for relation_paths in q_relation_paths])
    # Any path beyond the actual number of paths for a query will have a length of 0,
    # hence will not be run through the RNN and its hidden state will be 0.
    num_nodes = np.zeros([len(q_relation_paths),
                          max_paths])
    # Filled with ones since this tensor is used to divide the sum of all word embeddings in one relation
    # by actual number of words, so need to avoid division by 0. If a relation is made up of PAD only, then
    # this will yield 0, since [PAD] has a 0-filled embd. vector
    num_words = np.ones([len(q_relation_paths),
                         max_paths,
                         max_path_length])

    indexed_relation_paths = np.full(shape=[len(q_relation_paths),
                                            max_paths,
                                            max_path_length,
                                            max_relation_length],
                                     fill_value=vocab[PAD])
    indexed_entity_paths = np.full(shape=[len(q_relation_paths),
                                          max_paths,
                                          max_path_length],
                                   fill_value=vocab[PAD])

    for i, (relation_paths, entity_paths) in enumerate(zip(q_relation_paths, q_entity_paths)):

        for j, (rel_path, ent_path) in enumerate(zip(relation_paths, entity_paths)):
            num_nodes[i, j] = len(rel_path)

            for k, (rel, ent) in enumerate(zip(rel_path, ent_path)):
                indexed_entity_paths[i, j, k] = vocab[ent] if ent in vocab else vocab[UNK]
                num_words[i, j, k] = len(rel)

                for l, word in enumerate(rel):
                    indexed_relation_paths[i, j, k, l] = vocab[word] if word in vocab else vocab[UNK]
    return (indexed_relation_paths,
            indexed_entity_paths,
            num_paths,
            num_nodes,
            num_words)


def get_indexed_target_relations(target_relations,
                                 target_relation_vocab):
    '''

    :param target_relations: relations appearing in the question of each query;
    for med-hop this is just 'interacts-with'
    :param target_relation_vocab: keys: target relations
                                  values: index of the target relation in the target relation embedding matrix
    :return: nparray with indexed target relations
    '''

    indexed_target_relations = np.zeros(len(target_relations))
    for i, target_relation in enumerate(target_relations):
        indexed_target_relations[i] = target_relation_vocab[target_relation]

    return indexed_target_relations


def get_labels(labels_arr):
    return np.array(labels_arr)
