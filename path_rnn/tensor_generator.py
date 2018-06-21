import random

import numpy as np

from parsing.special_tokens import *


def get_indexed_paths(q_relation_paths,
                      q_entity_paths,
                      target_relations,
                      document_store,
                      relation_token_vocab,
                      entity_vocab,
                      target_relation_vocab,
                      max_path_length,
                      max_relation_length,
                      num_words_filler=1,
                      replace_in_doc=True,
                      truncate_doc=False):
    '''
    eg: Hanging Gardens are located in Mumbai , capital of India

    :param q_relation_paths: list containing an entry for each query: the entry represents the list of relation paths
    corresponding to that query; eg: [['are located in', ', capital of', '#END'], ...]
    :param q_entity_paths: list containing an entry for each query: the entry represents the list of entity paths
    corresponding to that query; eg: [['Hanging_Gardens', 'Mumbai', 'India'], ...]
    :param target_relations: list containing the target relation for each query: eg: ['interacts_with',...]
    :param document_store: a DocumentStore object which contains all the documents in the dataset ordered by id,
    along with the entities they contain and their position in the doc.
    :param labels: list containing the labels for each query
    :param relation_token_vocab: dict with keys: words, values: indices of the words,
    corresponding to rows in the embedding matr
    :param entity_vocab: dict with keys: words, values: indices of the words,
    corresponding to rows in the embedding matr
    :param target_relation_vocab: dict with keys: words, values: indices of the words,
    corresponding to rows in the embedding matr
    :param max_path_length: max. number of relations and entities per path (including final dummy relation); paths with
    more nodes will be discarded
    :param max_relation_length: max. number of tokens per relation; relations with more tokens will have the extra
    tokens truncated
    :param replace_in_doc: whether to replace the entities involved in a relation in the document with special tokerns
    ent1, ent2
    :param truncate_doc: whether to truncate document to only the span containing both involved entities
    :return: indexed relation paths: [total_paths, max_path_length, max_rel_length]
             indexed entity paths: [total_paths, max_path_length]
             indexed_target_relations: [total_paths]
             path_partitions: [num_queries, 2] - specifies for each  query start and end index of its corresponding
             paths
             path_lengths: [total_paths] - num of  nodes per path
             num_words: [total_paths, max_path_length] - num of words for each relation in each path

    '''
    path_partitions = np.zeros((len(q_relation_paths), 2), dtype=int)
    total_paths = np.sum([len(rel_paths) for rel_paths in q_relation_paths])
    print('Total paths: {}'.format(total_paths))
    # Any path beyond the actual number of paths for a query will have a length of 0,
    # hence will not be run through the RNN and its hidden state will be 0.
    path_lengths = np.zeros([total_paths])
    # Filled with ones since this tensor is used to divide the sum of all word embeddings in one relation
    # by actual number of words, so need to avoid division by 0. If a relation is made up of PAD only, then
    # this will yield 0, since [PAD] has a 0-filled embd. vector
    num_words = np.full(shape=[total_paths,
                               max_path_length],
                        fill_value=num_words_filler)

    indexed_relation_paths = np.full(shape=[total_paths,
                                            max_path_length,
                                            max_relation_length],
                                     fill_value=relation_token_vocab[PAD])
    indexed_entity_paths = np.full(shape=[total_paths,
                                          max_path_length],
                                   fill_value=entity_vocab[PAD])
    indexed_target_relations = np.zeros(shape=total_paths)

    path_idx = 0
    for query_idx, (relation_paths, entity_paths, target_relation) in enumerate(zip(q_relation_paths,
                                                                                    q_entity_paths,
                                                                                    target_relations)):
        partition_start = path_idx
        for (rel_path, ent_path) in zip(relation_paths,
                                        entity_paths):
            path_lengths[path_idx] = len(rel_path)
            indexed_target_relations[path_idx] = target_relation_vocab[target_relation]

            for rel_idx, (rel, ent_pair) in enumerate(zip(rel_path,
                                                          zip(ent_path, ent_path[1:] + [-1]))):
                indexed_entity_paths[path_idx, rel_idx] = entity_vocab[ent_pair[0]] if ent_pair[0] in entity_vocab \
                    else entity_vocab[UNK]

                if rel != -1:
                    if replace_in_doc:
                        rel_doc = document_store.get_document(rel, ent_pair[0], ent_pair[1], truncate_doc)
                    else:
                        rel_doc = document_store.documents[rel]
                else:
                    rel_doc = [END]

                num_words[path_idx, rel_idx] = len(rel_doc)

                for word_idx, word in enumerate(rel_doc):
                    indexed_relation_paths[path_idx, rel_idx, word_idx] = relation_token_vocab[word] \
                        if word in relation_token_vocab else relation_token_vocab[UNK]
            path_idx += 1
        partition_end = path_idx
        path_partitions[query_idx, 0] = partition_start
        path_partitions[query_idx, 1] = partition_end
    return (indexed_relation_paths,
            indexed_entity_paths,
            indexed_target_relations,
            path_partitions,
            path_lengths,
            num_words)


def get_synthetic_dataset(num_positive,
                          num_negative,
                          path_length,
                          paths_per_query,
                          doc_range_positive,
                          doc_range_negative,
                          positive_entities,
                          negative_entities,
                          target_relation):
    q_relation_paths = []
    q_entity_paths = []
    target_relations = []
    labels = []

    for i in range(num_positive):
        paths = []
        entity_paths = []
        for j in range(paths_per_query):
            path = []
            ent_path = []
            for k in range(path_length):
                doc_idx = np.random.randint(doc_range_positive[0], doc_range_positive[1])
                path.append(doc_idx)
                ent_path.append(random.choice(positive_entities))
            paths.append(path)
            entity_paths.append(ent_path)
        q_relation_paths.append(paths)
        q_entity_paths.append(entity_paths)
        target_relations.append(target_relation)
        labels.append(1)

    for i in range(num_negative):
        paths = []
        entity_paths = []
        for j in range(paths_per_query):
            path = []
            ent_path = []
            for k in range(path_length):
                doc_idx = np.random.randint(doc_range_negative[0], doc_range_negative[1])
                path.append(doc_idx)
                ent_path.append(random.choice(negative_entities))
            paths.append(path)
            entity_paths.append(ent_path)
        q_relation_paths.append(paths)
        q_entity_paths.append(entity_paths)
        target_relations.append(target_relation)
        labels.append(0)

    return {'relation_paths': q_relation_paths,
            'entity_paths': q_entity_paths,
            'relation': target_relations,
            'label': labels}


def get_labels(labels_arr):
    return np.array(labels_arr)
