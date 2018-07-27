import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from parsing.special_tokens import *
from path_rnn_v2.util.embeddings import RandomEmbeddings


def get_medhop_tensors(query_rel_seq,
                       query_ent_seq,
                       query_target_rel,
                       query_label,
                       document_store,
                       rel_embeddings,
                       ent_embeddings,
                       target_rel_embeddings,
                       max_path_len,
                       max_rel_len,
                       max_ent_len,
                       rel_token_default=PAD,
                       ent_token_default=PAD,
                       rel_retrieve_params=None,
                       ent_retrieve_params=None):
    rel_retrieve_params = {} if rel_retrieve_params is None else rel_retrieve_params
    ent_retrieve_params = {} if ent_retrieve_params is None else ent_retrieve_params

    partition = np.zeros((len(query_rel_seq), 2), dtype=int)
    total_paths = np.sum([len(seqs) for seqs in query_rel_seq])
    print('Total paths: {}'.format(total_paths))
    path_len = np.zeros(total_paths)
    rel_len = np.zeros(shape=[total_paths,
                              max_path_len])
    ent_len = np.zeros(shape=[total_paths,
                              max_path_len])

    rel_seq = np.full(shape=[total_paths,
                             max_path_len,
                             max_rel_len],
                      fill_value=rel_embeddings.get_idx(rel_token_default))
    ent_seq = np.full(shape=[total_paths,
                             max_path_len,
                             max_ent_len],
                      fill_value=rel_embeddings.get_idx(ent_token_default))
    target_rel = np.zeros(shape=total_paths)

    path_idx = 0
    for query_idx, (rel_seqs, ent_seqs, t_rel) in enumerate(zip(query_rel_seq,
                                                                query_ent_seq,
                                                                query_target_rel)):
        target_rel[path_idx] = target_rel_embeddings.get_idx(t_rel)
        partition_start = path_idx
        for (rs, es) in zip(rel_seqs,
                            ent_seqs):
            path_len[path_idx] = len(rs)

            for node_idx, (rel, ent_pair) in enumerate(zip(rs,
                                                           zip(es, es[1:] + [None]))):
                if rel != -1:
                    rel_repr = document_store.get_document(rel, ent_pair[0], ent_pair[1], **rel_retrieve_params)
                    ent_repr = document_store.get_entity_neighb(rel, ent_pair[0], **ent_retrieve_params)
                else:
                    rel_repr = [END]
                    ent_repr = [
                        ent_pair[0] if 'replacement' not in ent_retrieve_params else ent_retrieve_params['replacement']]

                rel_len[path_idx, node_idx] = len(rel_repr)
                ent_len[path_idx, node_idx] = len(ent_repr)

                for word_idx, word in enumerate(rel_repr):
                    rel_seq[path_idx, node_idx, word_idx] = rel_embeddings.get_idx(word)

                for word_idx, word in enumerate(ent_repr):
                    ent_seq[path_idx, node_idx, word_idx] = ent_embeddings.get_idx(word)
            path_idx += 1
        partition_end = path_idx
        partition[query_idx, 0] = partition_start
        partition[query_idx, 1] = partition_end

    if query_label is not None:
        label = np.array(query_label)

        return (rel_seq,
                ent_seq,
                path_len,
                rel_len,
                ent_len,
                target_rel,
                partition,
                label)
    return (rel_seq,
            ent_seq,
            path_len,
            rel_len,
            ent_len,
            target_rel,
            partition)


def get_medhop_distance_tensors(query_rel_seq,
                                query_ent_seq,
                                query_target_rel,
                                query_label,
                                document_store,
                                target_rel_embeddings,
                                max_path_len, ):
    partition = np.zeros((len(query_rel_seq), 2), dtype=int)

    total_paths = np.sum([len(seqs) for seqs in query_rel_seq])
    print('Total paths: {}'.format(total_paths))

    path_len = np.zeros(total_paths)

    rel_seq = np.zeros(shape=[total_paths,
                              max_path_len,
                              1])

    target_rel = np.zeros(shape=total_paths)

    path_idx = 0
    for query_idx, (rel_seqs, ent_seqs, t_rel) in enumerate(zip(query_rel_seq,
                                                                query_ent_seq,
                                                                query_target_rel)):
        target_rel[path_idx] = target_rel_embeddings.get_idx(t_rel)
        partition_start = path_idx
        for (rs, es) in zip(rel_seqs,
                            ent_seqs):
            path_len[path_idx] = len(rs) - 1

            for node_idx, (rel, ent_pair) in enumerate(zip(rs,
                                                           zip(es, es[1:] + [None]))):
                if rel != -1:
                    dist = document_store.get_min_distance(rel, ent_pair[0], ent_pair[1])
                    rel_seq[path_idx, node_idx, 0] = dist
            path_idx += 1
        partition_end = path_idx
        partition[query_idx, 0] = partition_start
        partition[query_idx, 1] = partition_end

    label = np.array(query_label)

    return (rel_seq,
            path_len,
            target_rel,
            partition,
            label)


if __name__ == "__main__":
    path = './data'
    train = pd.read_json(
        '{}/sentwise=F_cutoff=4_limit={}_method={}_tokenizer=punkt_medhop_train_mini.json'.format(path, 100,
                                                                                                  'shortest'))
    train_document_store = pickle.load(open('{}/train_mini_doc_store_punkt.pickle'.format(path), 'rb'))
    target_embeddings = RandomEmbeddings(train['relation'],
                                         name='target_rel_emb',
                                         embedding_size=10,
                                         unk_token=None,
                                         initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                     stddev=1.0,
                                                                                     dtype=tf.float64))
    tensors = get_medhop_distance_tensors(query_rel_seq=train['relation_paths'],
                                          query_ent_seq=train['entity_paths'],
                                          query_target_rel=train['relation'],
                                          query_label=train['label'],
                                          max_path_len=4,
                                          target_rel_embeddings=target_embeddings,
                                          document_store=train_document_store)

    (rel_seq, path_len, target_rel, partition, label) = tensors
    print(rel_seq)
    print(np.mean(rel_seq[np.argwhere(label == 1)]))
    print(np.mean(rel_seq[np.argwhere(label == 0)]))
    print(path_len)
