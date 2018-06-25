import pickle

import pandas as pd
import tensorflow as tf

from path_rnn.tensor_generator import get_indexed_paths
from path_rnn_v2.util.embeddings import Word2VecEmbeddings, RandomEmbeddings
from path_rnn_v2.util.sequence_encoder import encoder
from parsing.special_tokens import *

if __name__ == '__main__':
    document_store = pickle.load(open('./data/train_doc_store_punkt.pickle',
                                      'rb'))
    dataset = pd.read_json(
        './data/sentwise=F_cutoff=4_limit=100_method=shortest_tokenizer=punkt_medhop_train.json')
    dataset = dataset.sample(2, random_state=10)

    relation_token_emb = Word2VecEmbeddings('medhop_word2vec_punkt',
                                            name='token_embd',
                                            unk_token=UNK,
                                            trainable=False,
                                            special_tokens=[(ENT_1, False), (ENT_2, False), (ENT_X, False),
                                                            (UNK, False),
                                                            (END, False), (PAD, True)])
    entity_emb = relation_token_emb
    target_relation_emb = RandomEmbeddings(dataset['relation'],
                                           name='target_rel_emb',
                                           embedding_size=100,
                                           unk_token=None,
                                           initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                       stddev=1.0,
                                                                                       dtype=tf.float64))

    # [batch_size, max_path_length, max_relation_length]
    relation_seq = tf.placeholder(tf.int32,
                                  shape=[None,
                                         5,
                                         570])
    # [batch_size, max_path_length]
    relation_len = tf.placeholder(tf.int32,
                                  shape=[None,
                                         5])

    # [vocab_size, embd_dim]
    relation_token_emb_matrix = relation_token_emb.get_embedding_matrix_tensor()
    relation_token_emb_dim = relation_token_emb_matrix.get_shape()[1].value

    # [batch_size, max_path_length, max_relation_length, emb_dim]
    relation_seq_token_embedded = tf.nn.embedding_lookup(relation_token_emb_matrix, relation_seq)
    # project
    # [batch_size, max_path_length, max_relation_length, emb_dim]
    relation_seq_token_embedded = tf.layers.dense(relation_seq_token_embedded,
                                                  relation_token_emb_dim,
                                                  activation=tf.tanh,
                                                  use_bias=False,
                                                  name='projection_rel_tok')

    # max_path_len x [batch_size]
    relation_len_unstacked = tf.unstack(relation_len, axis=1)
    # max_path_len x [batch_size, max_relation_length, emb_dim])
    relation_seq_unstacked = tf.unstack(relation_seq_token_embedded, axis=1)

    relation_repr_unstacked = []
    for rs, rl in zip(relation_seq_unstacked, relation_len_unstacked):
        # [batch_size, repr_dim]
        output = encoder(sequence=rs, seq_length=rl, repr_dim=100, module='average', name='relation_encoder',
                         reuse=tf.AUTO_REUSE)
        relation_repr_unstacked.append(output)

    # [batch_size, max_path_length, repr_dim]
    relation_repr = tf.stack(relation_repr_unstacked, axis=1)

    (indexed_relation_paths,
     indexed_entity_paths,
     indexed_target_relations,
     path_partitions,
     path_lengths,
     num_words) = get_indexed_paths(q_relation_paths=dataset['relation_paths'],
                                    q_entity_paths=dataset['entity_paths'],
                                    target_relations=dataset['relation'],
                                    document_store=document_store,
                                    relation_token_embeddings=relation_token_emb,
                                    entity_embeddings=entity_emb,
                                    target_relation_embeddings=target_relation_emb,
                                    max_path_length=5,
                                    max_relation_length=570,
                                    num_words_filler=0,
                                    replace_in_doc=True,
                                    truncate_doc=False)

    print(num_words)
    with tf.train.MonitoredTrainingSession() as sess:
        test_relation_repr = sess.run([relation_repr],
                                      feed_dict={
                                          relation_seq: indexed_relation_paths,
                                          relation_len: num_words
                                      })[0]
        print(test_relation_repr.shape)
        print(test_relation_repr[:, :, :5])
