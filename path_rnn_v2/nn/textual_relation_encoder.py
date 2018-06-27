import numpy as np
import tensorflow as tf

from parsing.special_tokens import *
from path_rnn_v2.util.embeddings import embed_sequence, Word2VecEmbeddings
from path_rnn_v2.util.sequence_encoder import encoder


def encode_relation(rel_seq, rel_length, embd,
                    seq_embedder_params, seq_encoder_params, name='textual_relation_encoder'):
    '''

    :param rel_seq: [batch_size, max_path_length, max_rel_length]
    :param rel_length: [bath_size, max_path_length]
    :param embd: An instantiation of Embeddings class
    :param seq_embedder_params: kwargs for the embedder
    :param seq_encoder_params: kwargs for the encoder
    :param name: variable scope name
    :return:
    '''
    with tf.variable_scope(name):
        # [batch_size, max_path_length, max_rel_length, embd_dim]
        rel_seq_embd = embed_sequence(seq=rel_seq, embd=embd, **seq_embedder_params)

        # max_path_length x [batch_size]
        rel_length_unstacked = tf.unstack(rel_length, axis=1)
        # max_path_length x [batch_size, max_rel_length, embd_dim]
        rel_seq_unstacked = tf.unstack(rel_seq_embd, axis=1)

        rel_seq_repr_unstacked = []
        for seq, len in zip(rel_seq_unstacked, rel_length_unstacked):
            # [batch_size, repr_dim]
            output = encoder(seq, len, reuse=tf.AUTO_REUSE, **seq_encoder_params)
            rel_seq_repr_unstacked.append(output)

        # [batch_size, max_path_length, repr_dim]
        rel_repr = tf.stack(rel_seq_repr_unstacked, axis=1)

    return rel_repr


# TEST
if __name__ == '__main__':
    batch_size = 2
    max_path_length = 2
    max_relation_length = 6
    repr_dim = 20

    embd = Word2VecEmbeddings('medhop_word2vec_punkt',
                              name='token_embd',
                              unk_token=UNK,
                              trainable=False,
                              special_tokens=[(ENT_1, False), (ENT_2, False), (ENT_X, False),
                                              (UNK, False),
                                              (END, False), (PAD, True)])

    test_rel_seq = np.full([batch_size, max_path_length, max_relation_length],
                           fill_value=embd.get_idx(PAD))
    test_rel_length = np.zeros([batch_size, max_path_length])

    embd.embedding_matrix[embd.get_idx('the')] = 1.0
    embd.embedding_matrix[embd.get_idx('protein')] = 2.0
    embd.embedding_matrix[embd.get_idx('is')] = 3.0
    embd.embedding_matrix[embd.get_idx('good')] = 4.0
    embd.embedding_matrix[embd.get_idx('drug')] = 5.0
    embd.embedding_matrix[embd.get_idx('interacts')] = 6.0
    embd.embedding_matrix[embd.get_idx('interact')] = 6.0
    embd.embedding_matrix[embd.get_idx('does')] = 7.0
    embd.embedding_matrix[embd.get_idx('not')] = 8.0

    t1 = np.array([embd.get_idx(token) for token in 'the protein is good'.split(' ')])
    t1_avg = np.mean(np.array([embd.embedding_matrix[idx] for idx in t1]), axis=0)

    t2 = np.array([embd.get_idx(token) for token in 'the drug interacts'.split(' ')])
    t2_avg = np.mean(np.array([embd.embedding_matrix[idx] for idx in t2]), axis=0)

    t3 = np.array([embd.get_idx(token) for token in 'the drug does not interact'.split(' ')])
    t3_avg = np.mean(np.array([embd.embedding_matrix[idx] for idx in t3]), axis=0)

    test_rel_seq[0, 0, :len(t1)] = t1
    test_rel_length[0, 0] = len(t1)
    test_rel_seq[0, 1, :len(t2)] = t2
    test_rel_length[0, 1] = len(t2)
    test_rel_seq[1, 0, :len(t3)] = t3
    test_rel_length[1, 0] = len(t3)

    print('Test relation sequence:\n {}'.format(test_rel_seq))
    print('Test relation lengths:\n {}'.format(test_rel_length))

    rel_seq = tf.placeholder(tf.int32,
                             shape=[None,
                                    max_path_length,
                                    max_relation_length])
    rel_len = tf.placeholder(tf.int32,
                             shape=[None,
                                    max_path_length])
    rel_repr = encode_relation(rel_seq, rel_len, embd,
                               seq_embedder_params={'name': 'test_embd',
                                                    'with_projection': False},
                               seq_encoder_params={
                                   'module': 'average'
                               })
    rel_repr_lstm = encode_relation(rel_seq, rel_len, embd,
                                    name='lstm_encoder',
                                    seq_embedder_params={'name': 'test_embd',
                                                         'with_projection': False},
                                    seq_encoder_params={
                                        'module': 'lstm',
                                        'repr_dim': repr_dim
                                    })

    for op in tf.get_default_graph().get_operations():
        print(str(op.name))
    with tf.train.MonitoredTrainingSession() as sess:
        test_rel_seq_repr = sess.run(rel_repr,
                                     feed_dict={rel_seq: test_rel_seq,
                                                rel_len: test_rel_length})
        print('Output shape', test_rel_seq_repr.shape)
        np.testing.assert_allclose(t1_avg, test_rel_seq_repr[0, 0, :])
        np.testing.assert_allclose(t2_avg, test_rel_seq_repr[0, 1, :])
        np.testing.assert_allclose(t3_avg, test_rel_seq_repr[1, 0, :])
        assert (0 == test_rel_seq_repr[1, 1, :]).all()

        test_rel_seq_repr_lstm = sess.run(rel_repr_lstm,
                                          feed_dict={rel_seq: test_rel_seq,
                                                     rel_len: test_rel_length}
                                          )
        assert test_rel_seq_repr_lstm.shape == (batch_size, max_path_length, 2 * repr_dim)
        assert (0 == test_rel_seq_repr_lstm[1, 1, :]).all()
