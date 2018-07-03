import numpy as np
import tensorflow as tf

from parsing.special_tokens import *
from path_rnn_v2.util.embeddings import Word2VecEmbeddings
from path_rnn_v2.util.sequence_encoder import encoder


def encode_path_elem(elem_seq, elem_length, embd,
                     seq_embedder_params, seq_encoder_params, is_eval=True, name='path_elem_encoder'):
    '''

    :param elem_seq: [batch_size, max_path_length, max_elem_length]
    :param elem_length: [bath_size, max_path_length]
    :param embd: An instantiation of Embeddings class
    :param seq_embedder_params: kwargs for the embedder
    :param seq_encoder_params: kwargs for the encoder
    :param is_eval: whether the tensors are for eval or train => influences dropout
    :param name: variable scope name
    :return:
    '''
    # [batch_size, max_path_length, max_rel_length, embd_dim]
    elem_seq_embd = embd.embed_sequence(seq=elem_seq, **seq_embedder_params)
    with tf.variable_scope(name):
        # max_path_length x [batch_size]
        elem_length_unstacked = tf.unstack(elem_length, axis=1)
        # max_path_length x [batch_size, max_rel_length, embd_dim]
        elem_seq_unstacked = tf.unstack(elem_seq_embd, axis=1)

        elem_seq_repr_unstacked = []

        for seq, len in zip(elem_seq_unstacked, elem_length_unstacked):
            output = encoder(seq, len, reuse=tf.AUTO_REUSE, is_eval=is_eval, **seq_encoder_params)

            elem_seq_repr_unstacked.append(output)

        # [batch_size, max_path_length, repr_dim]
        elem_repr = tf.stack(elem_seq_repr_unstacked, axis=1)

    return elem_repr


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
    rel_repr = encode_path_elem(rel_seq, rel_len, embd,
                                seq_embedder_params={'name': 'test_embd',
                                                     'with_projection': False},
                                seq_encoder_params={
                                    'module': 'average'
                                })
    rel_repr_lstm = encode_path_elem(rel_seq, rel_len, embd,
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
