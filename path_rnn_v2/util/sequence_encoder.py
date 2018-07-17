import numpy as np
import tensorflow as tf

from path_rnn_v2.util import rnn
from path_rnn_v2.util.activations import activation_from_string
from path_rnn_v2.util.attention import attend_sequence
from path_rnn_v2.util.ops import extract_axis_1, replace_val


def encoder(sequence, seq_length, repr_dim=100, module='lstm', name='encoder', reuse=False, activation=None,
            dropout=None, is_eval=True, extra_args=None):
    '''

    :param sequence: [batch_size, max_sequence_length, input_dim] tensor
    :param seq_length: [batch_size]
    :param repr_dim: output representation size, in case a neural layer is used
    :param module: 'lstm', 'rnn', 'gru', 'dense', 'average'
    :param name: scope name
    :param reuse: scope reuse
    :param activation: if not None, the output is passed through the given activation
    :param dropout: if 0.0 output is passed through a dropout layer
    :param is_eval: whether the tensors are for eval or train => influences dropout
    :param extra_args:
    :return:
    '''
    if extra_args is None:
        extra_args = {}

    with tf.variable_scope(name, reuse=reuse):
        if module == 'lstm':
            # [batch_size, repr_dim x 2]
            out = bi_lstm(repr_dim, sequence, seq_length, **extra_args)
        elif module == 'rnn':
            # [batch_size, repr_dim x 2]
            out = bi_rnn(repr_dim, tf.nn.rnn_cell.BasicRNNCell(repr_dim, activation_from_string(activation)),
                         sequence, seq_length, **extra_args)
        elif module == 'gru':
            # [batch_size, repr_dim x 2]
            out = bi_rnn(repr_dim, tf.contrib.rnn.GRUBlockCell(repr_dim), sequence, seq_length, **extra_args)
        elif module == 'additive_attention':
            # [batch_size, max_seq_len, repr_dimx x 2]
            out = bi_lstm(repr_dim, sequence, seq_length, last_output=False, **extra_args)
            # [batch_size, repr_dim x 2]
            out, _ = attend_sequence(out, seq_length, module='additive')
        elif module == 'dense':
            # [batch_size, repr_dim x 2]
            out = tf.layers.dense(sequence, repr_dim)
        elif module == 'average':
            # clip seq lengths such that if a sequence has 0 tokens
            # division by 0 is avoided. It is assumed that a tensor with
            # 0 tokens is filled with a special PAD character which has a 0-filled
            # vector repr
            seq_length = replace_val(seq_length, 0, 1)

            # [batch_size, input_dim]
            out = tf.reduce_sum(sequence, axis=1) / tf.cast(tf.expand_dims(seq_length, axis=1), dtype=sequence.dtype)
        elif module == 'identity':
            # should only be called when max_seq_len is 1
            # converts [batch_size, 1, input_dim] to [batch_size, input_dim]
            out = tf.squeeze(sequence, axis=1)

        if activation:
            out = activation_from_string(activation)(out)

        if dropout is not None:
            out = tf.cond(
                tf.logical_and(tf.greater(dropout, 0.0), tf.logical_not(is_eval)),
                lambda: tf.nn.dropout(out, 1.0 - dropout, noise_shape=[tf.shape(out)[0], tf.shape(out)[-1]]),
                lambda: out)
    return out


# RNN Encoders
def _bi_rnn(size, fused_rnn, sequence, seq_length, with_projection=False, projection_activation=None,
            with_backward=True, last_output=True):
    output = rnn.fused_birnn(fused_rnn, sequence, seq_length, with_backward=with_backward,
                             dtype=tf.float32, scope='rnn')[0]

    if last_output:
        if with_backward:
            # [batch_size, repr_dim x 2]
            output = tf.concat([extract_axis_1(output[0], seq_length - 1),
                                extract_axis_1(output[1], seq_length - 1)], axis=1)
        else:
            # [batch_size, repr_dim]
            output = extract_axis_1(output, seq_length - 1)
    else:
        if with_backward:
            # [batch_size, max_seq_len, repr_dim x 2]
            output = tf.concat([output[0], output[1]], axis=2)

    if with_projection:
        projection_initializer = tf.constant_initializer(np.concatenate([np.eye(size), np.eye(size)]))
        projection_activation = activation_from_string(
            projection_activation) if projection_activation is not None else projection_activation
        output = tf.layers.dense(output, size, kernel_initializer=projection_initializer, name='projection',
                                 activation=projection_activation)
    return output


def bi_lstm(size, sequence, seq_length, with_projection=False, projection_activation=None, with_backward=True,
            last_output=True):
    return _bi_rnn(size, (tf.contrib.rnn.LSTMBlockFusedCell(size), tf.contrib.rnn.LSTMBlockFusedCell(size)), sequence,
                   seq_length, with_projection,
                   projection_activation, with_backward, last_output)


def bi_rnn(size, rnn_cell, sequence, seq_length, with_projection=False, projection_activation=None, with_backward=True,
           last_output=True):
    fused_rnn = (tf.contrib.rnn.FusedRNNCellAdaptor(rnn_cell, use_dynamic_rnn=True),
                 tf.contrib.rnn.FusedRNNCellAdaptor(rnn_cell, use_dynamic_rnn=True))
    return _bi_rnn(size, fused_rnn, sequence, seq_length, with_projection, projection_activation, with_backward,
                   last_output)


if __name__ == '__main__':
    batch_size = 3
    max_sequence_length = 5
    input_dim = 10
    repr_dim = 15

    test_seq_length = [3, 4, 0]
    test_sequence = np.zeros([batch_size, max_sequence_length, input_dim])
    for i in range(batch_size):
        for j in range(test_seq_length[i]):
            test_sequence[i, j, :] = j + 1

    sequence = tf.placeholder(tf.float32, shape=[None, max_sequence_length, input_dim])
    unit_sequence = tf.placeholder(tf.float32, shape=[None, 1, input_dim])
    seq_length = tf.placeholder(tf.int32, shape=[None])

    avg_encoded = encoder(sequence, seq_length, module='average', name='average')
    gru_encoded = encoder(sequence, seq_length, module='gru', repr_dim=repr_dim, name='gru')
    identity_encoded = encoder(unit_sequence, seq_length, module='identity', name='identity')
    attention_encoded = encoder(sequence, seq_length, repr_dim=repr_dim, module='additive_attention', name='attention')

    with tf.train.MonitoredTrainingSession() as sess:
        test_encoded = sess.run(avg_encoded,
                                feed_dict={
                                    sequence: test_sequence,
                                    seq_length: test_seq_length
                                })
        assert test_encoded[0, 0] == 2.0
        assert test_encoded[1, 0] == 2.5
        assert test_encoded[2, 0] == 0.0

        test_encoded = sess.run(gru_encoded,
                                feed_dict={
                                    sequence: test_sequence,
                                    seq_length: test_seq_length
                                })

        assert test_encoded.shape == (batch_size, repr_dim * 2)
        assert (test_encoded[2, :] == 0).all()

        test_encoded = sess.run(attention_encoded,
                                feed_dict={
                                    sequence: test_sequence,
                                    seq_length: test_seq_length
                                })

        assert (test_encoded.shape == (batch_size, repr_dim * 2))
        assert (test_encoded[2] == 0.0).all()

        test_sequence = np.zeros([batch_size, 1, input_dim])
        test_seq_length = np.ones([batch_size])

        for i in range(batch_size):
            test_sequence[i, :] = i

        test_encoded = sess.run(identity_encoded,
                                feed_dict={
                                    unit_sequence: test_sequence,
                                    seq_length: test_seq_length
                                })

        assert test_encoded.shape == (batch_size, input_dim)
        assert (test_encoded[0] == 0).all()
        assert (test_encoded[1] == 1).all()
        assert (test_encoded[2] == 2).all()
