import numpy as np
import tensorflow as tf


def attend_sequence(seq, seq_len, name='attention', module='additive'):
    '''

    :param seq: [batch_size, max_seq_len, repr_dim] - sequence to attend. Normally output of an LSTM.
    :param seq_len: [batch_size]
    :param name: scope name
    :param module: one of: 'additive' (tbc.)
    :return:
    [batch_size, max_seq_len, repr_dim] - attended sequence
    '''
    attended_seq = None
    attention_weights = None

    with tf.variable_scope(name):
        if module == 'additive':
            # [batch_size, seq_len, 1]
            attention_scores = tf.layers.dense(seq, 1)
            # [batch_size, seq_len] - contains -1000.0 to the right of the length of each seq. in the batch.
            # This ensures that anything outside the sequence receives an attention weight approximately to 0
            mask = (1.0 - tf.sequence_mask(seq_len, seq.shape[1], dtype=tf.float32)) * (-1000.0)

            # [batch_size, seq_len]
            attention_scores = attention_scores + tf.expand_dims(mask, axis=2)
            # [batch_size, seq_len]
            attention_weights = tf.nn.softmax(attention_scores, axis=1)

            # [batch_size, repr_dim]
            attended_seq = tf.reduce_sum(attention_weights * seq, axis=1)
        else:
            raise ValueError('{} is not a valid module name.'.format(module))
    return attended_seq, tf.squeeze(attention_weights, axis=2)


if __name__ == '__main__':
    batch_size = 2
    max_seq_len = 5
    repr_size = 10

    test_seq = np.zeros([batch_size, max_seq_len, repr_size])
    test_seq_len = np.zeros([batch_size])

    test_seq[0, 0, :] = 1
    test_seq[0, 1, :] = 2
    test_seq[1, 0, :] = 2

    test_seq_len[0] = 2
    test_seq_len[1] = 1

    seq = tf.placeholder(tf.float32, shape=[None, max_seq_len, repr_size])
    seq_len = tf.placeholder(tf.int32, shape=[None])
    attended_seq, attention_weights = attend_sequence(seq, seq_len)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        test_attended_seq, test_attention_weights = sess.run([attended_seq, attention_weights], feed_dict={
            seq: test_seq,
            seq_len: test_seq_len
        })

        assert test_attended_seq.shape == (batch_size, repr_size)
        assert test_attention_weights.shape == (batch_size, max_seq_len)

        assert (test_attention_weights[0, int(test_seq_len[0]):] == 0).all()
        assert (test_attention_weights[1, int(test_seq_len[1]):] == 0).all()
        assert np.logical_and(1.0 < test_attended_seq[0, :], test_attended_seq[0, :] < 2.0).all()
        assert (test_attended_seq[1, :] == 2.0).all()
