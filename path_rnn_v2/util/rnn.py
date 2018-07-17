import tensorflow as tf


def fused_birnn(fused_rnn, inputs, sequence_length, initial_state=(None, None), dtype=None, scope=None,
                time_major=False, with_backward=True):
    with tf.variable_scope(scope or "BiRNN"):
        sequence_length = tf.cast(sequence_length, tf.int32)
        if not time_major:
            inputs = tf.transpose(inputs, [1, 0, 2])

        with tf.variable_scope('FW'):
            fw_fused_rnn = fused_rnn[0]
            outputs_fw, state_fw = fw_fused_rnn(inputs, sequence_length=sequence_length, initial_state=initial_state[0],
                                                dtype=dtype)

        if with_backward:
            bw_fused_rnn = fused_rnn[1]
            outputs_bw, state_bw = fused_rnn_backward(bw_fused_rnn, inputs, sequence_length, initial_state[1], dtype)

    if not time_major:
        outputs_fw = tf.transpose(outputs_fw, [1, 0, 2])
        if with_backward:
            outputs_bw = tf.transpose(outputs_bw, [1, 0, 2])

    if type(state_fw) is tf.contrib.rnn.LSTMStateTuple:
        state_fw = state_fw.c
        if with_backward:
            state_bw = state_bw.c

    if with_backward:
        return (outputs_fw, outputs_bw), (state_fw, state_bw)
    else:
        return outputs_fw, state_fw


def fused_rnn_backward(fused_rnn, inputs, sequence_length, initial_state=None, dtype=None,
                       time_major=True):
    with tf.variable_scope('BW'):

        if not time_major:
            inputs = tf.transpose(inputs, [1, 0, 2])

        rev_inputs = tf.reverse_sequence(inputs, sequence_length, 0, 1)
        rev_outputs, last_state = fused_rnn(rev_inputs, sequence_length=sequence_length,
                                            initial_state=initial_state,
                                            dtype=dtype)
        outputs = tf.reverse_sequence(rev_outputs, sequence_length, 0, 1)

        if not time_major:
            outputs = tf.transpose(outputs, [1, 0, 2])

    return outputs, last_state
