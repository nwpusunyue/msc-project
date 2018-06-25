import tensorflow as tf


def extract_axis_1(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted. [batch_size, seq_len, ...]
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """

    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)

    return res


def replace_val(data, val_to_replace, replacement):
    return tf.where(condition=tf.equal(data, val_to_replace), x=tf.fill(tf.shape(data), replacement),
                    y=data)
