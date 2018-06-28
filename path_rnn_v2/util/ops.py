import tensorflow as tf


def extract_axis_1(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted. [batch_size, seq_len, ...]
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """
    # ensure all indices are positive
    ind = tf.nn.relu(ind)
    batch_range = tf.range(tf.shape(data, out_type=ind.dtype)[0], dtype=ind.dtype)
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)

    return res


def replace_val(data, val_to_replace, replacement):
    return tf.where(condition=tf.equal(data, val_to_replace), x=tf.fill(tf.shape(data), replacement),
                    y=data)


def create_reset_metric(metric, scope='reset_metrics', **metric_args):
    with tf.variable_scope(scope) as scope:
        metric_op, update_op = metric(**metric_args)
        vars = tf.contrib.framework.get_variables(
            scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(vars)
    return metric_op, update_op, reset_op
