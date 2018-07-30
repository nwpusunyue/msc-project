import numpy as np
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


def rank_loss(y_true, y_pred, margin=0.0):
    """
    Computes a ranking loss as follows:
    sum over all positives (max(0, max_neg_score - positive_score + margin))
    :param y_true: [batch_size]
    :param y_pred: [batch_size]
    :param margin: margin by which we want the positive example to be larger than the negative examples
    :return: loss value
    """
    y_true = tf.cast(y_true, y_pred.dtype)

    max_neg_score = tf.reduce_max((1.0 - y_true) * y_pred)
    return tf.reduce_sum(tf.nn.relu(y_true * (max_neg_score - y_pred + margin)))


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


if __name__ == '__main__':
    x = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_true = tf.placeholder(shape=[None], dtype=tf.int32)
    margin = tf.placeholder(shape=None, dtype=tf.float32)

    y_pred = tf.layers.dense(tf.layers.dense(x, 10, activation=tf.nn.relu), 1)
    y_pred = tf.nn.sigmoid(y_pred)
    y_pred = tf.squeeze(y_pred, axis=1)

    y_pred_const = tf.placeholder(shape=[None], dtype=tf.float32)
    loss_const = rank_loss(y_true=y_true, y_pred=y_pred_const, margin=margin)

    loss = rank_loss(y_true=y_true, y_pred=y_pred, margin=margin)
    train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)


    test_x = np.expand_dims(np.array([1, 2, 3, 100, 101, 102, 103]), axis=1)
    test_y_true = [1, 1, 1, 0, 0, 0, 0]
    test_y_pred_const = [0.8, 0.6, 0.2, 0.7, 0.3, 0.3, 0.3]
    test_margin = 0.2

    with tf.train.MonitoredTrainingSession() as sess:
        test_loss = sess.run(loss_const, feed_dict={y_true: test_y_true,
                                                    y_pred_const: test_y_pred_const,
                                                    margin: test_margin})
        assert np.isclose(test_loss, 1.1)

        for i in range(1000):
            _, test_y_pred, test_loss = sess.run([train_step, y_pred, loss], feed_dict={y_true: test_y_true,
                                                                                        x: test_x,
                                                                                        margin: test_margin})
            if np.isclose(test_loss, 0.0):
                break

        assert (test_y_pred[:3] > test_margin).all()
        assert (test_y_pred[3:] < 1e-1).all()
