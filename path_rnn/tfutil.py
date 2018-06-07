import tensorflow as tf


def clip(x, epsilon=1e-10):
    return tf.clip_by_value(x, epsilon, 1.0)


def loss_function(prob, label, epsilon=1e-10):
    return - label * tf.log(clip(prob, epsilon)) - (1 - label) * tf.log(clip(1 - prob, epsilon))
