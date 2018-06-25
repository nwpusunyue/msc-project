import tensorflow as tf


def activation_from_string(activation_str):
    if activation_str is None:
        return tf.identity
    return getattr(tf.nn, activation_str)
