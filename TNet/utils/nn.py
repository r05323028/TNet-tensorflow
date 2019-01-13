import tensorflow as tf


def hard_sigmoid(x):
    x_ = tf.math.maximum(0.0, tf.math.minimum(1.0, x*0.2 + 0.5))

    return x_