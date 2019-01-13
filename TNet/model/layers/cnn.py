import tensorflow as tf
import numpy as np


class CNN:

    def __init__(self, hidden_nums=50, filter_nums=50):
        self.hidden_nums = hidden_nums

        # parameters
        self.kernel_size = 3
        self.filter_nums = filter_nums

        with tf.variable_scope('CNN_Variables', reuse=tf.AUTO_REUSE):
            self.W = tf.get_variable(
                name='CNN_W',
                shape=[self.kernel_size, 2*self.hidden_nums, 1, self.filter_nums],
                initializer=tf.initializers.random_uniform(-0.01, 0.01))

            self.b = tf.get_variable(
                name='CNN_b',
                shape=[self.filter_nums],
                initializer=tf.zeros_initializer()
            )

    def __call__(self, hidden_states):
        # add channels
        hs = tf.expand_dims(hidden_states, axis=-1)

        # (batch_size, sentence_length - kernel_size + 1, channel_size, filter_nums)
        c = tf.nn.conv2d(
            hs,
            filter=self.W,
            strides=[1, 1, 1, 1],
            padding='VALID'
        )

        c = tf.nn.relu(c + self.b)

        pooled_c = tf.squeeze(tf.reduce_max(c, axis=1), axis=1)

        tf.summary.histogram('CNN/kernel', self.W)
        tf.summary.histogram('CNN/bias', self.b)

        return pooled_c, c
        