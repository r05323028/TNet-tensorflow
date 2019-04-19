import tensorflow as tf


class Projection:
    
    def __init__(self, hparams, mode):
        self.filter_nums = int(hparams[mode]['num_feature_maps'])
        self.output_nums = 3

        with tf.variable_scope('Softmax'):
            self.weights = {
                'output_weights': tf.get_variable(
                    name='output_W',
                    shape=[self.filter_nums, self.output_nums],
                    initializer=tf.initializers.random_uniform(-0.01, 0.01)
                )
            }
            self.bias = {
                'output_bias': tf.get_variable(
                    name='output_b',
                    shape=[self.output_nums],
                    initializer=tf.zeros_initializer()
                )
            }

    def __call__(self, cnn_output):
        output = tf.matmul(cnn_output, self.weights['output_weights']) + self.bias['output_bias']

        tf.summary.histogram('Projection/W', self.weights['output_weights'])
        tf.summary.histogram('Projection/b', self.bias['output_bias'])

        return output

