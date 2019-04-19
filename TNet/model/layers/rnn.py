import numpy as np
import tensorflow as tf


class BiLSTM:

    def __init__(self, hparams, scope):
        self.scope = scope
        self.hidden_nums = int(hparams['global']['hidden_size'])

    def __call__(self, hs, seq_len):
        with tf.variable_scope('LSTM_Variables_%s' % self.scope, reuse=tf.AUTO_REUSE):
            self.fw_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_nums, name='fw_lstm_cell_%s' % self.scope)
            self.bw_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_nums, name='bw_lstm_cell_%s' % self.scope)

            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(
                self.fw_cell,
                self.bw_cell,
                hs,
                sequence_length=seq_len,
                dtype=tf.float32
            )

            output = tf.concat(
                [fw_out, bw_out], 
                axis=-1)
        
        return output

        


