import tensorflow as tf

from TNet.model.layers.rnn import BiLSTM
from TNet.utils.nn import hard_sigmoid


class CPT:

    def __init__(self, hparams, mode='as'):
        self.mode = mode
        self.hidden_nums = int(hparams['global']['hidden_size'])

        with tf.variable_scope('TST_Variables', reuse=tf.AUTO_REUSE):
            self.weights = {
                'trans_weights': tf.get_variable(
                    name='trans_W',
                    shape=[4 * self.hidden_nums, 2 * self.hidden_nums],
                    initializer=tf.initializers.random_uniform(-0.01, 0.01)
                )
            }
            self.bias = {
                'trans_bias': tf.get_variable(
                    name='trans_b',
                    shape=[2 * self.hidden_nums],
                    initializer=tf.zeros_initializer()
                )
            }

        if self.mode == 'as':
            with tf.variable_scope("AS_Variables", reuse=tf.AUTO_REUSE):
                self.as_weights = {
                    'gate_weights': tf.get_variable(
                        name='gate_W',
                        shape=[2 * self.hidden_nums, 2 * self.hidden_nums],
                        initializer=tf.initializers.random_uniform(-0.01, 0.01)
                    )
                }
                self.as_bias = {
                    'gate_bias': tf.get_variable(
                        name='gate_b',
                        shape=[2 * self.hidden_nums],
                        initializer=tf.zeros_initializer()
                    )
                }

    def _tst(self, target_hidden_states, hidden_states):
        hidden_sp = tf.shape(hidden_states)
        batch_size = hidden_sp[0]

        # (max_seq_length, batch_size, 2 * hidden_size)
        hs_ = tf.transpose(hidden_states, perm=[1, 0, 2])
        # (batch_size, 2 * hidden_size, target_length)
        t_ = tf.transpose(target_hidden_states, perm=[0, 2, 1])

        # tst
        sentence_index = 0
        sentence_array = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)

        def body(sentence_index, sentence_array):
            # (batch_size, 2 * hidden_size)
            hi = tf.transpose(tf.gather_nd(hs_, [[sentence_index]]), perm=[1, 2, 0])

            # (batch_size, target_length)
            ai = tf.math.softmax(tf.squeeze(tf.matmul(target_hidden_states, hi), axis=-1))

            # (batch_size, 2 * hidden_size, 1)
            ti = tf.matmul(t_, tf.expand_dims(ai, axis=-1))

            # squeeze dim=1
            hi = tf.squeeze(hi, axis=-1)
            ti = tf.squeeze(ti, axis=-1)

            # concatenate (batch_size, 1, 4 * hidden_size)
            concated_hi = tf.concat([hi, ti], axis=-1)
            concated_hi = tf.reshape(concated_hi, [batch_size, 1, 4 * self.hidden_nums])

            if batch_size == 1:
                hi_new = tf.math.tanh(
                    tf.matmul(concated_hi, self.weights['trans_weights']) + self.bias['trans_bias']
                )

            else:
                hi_new = tf.math.tanh(
                    tf.matmul(concated_hi, tf.tile(tf.expand_dims(self.weights['trans_weights'], axis=0), [batch_size, 1, 1])) + self.bias['trans_bias']
                )

            hi_new = tf.squeeze(hi_new, axis=1)

            sentence_array = sentence_array.write(sentence_index, hi_new)

            return (sentence_index + 1, sentence_array)

        def cond(sentence_index, sentence_array):
            return sentence_index < hidden_sp[1]

        _, sentence_array = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=(sentence_index, sentence_array)
        )

        sentence_array = tf.transpose(sentence_array.stack(), perm=[1, 0, 2])

        tf.summary.histogram('TST/transform_W', self.weights['trans_weights'])
        tf.summary.histogram('TST/transform_b', self.bias['trans_bias'])

        return sentence_array

    def _as_layer(self, target_hidden_states, hidden_states):
        hidden_sp = tf.shape(hidden_states)
        batch_size = hidden_sp[0]

        # AS
        # gate
        if batch_size == 1:
            trans_gate = tf.sigmoid(
                tf.matmul(hidden_states, self.as_weights['gate_weights']) + self.as_bias['gate_bias']
            )
        
        else:
            trans_gate = tf.sigmoid(
                tf.matmul(hidden_states, tf.tile(tf.expand_dims(self.as_weights['gate_weights'], axis=0), (batch_size, 1, 1))) + self.as_bias['gate_bias']
            )

        hidden_states_ = self._tst(target_hidden_states, hidden_states)

        # summarize
        tf.summary.histogram('CPT_AS/gate_W', self.as_weights['gate_weights'])
        tf.summary.histogram('CPT_AS/gate_b', self.as_bias['gate_bias'])

        return trans_gate * hidden_states_ + (1.0 - trans_gate) * hidden_states

    def _lf_layer(self, target_hidden_states, hidden_states):
        hidden_states_ = self._tst(target_hidden_states, hidden_states)

        return hidden_states_ + hidden_states

    def __call__(self, target_hidden_states, hidden_states):
        """
        Input: {
            target_embeddings: (?, ?, embedding_size), 
            target_sequence_length: (?, ), 
            hidden_states: (?, ?, 2 * hidden_nums)
            }
        """
        if self.mode == 'as':
            output = self._as_layer(target_hidden_states, hidden_states)

        else:
            output = self._lf_layer(target_hidden_states, hidden_states)

        return output

