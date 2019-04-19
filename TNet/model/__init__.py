import os
import numpy as np
import tensorflow as tf
from configparser import ConfigParser, ExtendedInterpolation

from TNet.model.layers.rnn import BiLSTM
from TNet.model.layers.cpt import CPT
from TNet.model.layers.cnn import CNN
from TNet.model.layers.output import Projection


class TNet:
    
    def __init__(self, hparams, *args, **kwargs):
        self.model_name = kwargs.get('model_name')
        self.model_dir = 'models'
        self.log_dir = 'logs'
        self.dropout_rate = float(hparams['global']['dropout_rate'])
        self.lr = float(hparams['global']['learning_rate'])
        self.mode = kwargs.get('mode')
        self.update_counter = 0

        self.bilstm_layer_bottom = BiLSTM(hparams, scope='bottom')
        self.bilstm_layer_cpt = BiLSTM(hparams,scope='cpt')
        self.cpt_layer = CPT(hparams, mode=self.mode)
        self.cnn_layer = CNN(hparams)
        self.output_layer = Projection(hparams, mode=self.mode)
        self._build_model()

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)

        # variables initializing
        self.sess.run(tf.global_variables_initializer())

    def _position_embedding(self, hs, pw):
        """
        @hs: (batch_size, sentence_length, 2 * hidden_nums)
        @pw: (batch_size, sentence_length)
        """
        weighted_hs = hs * tf.expand_dims(pw, axis=-1)

        return weighted_hs

    def _build_model(self):       
        # placeholder
        self.word_embeddings = tf.placeholder(tf.float32, shape=[None, None, 300])
        self.sequence_length = tf.placeholder(tf.int32, shape=[None])
        self.target_embeddings = tf.placeholder(tf.float32, shape=[None, None, 300])
        self.target_sequence_length = tf.placeholder(tf.int32, shape=[None])
        self.position_weight = tf.placeholder(tf.float32, shape=[None, None])
        self.labels = tf.placeholder(tf.int32, shape=[None, 3])
        self.is_training = tf.placeholder(tf.bool)

        opt = tf.train.AdamOptimizer(learning_rate=self.lr)

        # layers
        
        # dropout word embeddings
        self.word_embeddings = tf.layers.dropout(self.word_embeddings, rate=self.dropout_rate, training=self.is_training)
        self.target_embeddings = tf.layers.dropout(self.target_embeddings, rate=self.dropout_rate, training=self.is_training)

        # (batch_size, target_sequence_length, 2*hidden_nums)        
        target_hidden_states = self.bilstm_layer_cpt(
            self.target_embeddings,
            self.target_sequence_length
        ) 
        
        # BiLSTM
        # (?, ?, 2 * hidden_nums)
        hidden_states = self.bilstm_layer_bottom(
            self.word_embeddings,
            self.sequence_length
        ) 

        # CPT 1
        # (batch_size, max_sequence_length, 2*hidden_nums)
        self.tailor_made_hidden_states_1 = self.cpt_layer(
            target_hidden_states,
            hidden_states
        )

        # position weighting
        self.tailor_made_hidden_states_1 = self._position_embedding(
            self.tailor_made_hidden_states_1, 
            self.position_weight
        )

        # CPT 2
        # (batch_size, max_sequence_length, 2*hidden_nums)
        self.tailor_made_hidden_states_2 = self.cpt_layer(
            target_hidden_states,
            self.tailor_made_hidden_states_1
        )

        # position weighting
        self.tailor_made_hidden_states_2 = self._position_embedding(
            self.tailor_made_hidden_states_2, 
            self.position_weight
        )

        # CNN feature extractor
        # (batch_size, filter_nums)
        self.convolved_hidden_states, self.feature_maps = self.cnn_layer(self.tailor_made_hidden_states_2)

        # dropout word representation
        self.convolved_hidden_states = tf.layers.dropout(self.convolved_hidden_states, rate=self.dropout_rate, training=self.is_training)

        # (batch_size, output_nums)
        self.logits = self.output_layer(self.convolved_hidden_states)

        # softmax cross entropy
        self.loss = tf.losses.softmax_cross_entropy(
            onehot_labels=self.labels,
            logits=self.logits
        )

        # accuracy
        self.acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(self.labels, axis=1), tf.argmax(self.logits, axis=1)),
            dtype=tf.float32))

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('acc', self.acc)

        # build training optimizer
        self.train_op = opt.minimize(self.loss)

    def save_model(self):
        self.saver.save(
            self.sess, 
            os.path.join(self.model_dir, self.model_name)
        )

    def load_model(self):
        self.saver.restore(
            self.sess,
            os.path.join(self.model_dir, self.model_name)
        )

    def train_on_batch(self, sentence_embeddings, sentence_length, target_embeddings, target_length, pw, labels):
        self.sess.run(
            self.train_op,
            feed_dict={
                self.word_embeddings: sentence_embeddings,
                self.sequence_length: sentence_length,
                self.target_embeddings: target_embeddings,
                self.target_sequence_length: target_length,
                self.position_weight: pw,
                self.labels: labels,
                self.is_training: True
            }
        )

        self.update_counter += 1
            
        self.training_summary = self.sess.run(
            self.merged,
            feed_dict={
                self.word_embeddings: sentence_embeddings,
                self.sequence_length: sentence_length,
                self.target_embeddings: target_embeddings,
                self.target_sequence_length: target_length,
                self.position_weight: pw,
                self.labels: labels,
                self.is_training: True
            }
        )

        self.train_writer.add_summary(self.training_summary, self.update_counter)

    def predict_on_batch(self, sentence_embeddings, sentence_length, target_embeddings, target_length, pw):
        max_idx = tf.argmax(self.logits, axis=1)
        
        pred = self.sess.run(
            tf.one_hot(max_idx, depth=3),
            feed_dict={
                self.word_embeddings: sentence_embeddings,
                self.sequence_length: sentence_length,
                self.target_embeddings: target_embeddings,
                self.target_sequence_length: target_length,
                self.position_weight: pw,
                self.is_training: False
            }
        )

        return pred

    def test_acc(self, sentence_embeddings, sentence_length, target_embeddings, target_length, pw, labels):
        acc = self.sess.run(
            self.acc,
            feed_dict={
                self.word_embeddings: sentence_embeddings,
                self.sequence_length: sentence_length,
                self.target_embeddings: target_embeddings,
                self.target_sequence_length: target_length,
                self.position_weight: pw,
                self.labels: labels,
                self.is_training: False
            }
        )

        return acc