import os
import numpy as np
import tensorflow as tf
from configparser import ConfigParser, ExtendedInterpolation

from TNet.model.layers.rnn import BiLSTM
from TNet.model.layers.cpt import CPT
from TNet.model.layers.cnn import CNN
from TNet.model.layers.output import Projection


class TNet:
    
    def __init__(self):
        config = ConfigParser(interpolation=ExtendedInterpolation())
        config.read('config.ini')
        self.model_name = config['basic']['model_name']
        self.model_dir = 'models'
        self.log_dir = config['basic']['log_dir']
        self.dropout_rate = 0.3
        self.lr = 0.002
        self.update_counter = 0

        self.bilstm_layer_bottom = BiLSTM(scope='bottom')
        self.bilstm_layer_cpt = BiLSTM(scope='cpt')
        self.cpt_layer = CPT(mode='as')
        self.cnn_layer = CNN()
        self.output_layer = Projection()
        self._build_model()

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
        self.evaluate_writer = tf.summary.FileWriter(self.log_dir + '/evaluate', self.sess.graph)

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
        self.dropout_placeholder = tf.placeholder(tf.bool)

        opt = tf.train.AdamOptimizer(learning_rate=self.lr)

        # layers
        
        # dropout word embeddings
        self.word_embeddings = tf.layers.dropout(self.word_embeddings, rate=self.dropout_rate, training=self.dropout_placeholder)
        self.target_embeddings = tf.layers.dropout(self.target_embeddings, rate=self.dropout_rate, training=self.dropout_placeholder)

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
        self.convolved_hidden_states = tf.layers.dropout(self.convolved_hidden_states, rate=self.dropout_rate, training=self.dropout_placeholder)

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

    def _get_train_eval_split(self, sentence_embeddings, sentence_length, target_embeddings, target_length, pw, labels, evaluation_size=None):
        batch_size = sentence_embeddings.shape[0]
        split_point = int(batch_size * evaluation_size)

        # training
        training_sentence_embeddings = sentence_embeddings[:split_point]
        training_sentence_length = sentence_length[:split_point]
        training_target_embeddings = target_embeddings[:split_point]
        training_target_length = target_length[:split_point]
        train_pw = pw[:split_point]
        training_labels = labels[:split_point]

        # evaluating
        eval_sentence_embeddings = sentence_embeddings[split_point:]
        eval_sentence_length = sentence_length[split_point:]
        eval_target_embeddings = target_embeddings[split_point:]
        eval_target_length = target_length[split_point:]
        eval_pw = pw[split_point:]
        eval_labels = labels[split_point:]

        return (
            (training_sentence_embeddings, training_sentence_length, training_target_embeddings, training_target_length, train_pw, training_labels),
            (eval_sentence_embeddings, eval_sentence_length, eval_target_embeddings, eval_target_length, eval_pw, eval_labels),
        )

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

    def train_on_batch(self, sentence_embeddings, sentence_length, target_embeddings, target_length, pw, labels, evaluation_size=0.2):
        if evaluation_size:
            (training_data, evaluating_data) = self._get_train_eval_split(
                sentence_embeddings, 
                sentence_length, 
                target_embeddings, 
                target_length,
                pw,
                labels, 
                evaluation_size=evaluation_size)
            
            self.sess.run(
                self.train_op,
                feed_dict={
                    self.word_embeddings: training_data[0],
                    self.sequence_length: training_data[1],
                    self.target_embeddings: training_data[2],
                    self.target_sequence_length: training_data[3],
                    self.position_weight: training_data[4],
                    self.labels: training_data[5],
                    self.dropout_placeholder: True
                }
                )
            
            self.update_counter += 1

            self.training_summary = self.sess.run(
                self.merged,
                feed_dict={
                    self.word_embeddings: training_data[0],
                    self.sequence_length: training_data[1],
                    self.target_embeddings: training_data[2],
                    self.target_sequence_length: training_data[3],
                    self.position_weight: training_data[4],
                    self.labels: training_data[5],
                    self.dropout_placeholder: True
                }
            )
            self.evaluating_summary = self.sess.run(
                self.merged,
                feed_dict={
                    self.word_embeddings: evaluating_data[0],
                    self.sequence_length: evaluating_data[1],
                    self.target_embeddings: evaluating_data[2],
                    self.target_sequence_length: evaluating_data[3],
                    self.position_weight: evaluating_data[4],
                    self.labels: evaluating_data[5],
                    self.dropout_placeholder: False
                }
            )

            self.train_writer.add_summary(self.training_summary, self.update_counter)
            self.evaluate_writer.add_summary(self.evaluating_summary, self.update_counter)
        
        else:
            self.sess.run(
                self.train_op,
                feed_dict={
                    self.word_embeddings: sentence_embeddings,
                    self.sequence_length: sentence_length,
                    self.target_embeddings: target_embeddings,
                    self.target_sequence_length: target_length,
                    self.position_weight: pw,
                    self.labels: labels,
                    self.dropout_placeholder: True
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
                    self.dropout_placeholder: True
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
                self.dropout_placeholder: False
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
                self.dropout_placeholder: False
            }
        )

        return acc