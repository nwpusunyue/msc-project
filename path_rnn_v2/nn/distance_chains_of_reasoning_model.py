import pprint

import tensorflow as tf

from path_rnn_v2.nn.base_model import BaseModel
from path_rnn_v2.nn.path_rnn import PathRnn
from path_rnn_v2.util.ops import create_reset_metric


class DistanceChainsOfReasoningModel(BaseModel):

    def __init__(self, model_params, train_params):
        super().__init__()
        self._tensors = {}
        self.train_params = train_params
        self._setup_model(model_params)
        self._setup_training(self.tensors['loss'], **self.train_params)
        self._setup_evaluation()
        self._setup_summaries()
        self._saver = tf.train.Saver()
        self._variables = tf.global_variables()

    def _setup_model(self, params):
        self.max_path_len = params['max_path_len']

        self.target_embedder = params['target_embedder']
        self.target_embedder_params = params['target_embedder_params']
        self.path_encoder_params = params['path_encoder_params']
        self.path_rnn_params = params['path_rnn_params']

        self._setup_placeholders(max_path_len=self.max_path_len)

        # [batch_size, emb_dim]
        target_rel_embd = self.target_embedder.embed_sequence(self.target_rel,
                                                  name='target_relation_embedder',
                                                  **self.target_embedder_params)

        path_rnn = PathRnn()
        # [num_queries_in_batch]
        score = path_rnn.get_output(self.rel_seq, self.seq_len,
                                    self.path_partition, target_rel_embd,
                                    self.path_encoder_params, **self.path_rnn_params)

        prob = tf.sigmoid(score)
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.label, dtype=score.dtype),
                                                    logits=score))

        self._tensors['loss'] = loss
        self._tensors['prob'] = prob
        self._training_variables = tf.trainable_variables()

    def _setup_evaluation(self):
        mean_loss, update_op_loss, reset_op_loss = create_reset_metric(tf.metrics.mean, 'mean_loss',
                                                                       values=self.tensors['loss'])
        self._tensors['mean_loss'] = mean_loss
        self._tensors['update_op_loss'] = update_op_loss
        self._tensors['reset_op_loss'] = reset_op_loss

        acc, update_op_acc, reset_op_acc = create_reset_metric(tf.metrics.accuracy, 'mean_acc', labels=self.label,
                                                               predictions=tf.round(self.tensors['prob']))

        self._tensors['acc'] = acc
        self._tensors['update_op_acc'] = update_op_acc
        self._tensors['reset_op_acc'] = reset_op_acc

        tf.summary.scalar('train_loss', self.tensors['loss'], collections=['summary_train'])
        tf.summary.scalar('test_loss', mean_loss, collections=['summary_test'])
        tf.summary.scalar('train_eval_loss', mean_loss, collections=['summary_train_eval'])
        tf.summary.scalar('test_acc', acc, collections=['summary_test'])
        tf.summary.scalar('train_eval_acc', acc, collections=['summary_train_eval'])

    def _setup_placeholders(self, max_path_len):
        # [batch_size, max_path_len, 1]
        self.rel_seq = tf.placeholder(tf.float32,
                                      shape=[None, max_path_len, 1])
        # [batch_size]
        self.seq_len = tf.placeholder(tf.int64,
                                      shape=[None])
        # [batch_size]
        self.target_rel = tf.placeholder(tf.int64,
                                         shape=[None])

        # [num_queries_in_batch]
        self.path_partition = tf.placeholder(tf.int64,
                                             shape=[None, 2])
        # [num_queries_in_batch]
        self.label = tf.placeholder(tf.int64,
                                    shape=[None])
        self._placeholders = {
            'rel_seq': self.rel_seq,
            'seq_len': self.seq_len,
            'target_rel': self.target_rel,
            'partition': self.path_partition,
            'label': self.label
        }

    def train_step(self, batch, sess, summ_writer=None):
        if summ_writer is not None:
            summ, loss, _ = sess.run([self.tensors['summary_train'],
                                      self.tensors['loss'],
                                      self.tensors['train_op']],
                                     feed_dict=self.convert_to_feed_dict(batch))
            summ_writer.add_summary(summ, tf.train.global_step(sess, self.tensors['global_step']))
        else:
            loss, _ = sess.run([self.tensors['loss'],
                                self.tensors['train_op']],
                               feed_dict=self.convert_to_feed_dict(batch))
        return loss

    def eval_step(self, batch, sess, reset=False, summ_writer=None, summ_collection=None):
        if batch is not None:
            sess.run([self.tensors['update_op_loss'], self.tensors['update_op_acc']],
                     feed_dict=self.convert_to_feed_dict(batch))

        if summ_writer is not None and summ_collection is not None and summ_collection in self.tensors:
            summ, loss, acc = sess.run([self.tensors[summ_collection], self.tensors['mean_loss'], self.tensors['acc']])
            summ_writer.add_summary(summ, tf.train.global_step(sess, self.tensors['global_step']))
        else:
            loss, acc = sess.run([self.tensors['mean_loss'], self.tensors['acc']])

        if reset:
            sess.run([self.tensors['reset_op_loss'], self.tensors['reset_op_acc']])
        return loss, acc

    def predict_step(self, batch, sess):
        return sess.run(self.tensors['prob'],
                        feed_dict=self.convert_to_feed_dict(batch))

    @property
    def params_str(self):
        return ('===============Model parameters=============\n'
                'Max path length: {}\n'
                'Embedder params:\n'
                '{}\n'
                '{}\n'
                'Path RNN params:\n'
                '{}\n'
                '===============Train parameters=============\n'
                '{}\n'
                '============================================\n'.format(self.max_path_len,
                                                                        self.target_embedder.config_str,
                                                                        pprint.pformat(self.target_embedder_params),
                                                                        pprint.pformat(self.path_rnn_params),
                                                                        pprint.pformat(self.train_params)))
