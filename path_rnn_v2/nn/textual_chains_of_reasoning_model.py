import pprint

import tensorflow as tf

from path_rnn_v2.nn.base_model import BaseModel
from path_rnn_v2.nn.path_rnn import PathRnn
from path_rnn_v2.nn.textual_path_elem_encoder import encode_path_elem
from path_rnn_v2.util.embeddings import RandomEmbeddings
from path_rnn_v2.util.ops import create_reset_metric
from path_rnn_v2.util.ops import rank_loss


class TextualChainsOfReasoningModel(BaseModel):

    def __init__(self, model_params, train_params):
        super().__init__()
        self._tensors = {}
        self.train_params = train_params
        self._setup_model(model_params)
        self._setup_training(self.tensors['loss'], **self.train_params)
        self._setup_evaluation()
        self._setup_summaries()
        self._variables = tf.global_variables()
        self._saver = tf.train.Saver([var for var in self.train_variables])

    def _setup_model(self, model_params):
        self.rel_only = model_params['rel_only'] if 'rel_only' in model_params else False
        self.use_rank_loss = model_params['use_rank_loss'] if 'use_rank_loss' in model_params else False
        self.max_path_len = model_params['max_path_len']
        self.label_smoothing = model_params['label_smoothing'] if 'label_smoothing' in model_params else None

        self.max_rel_len = model_params['max_rel_len']
        self.rel_embedder = model_params['relation_embedder']
        self.rel_embedder_params = model_params['relation_embedder_params']
        self.rel_encoder_params = model_params['relation_encoder_params']
        self.rel_encoder_name = 'rel_seq_encoder' if 'rel_encoder_name' not in model_params else model_params[
            'rel_encoder_name']

        if self.use_rank_loss:
            self.margin = model_params['margin'] if 'margin' in model_params else 0.0
        else:
            self.margin = None

        if not self.rel_only:
            self.max_ent_len = model_params['max_ent_len']
            self.ent_embedder = model_params['entity_embedder']
            self.ent_embedder_params = model_params['entity_embedder_params']
            self.ent_encoder_params = model_params['entity_encoder_params']
            self.ent_encoder_name = 'ent_seq_encoder' if 'ent_encoder_name' not in model_params else model_params[
                'ent_encoder_name']

        self.target_rel_embedder = model_params['target_embedder']
        self.target_rel_embedder_params = model_params['target_embedder_params']

        self.path_encoder_params = model_params['path_encoder_params']
        self.path_rnn_params = model_params['path_rnn_params']

        self._setup_placeholders()

        # encode rel seq and ent seq
        # [batch_size, max_path_len, rel_repr_dim]
        rel_seq_enc = encode_path_elem(self.rel_seq, self.rel_len, self.rel_embedder, self.rel_embedder_params,
                                       self.rel_encoder_params, is_eval=self.is_eval,
                                       name=self.rel_encoder_name)

        if not self.rel_only:
            # [batch_size, max_path_len, ent_repr_dim]
            ent_seq_enc = encode_path_elem(self.ent_seq, self.ent_len, self.ent_embedder, self.ent_embedder_params,
                                           self.ent_encoder_params, is_eval=self.is_eval,
                                           name=self.ent_encoder_name)
        else:
            ent_seq_enc = None

        # [batch_size, target_rel_repr_dim]
        target_rel_enc = self.target_rel_embedder.embed_sequence(self.target_rel,
                                                                 **self.target_rel_embedder_params)

        # encode paths and compute scores
        path_rnn = PathRnn()
        score = path_rnn.get_output(rel_seq=rel_seq_enc,
                                    ent_seq=ent_seq_enc,
                                    seq_len=self.seq_len,
                                    target_rel=target_rel_enc,
                                    path_partitions=self.partition,
                                    encoder_params=self.path_encoder_params,
                                    is_eval=self.is_eval,
                                    **self.path_rnn_params)

        prob = tf.sigmoid(score)
        if not self.use_rank_loss:
            if self.label_smoothing is not None:
                # [num_queries]
                smoothed_label = tf.cast(self.label, dtype=score.dtype) * (
                        1 - self.label_smoothing) + 0.5 * self.label_smoothing
            else:
                smoothed_label = tf.cast(self.label, dtype=score.dtype)

            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=smoothed_label,
                                                        logits=score))
        else:
            loss = rank_loss(y_true=self.label, y_pred=prob, margin=self.margin)
            #loss = tf.losses.hinge_loss(labels=self.label, logits=score)

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

    def _setup_placeholders(self):
        # [batch_size]
        self.seq_len = tf.placeholder(tf.int32,
                                      shape=[None])

        # [batch_size, max_path_len, max_rel_len]
        self.rel_seq = tf.placeholder(tf.int32,
                                      shape=[None, self.max_path_len, self.max_rel_len])
        # [batch_size, max_path_len]
        self.rel_len = tf.placeholder(tf.int32,
                                      shape=[None, self.max_path_len])

        if not self.rel_only:
            # [batch_size, max_path_len, max_ent_len]
            self.ent_seq = tf.placeholder(tf.int32,
                                          shape=[None, self.max_path_len, self.max_ent_len])
            # [batch_size, max_path_len]
            self.ent_len = tf.placeholder(tf.int32,
                                          shape=[None, self.max_path_len])

        # [num_queries_in_batch]
        self.target_rel = tf.placeholder(tf.int32,
                                         shape=[None])
        # [num_queries_in_batch]
        self.partition = tf.placeholder(tf.int32,
                                        shape=[None, 2])
        # [num_queries_in_batch]
        self.label = tf.placeholder(tf.int32,
                                    shape=[None])

        # bool
        self.is_eval = tf.placeholder(tf.bool,
                                      shape=())

        self._placeholders = {'rel_seq': self.rel_seq,
                              'seq_len': self.seq_len,
                              'rel_len': self.rel_len,
                              'target_rel': self.target_rel,
                              'partition': self.partition,
                              'label': self.label,
                              'is_eval': self.is_eval}
        if not self.rel_only:
            self._placeholders.update({
                'ent_seq': self.ent_seq,
                'ent_len': self.ent_len,
            })

    @property
    def params_str(self):
        return ('===============Model parameters=============\n'
                'Max path length: {}\n'
                'Label smoothing: {}\n'
                '{}\n'
                '{}\n'
                'Path RNN params:\n'
                '{}\n'
                '{}\n'
                'Use rank loss: {}\n'
                'Rank loss margin: {}\n'
                '===============Train parameters=============\n'
                '{}\n'
                '============================================\n'.format(self.max_path_len,
                                                                        self.label_smoothing,
                                                                        self._rel_params_str(),
                                                                        self._ent_params_str(),
                                                                        pprint.pformat(self.path_encoder_params),
                                                                        pprint.pformat(self.path_rnn_params),
                                                                        self.use_rank_loss,
                                                                        self.margin,
                                                                        pprint.pformat(self.train_params)))

    def _rel_params_str(self):
        return ('Max relation length: {}\n'
                'Relation Embedder params:\n'
                '{}\n'
                '{}\n'
                'Relation Encoder params:\n'
                '{}\n').format(self.max_rel_len,
                               self.rel_embedder.config_str,
                               pprint.pformat(self.rel_embedder_params),
                               pprint.pformat(self.rel_encoder_params))

    def _ent_params_str(self):
        if not self.rel_only:
            return ('Max entity length: {}\n'
                    'Entity Embedder params:\n'
                    '{}\n'
                    '{}\n'
                    'Entity Encoder params:\n'
                    '{}\n').format(self.max_rel_len,
                                   self.rel_embedder.config_str,
                                   pprint.pformat(self.ent_embedder_params),
                                   pprint.pformat(self.ent_encoder_params))
        else:
            return 'Relation only: True\n'

    def train_step(self, batch, sess, summ_writer=None):
        batch['is_eval'] = False
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
            batch['is_eval'] = True
            sess.run([self.tensors['update_op_loss'],
                      self.tensors['update_op_acc']],
                     feed_dict=self.convert_to_feed_dict(batch))

        if summ_writer is not None and summ_collection is not None and summ_collection in self.tensors:
            summ, loss, acc = sess.run(
                [self.tensors[summ_collection], self.tensors['mean_loss'], self.tensors['acc']])
            summ_writer.add_summary(summ, tf.train.global_step(sess, self.tensors['global_step']))
        else:
            loss, acc = sess.run([self.tensors['mean_loss'], self.tensors['acc']])

        if reset:
            sess.run([self.tensors['reset_op_loss'], self.tensors['reset_op_acc']])
        return loss

    def predict_step(self, batch, sess):
        batch['is_eval'] = True
        return sess.run(self.tensors['prob'],
                        feed_dict=self.convert_to_feed_dict(batch))


if __name__ == '__main__':
    emb_dim = 50
    rel_ent_embedder = RandomEmbeddings(tokens=['a', 'b', 'c', 'd'], embedding_size=emb_dim, name='random_emb',
                                        unk_token='a', initializer=tf.initializers.random_normal())
    target_embedder = RandomEmbeddings(tokens=['interacts'], embedding_size=emb_dim, name='target_random_emb',
                                       unk_token='interacts', initializer=tf.initializers.random_normal())

    model_params = {
        'max_path_len': 5,
        'max_rel_len': 10,
        'max_ent_len': 1,
        'relation_embedder': rel_ent_embedder,
        'relation_embedder_params': {
            'max_norm': None,
            'with_projection': True,
            'projection_activation': None,
            'projection_dim': emb_dim,
            'name': 'relation_entity_embedder',
            'reuse': False
        },
        'relation_encoder_params': {
            'module': 'average',
            'activation': None,
            'dropout': None,
        },
        'entity_embedder': rel_ent_embedder,
        'entity_embedder_params': {
            'max_norm': None,
            'with_projection': True,
            'projection_activation': None,
            'projection_dim': emb_dim,
            'name': 'relation_entity_embedder',
            'reuse': True
        },
        'entity_encoder_params': {
            'module': 'identity',
            'activation': None,
            'dropout': None
        },
        'target_embedder': target_embedder,
        'target_embedder_params': {
            'max_norm': None,
            'with_projection': True,
            'projection_activation': None,
            'projection_dim': None,
            'name': 'target_relation_embedder',
            'reuse': False
        },
        'path_encoder_params': {
            'repr_dim': emb_dim,
            'module': 'lstm',
            'activation': None,
            'dropout': None,
            'extra_args': {
                'with_backward': False,
                'with_projection': False
            }
        },
        'path_rnn_params': {
            'aggregator': 'max',
            'k': None
        }
    }

    train_params = {
        'optimizer': tf.train.AdamOptimizer(learning_rate=1e-3),
        'l2': 0.0001,
        'clip_op': None,
        'clip': None
    }

    model = TextualChainsOfReasoningModel(model_params, train_params)
    print(model.params_str)
