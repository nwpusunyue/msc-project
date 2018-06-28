import os
import pprint
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from itertools import chain
from path_rnn_v2.util.batch_generator import PartitionBatchGenerator
from path_rnn_v2.util.embeddings import RandomEmbeddings
from path_rnn_v2.util.ops import create_reset_metric
from path_rnn_v2.nn.base_model import BaseModel
from path_rnn_v2.nn.path_rnn import PathRnn
from sklearn.metrics import average_precision_score
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_dataset(pos_path, neg_path, dev_path):
    pos = pd.read_json(pos_path)
    neg = pd.read_json(neg_path)
    train = pd.concat([pos, neg], axis=0)
    dev = pd.read_json(dev_path)

    return train.reset_index(), dev


def get_numpy_arrays(embd, rel_paths, target_rel, label, max_path_len=None):
    path_partition = np.zeros((len(rel_paths), 2), dtype=int)

    total_paths = np.sum([len(rel_paths) for rel_paths in rel_paths])
    if max_path_len is None:
        max_path_len = np.max([np.max([len(path) for path in paths]) for paths in rel_paths])

    print('Max path len: {}'.format(max_path_len))
    print('Total paths: {}'.format(total_paths))

    path_len = np.zeros([total_paths], dtype=np.int)
    indexed_rel_path = np.zeros([total_paths, max_path_len], dtype=np.int)
    indexed_target_rel = np.zeros([total_paths], dtype=np.int)
    label = np.array(label, dtype=np.int)

    path_idx = 0
    for query_idx, (rels, target) in enumerate(zip(rel_paths,
                                                   target_rel)):
        partition_start = path_idx
        for (rel_path) in rels:
            path_len[path_idx] = len(rel_path)
            indexed_target_rel[path_idx] = embd.get_idx(target)

            for rel_idx, rel in enumerate(rel_path):
                if rel_idx < max_path_len:
                    indexed_rel_path[path_idx, rel_idx] = embd.get_idx(rel)
            path_idx += 1
        partition_end = path_idx
        path_partition[query_idx, 0] = partition_start
        path_partition[query_idx, 1] = partition_end

    return (path_partition, path_len, indexed_rel_path, indexed_target_rel, label)


class ChainsOfReasoningModel(BaseModel):

    def __init__(self, model_params, train_params):
        super().__init__()
        self._tensors = {}
        self.train_params = train_params
        self._setup_model(model_params)
        self._setup_training(self.tensors['loss'], **self.train_params)
        self._setup_evaluation()
        self._setup_summaries()
        self._variables = tf.global_variables()

    def _setup_model(self, params):
        self.max_path_len = params['max_path_len']

        self.relation_embedder = params['relation_embedder']
        self.embedder_params = params['embedder_params']
        self.encoder_params = params['encoder_params']
        self.path_rnn_params = params['path_rnn_params']

        self._setup_placeholders(max_path_len=self.max_path_len)

        # [batch_size, max_path_len, emb_dim]
        rel_seq_embd = self.relation_embedder.embed_sequence(self.rel_seq,
                                                             name='relation_embedder',
                                                             **self.embedder_params)
        # [batch_size, emb_dim]
        target_rel_embd = self.relation_embedder.embed_sequence(self.target_rel,
                                                                name='relation_embedder', reuse=True,
                                                                **self.embedder_params)

        path_rnn = PathRnn()
        # [num_queries_in_batch]
        score = path_rnn.get_output(rel_seq_embd, self.seq_len,
                                    self.path_partition, target_rel_embd,
                                    self.encoder_params, **self.path_rnn_params)

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
        tf.summary.scalar('test_acc', acc, collections=['summary_test'])
        tf.summary.scalar('train_eval_loss', mean_loss, collections=['summary_train_eval'])
        tf.summary.scalar('train_eval_acc', acc, collections=['summary_train_eval'])

    def _setup_placeholders(self, max_path_len):
        # [batch_size, max_path_len]
        self.rel_seq = tf.placeholder(tf.int64,
                                      shape=[None, max_path_len])
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
                'Encoder params:\n'
                '{}\n'
                'Path RNN params:\n'
                '{}\n'
                '===============Train parameters=============\n'
                '{}\n'
                '============================================\n'.format(self.max_path_len,
                                                                        self.relation_embedder.config_str,
                                                                        pprint.pformat(self.embedder_params),
                                                                        pprint.pformat(self.encoder_params),
                                                                        pprint.pformat(self.path_rnn_params),
                                                                        pprint.pformat(self.train_params)))


if __name__ == '__main__':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list='0',
                                                      per_process_gpu_memory_fraction=0.33))
    relation = '_people_person_nationality'
    train, dev = get_dataset(
        pos_path='./chains_of_reasoning_data/{}/parsed/positive_matrix.json'.format(relation),
        neg_path='./chains_of_reasoning_data/{}/parsed/negative_matrix.json'.format(relation),
        dev_path='./chains_of_reasoning_data/{}/parsed/dev_matrix.json'.format(relation))

    relation_vocab = set(chain.from_iterable(list(chain.from_iterable(train['relation_paths']))))
    relation_vocab.add('#UNK')
    relation_vocab.add('#END')
    embd_dim = 50
    random_embd = RandomEmbeddings(relation_vocab, embd_dim, 'rel_embd', '#UNK', tf.initializers.random_normal())
    (train_path_partition,
     train_path_len,
     train_indexed_rel_path,
     train_indexed_target_rel,
     train_label) = get_numpy_arrays(embd=random_embd,
                                     rel_paths=train[
                                         'relation_paths'],
                                     target_rel=train[
                                         'target_relation'],
                                     label=train['label'])
    (dev_path_partition,
     dev_path_len,
     dev_indexed_rel_path,
     dev_indexed_target_rel,
     dev_label) = get_numpy_arrays(embd=random_embd,
                                   rel_paths=dev[
                                       'relation_paths'],
                                   target_rel=dev[
                                       'target_relation'],
                                   label=dev['label'],
                                   max_path_len=train_indexed_rel_path.shape[1])
    model_params = {
        'max_path_len': train_indexed_rel_path.shape[1],
        'relation_embedder': random_embd,
        'embedder_params': {
            'max_norm': None,
            'with_projection': False,
            'projection_activation': None,
            'projection_dim': None
        },
        'encoder_params': {
            'repr_dim': embd_dim,
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

    model = ChainsOfReasoningModel(model_params=model_params, train_params=train_params)
    train_batch_generator = PartitionBatchGenerator(partition=train_path_partition,
                                                    label=train_label,
                                                    tensor_dict={
                                                        'rel_seq': train_indexed_rel_path,
                                                        'seq_len': train_path_len,
                                                        'target_rel': train_indexed_target_rel
                                                    },
                                                    batch_size=100)
    train_eval_batch_generator = PartitionBatchGenerator(partition=train_path_partition,
                                                         label=train_label,
                                                         tensor_dict={
                                                             'seq_len': train_path_len,
                                                             'rel_seq': train_indexed_rel_path,
                                                             'target_rel': train_indexed_target_rel
                                                         },
                                                         batch_size=100,
                                                         permute=False)
    dev_batch_generator = PartitionBatchGenerator(partition=dev_path_partition,
                                                  label=dev_label,
                                                  tensor_dict={
                                                      'seq_len': dev_path_len,
                                                      'rel_seq': dev_indexed_rel_path,
                                                      'target_rel': dev_indexed_target_rel
                                                  },
                                                  batch_size=20,
                                                  permute=False)

    steps = 1000
    check_period = 50

    with tf.train.MonitoredTrainingSession(config=config) as sess:
        summ_writer = tf.summary.FileWriter('./chains_of_reasoning_logs/run_{}_{}'.format(relation, time.time()))
        for i in tqdm(range(steps)):
            batch = train_batch_generator.get_batch()
            batch_loss = model.train_step(batch, sess, summ_writer=summ_writer)
            if i % check_period == 0:
                for j in range(train_eval_batch_generator.batch_count):
                    batch = train_eval_batch_generator.get_batch()
                    model.eval_step(batch, sess)

                metrics = model.eval_step(batch=None, sess=sess, reset=True, summ_writer=summ_writer,
                                          summ_collection='summary_train_eval')

                dev_prob = np.array([])
                dev_label = np.array([])

                print('Train loss: {} Train acc: {}'.format(metrics[0], metrics[1]))
                for j in range(dev_batch_generator.batch_count):
                    batch = dev_batch_generator.get_batch()
                    model.eval_step(batch, sess)
                    dev_prob = np.concatenate((dev_prob, model.predict_step(batch, sess)))
                    dev_label = np.concatenate((dev_label, batch['label']))

                ap = average_precision_score(y_true=dev_label, y_score=dev_prob)

                metrics = model.eval_step(batch=None, sess=sess, reset=True, summ_writer=summ_writer,
                                          summ_collection='summary_test')
                print('Dev loss: {} Dev acc: {} Dev ap: {}'.format(metrics[0], metrics[1], ap))
