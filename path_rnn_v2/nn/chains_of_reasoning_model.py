import os

import numpy as np
import pandas as pd
import tensorflow as tf

from itertools import chain
from path_rnn_v2.util.batch_generator import PartitionBatchGenerator
from path_rnn_v2.util.embeddings import RandomEmbeddings
from path_rnn_v2.util.ops import create_reset_metric
from path_rnn_v2.nn.base_model import BaseModel
from path_rnn_v2.nn.path_rnn import PathRnn
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
        self._setup_model(model_params)
        self._setup_training(self.tensors['loss'], **train_params)
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
        self
        mean_loss, update_op_loss, reset_op_loss = create_reset_metric(tf.metrics.mean, 'mean_loss',
                                                                       values=self.tensors['loss'])
        acc, update_op_acc, reset_op_acc = create_reset_metric(tf.metrics.accuracy, 'mean_acc', labels=self.label,
                                                               predictions=tf.round(self.tensors['prob']))

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

    def train_step(self, batch, sess):
        return sess.run([self.tensors['loss'],
                         self.tensors['train_op']],
                        feed_dict=self.convert_to_feed_dict(batch))[0]

    def print_params(self):
        print('##########Model parameters##########\n'
              'Max path length: {}\n'
              'Embedder params:\n'
              '{}\n'
              '{}\n'
              'Encoder params:\n'
              '{}\n'
              'Path RNN params:\n'
              '{}\n'
              '####################################\n')


if __name__ == '__main__':
    train, dev = get_dataset(
        pos_path='./chains_of_reasoning_data/_people_person_nationality/parsed/positive_matrix.json',
        neg_path='./chains_of_reasoning_data/_people_person_nationality/parsed/negative_matrix.json',
        dev_path='./chains_of_reasoning_data/_people_person_nationality/parsed/dev_matrix.json')

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
            'aggregator': 'logsumexp',
            'k': None
        }
    }

    train_params = {
        'optimizer': tf.train.AdamOptimizer(learning_rate=1e-3),
        'l2': 0.0,
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
    num_epocs = 1
    steps = 1000
    with tf.train.MonitoredTrainingSession() as sess:

        for i in tqdm(range(steps)):
            batch = train_batch_generator.get_batch()
            batch_loss = model.train_step(batch, sess)
            print(batch_loss)
