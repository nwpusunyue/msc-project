import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from itertools import chain
from parsing.special_tokens import *
from path_rnn_v2.util.embeddings import RandomEmbeddings, Word2VecEmbeddings
from path_rnn_v2.util.tensor_generator import get_medhop_tensors


class ProportionedPartitionBatchGenerator:

    def __init__(self,
                 partition,
                 label,
                 tensor_dict,
                 batch_size,
                 positive_prop,
                 permute=True):
        '''

        :param path_partition: np array of shape [num_part, 2], indicates start and end of each partition
        :param label: np array of shape [num_part], label for each example
        :param tensor_dict: dict of np arrays of shape [num_instances, ...], where
        each instance is associated with a partition.
        :param batch_size: batch size to generate
        '''
        self.size = partition.shape[0]
        self.batch_size = batch_size
        self.positive_prop = positive_prop

        self.partition = partition
        self.label = label
        self.tensor_dict = tensor_dict

        self.positive_idx = np.argwhere(self.label == 1).reshape((-1,))
        self.negative_idx = np.argwhere(self.label == 0).reshape((-1))

        self.positive_batch_size = int(self.positive_prop * self.batch_size)
        self.negative_batch_size = self.batch_size - self.positive_batch_size

        self.positive_idxs_generator = IndexGenerator(self.positive_idx,
                                                      self.positive_batch_size,
                                                      permute=permute)
        self.negative_ixs_generator = IndexGenerator(self.negative_idx,
                                                     self.negative_batch_size,
                                                     permute=permute)

        self.batch_count = max(self.positive_idxs_generator.total_batches, self.negative_ixs_generator.total_batches)

    def get_batch(self, debug=False):
        '''

        :param debug: will print start and end index of the batch and the retrieved ids
        :return: (batch_partition_ranges [batch_size, 2],
                  batch_tensor_dict,
                  labels [batch_size]
        '''
        positive = self.positive_idxs_generator.get_batch_idxs(debug)
        negative = self.negative_ixs_generator.get_batch_idxs(debug)
        idxs = np.random.permutation(np.concatenate((positive, negative), axis=0))
        if debug:
            print(idxs)
        return self._get_data(idxs)

    @property
    def epochs_completed(self):
        return min(self.positive_idxs_generator.epochs_completed, self.negative_ixs_generator.epochs_completed)

    def _get_data(self, idxs):
        ranges = self.partition[idxs]
        labels = self.label[idxs]
        tensor_idxs = list(chain.from_iterable([list(range(ranges[i, 0], ranges[i, 1]))
                                                for i in range(len(idxs))]))

        batch_tensor_dict = {'partition': self._generate_partition(ranges),
                             'label': labels}
        for key, tensor in self.tensor_dict.items():
            batch_tensor_dict[key] = tensor[tensor_idxs]

        return batch_tensor_dict

    def _generate_partition(self, ranges):
        partition = np.zeros(shape=ranges.shape)
        start = 0
        for idx, path_range in enumerate(ranges):
            partition[idx, 0] = start
            len = path_range[1] - path_range[0]
            partition[idx, 1] = start + len
            start += len
        return partition


class ProportionedSubsamplingPartitionBatchGenerator:

    def __init__(self,
                 partition,
                 label,
                 tensor_dict,
                 batch_size,
                 positive_prop,
                 partition_limit,
                 permute=True):
        '''

        :param path_partition: np array of shape [num_part, 2], indicates start and end of each partition
        :param label: np array of shape [num_part], label for each example
        :param tensor_dict: dict of np arrays of shape [num_instances, ...], where
        each instance is associated with a partition.
        :param batch_size: batch size to generate
        :param positive_prop: proportion of elements in each batch that should pe positive
        :param partition_limit: how many elements should a partition contain at most.
        '''
        print('Partition limit: {}'.format(partition_limit))
        self.size = partition.shape[0]
        self.batch_size = batch_size
        self.positive_prop = positive_prop
        self.partition_limit = partition_limit

        self.partition = partition
        self.label = label
        self.tensor_dict = tensor_dict

        self.positive_idx = np.argwhere(self.label == 1).reshape((-1,))
        self.negative_idx = np.argwhere(self.label == 0).reshape((-1))

        self.positive_batch_size = int(self.positive_prop * self.batch_size)
        self.negative_batch_size = self.batch_size - self.positive_batch_size

        self.positive_idxs_generator = IndexGenerator(self.positive_idx,
                                                      self.positive_batch_size,
                                                      permute=permute)
        self.negative_ixs_generator = IndexGenerator(self.negative_idx,
                                                     self.negative_batch_size,
                                                     permute=permute)

        self.batch_count = max(self.positive_idxs_generator.total_batches, self.negative_ixs_generator.total_batches)

    def get_batch(self, debug=False):
        '''

        :param debug: will print start and end index of the batch and the retrieved ids
        :return: (batch_partition_ranges [batch_size, 2],
                  batch_tensor_dict,
                  labels [batch_size]
        '''
        positive = self.positive_idxs_generator.get_batch_idxs(debug)
        negative = self.negative_ixs_generator.get_batch_idxs(debug)
        idxs = np.random.permutation(np.concatenate((positive, negative), axis=0))
        if debug:
            print(idxs)
        return self._get_data(idxs)

    @property
    def epochs_completed(self):
        return min(self.positive_idxs_generator.epochs_completed, self.negative_ixs_generator.epochs_completed)

    def _get_data(self, idxs):
        subsampled_idxs = self._subsample_partitions(idxs)
        labels = self.label[idxs]
        tensor_idxs = list(chain.from_iterable(subsampled_idxs))
        batch_tensor_dict = {'partition': self._generate_partition(subsampled_idxs),
                             'label': labels}
        for key, tensor in self.tensor_dict.items():
            batch_tensor_dict[key] = tensor[tensor_idxs]

        return batch_tensor_dict

    def _subsample_partitions(self, idxs):
        subsampled_idxs = []
        for idx in idxs:
            rng = self.partition[idx]
            if rng[1] - rng[0] > self.partition_limit:
                subsampled_idxs.append(np.random.choice(range(rng[0], rng[1]), self.partition_limit, replace=False))
            else:
                subsampled_idxs.append(np.array(range(rng[0], rng[1])))
        return subsampled_idxs

    def _generate_partition(self, ranges):
        partition = np.zeros(shape=[len(ranges), 2])
        start = 0
        for idx, path_range in enumerate(ranges):
            partition[idx, 0] = start
            length = path_range.shape[0]
            partition[idx, 1] = start + length
            start += length
        return partition


class PartitionBatchGenerator:

    def __init__(self,
                 partition,
                 label,
                 tensor_dict,
                 batch_size,
                 permute=True):
        '''

        :param path_partition: np array of shape [num_part, 2], indicates start and end of each partition
        :param label: np array of shape [num_part], label for each example
        :param tensor_dict: dict of np arrays of shape [num_instances, ...], where
        each instance is associated with a partition.
        :param batch_size: batch size to generate
        '''
        self.size = partition.shape[0]
        self.batch_size = batch_size

        self.partition = partition
        self.label = label
        self.tensor_dict = tensor_dict

        self.idxs_generator = IndexGenerator(np.arange(0, self.size),
                                             self.batch_size,
                                             permute=permute)
        self.batch_count = self.idxs_generator.total_batches

    def get_batch(self, debug=False):
        '''

        :param debug: will print start and end index of the batch and the retrieved ids
        :return: (batch_partition_ranges [batch_size, 2],
                  batch_tensor_dict,
                  labels [batch_size]
        '''
        idxs = self.idxs_generator.get_batch_idxs(debug)
        return self._get_data(idxs)

    @property
    def epochs_completed(self):
        return self.idxs_generator.epochs_completed

    def _get_data(self, idxs):
        if self.label is not None:
            ranges = self.partition[idxs]
            labels = self.label[idxs]
            tensor_idxs = list(chain.from_iterable([list(range(ranges[i, 0], ranges[i, 1]))
                                                    for i in range(len(idxs))]))

            batch_tensor_dict = {'partition': self._generate_partition(ranges),
                                 'label': labels}
        else:
            ranges = self.partition[idxs]
            tensor_idxs = list(chain.from_iterable([list(range(ranges[i, 0], ranges[i, 1]))
                                                    for i in range(len(idxs))]))

            batch_tensor_dict = {'partition': self._generate_partition(ranges)}
        for key, tensor in self.tensor_dict.items():
            batch_tensor_dict[key] = tensor[tensor_idxs]

        return batch_tensor_dict

    def _generate_partition(self, ranges):
        partition = np.zeros(shape=ranges.shape)
        start = 0
        for idx, path_range in enumerate(ranges):
            partition[idx, 0] = start
            len = path_range[1] - path_range[0]
            partition[idx, 1] = start + len
            start += len
        return partition


class IndexGenerator:
    def __init__(self,
                 idxs,
                 batch_size,
                 permute):
        self.idxs = idxs
        self.batch_size = batch_size
        self.total_batches = int(np.ceil(len(self.idxs) / self.batch_size))
        self.permute = permute
        self.epochs_completed = 0
        self.current_batch = 0

        if self.permute:
            self.idxs = np.random.permutation(idxs)

    def get_batch_idxs(self, debug=False):
        start = self.current_batch * self.batch_size
        end = np.min([(self.current_batch + 1) * self.batch_size, len(self.idxs)])

        batch_idxs = self.idxs[start:end]

        if debug:
            print(start, end, batch_idxs)

        self.current_batch += 1

        if self.current_batch == self.total_batches:
            if self.permute:
                self.idxs = np.random.permutation(self.idxs)
            self.current_batch = 0
            self.epochs_completed += 1

        return batch_idxs


if __name__ == '__main__':
    train = pd.read_json(
        './data/sentwise=F_cutoff=4_limit=100_method=shortest_tokenizer=punkt_medhop_train_mini.json')
    train_document_store = pickle.load(open('./data/train_mini_doc_store_punkt.pickle', 'rb'))
    word2vec_embeddings = Word2VecEmbeddings('./medhop_word2vec_punkt',
                                             name='token_embd',
                                             unk_token=UNK,
                                             trainable=False,
                                             special_tokens=[(ENT_1, False),
                                                             (ENT_2, False),
                                                             (ENT_X, False),
                                                             (UNK, False),
                                                             (END, False),
                                                             (PAD, True)])
    target_embeddings = RandomEmbeddings(train['relation'],
                                         name='target_rel_emb',
                                         embedding_size=100,
                                         unk_token=None,
                                         initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                     stddev=1.0,
                                                                                     dtype=tf.float64))

    train_tensors = get_medhop_tensors(train['relation_paths'],
                                       train['entity_paths'],
                                       train['relation'],
                                       train['label'],
                                       train_document_store,
                                       word2vec_embeddings,
                                       word2vec_embeddings,
                                       target_embeddings,
                                       max_path_len=5,
                                       max_rel_len=600,
                                       max_ent_len=1,
                                       rel_retrieve_params={
                                           'replacement': (ENT_1, ENT_2),
                                           'truncate': False
                                       },
                                       ent_retrieve_params={
                                           'neighb_size': 0
                                       })
    (rel_seq, ent_seq, path_len, rel_len, ent_len, target_rel, partition, label) = train_tensors

    pos = len(np.argwhere(label == 1))
    neg = len(np.argwhere(label == 0))
    positive_prop = float(pos) / neg
    batch_size = 10

    print('Positive:', pos)
    print('Negative:', neg)
    print('Proportion:', positive_prop)

    batch_gen = ProportionedPartitionBatchGenerator(partition,
                                                    label,
                                                    tensor_dict={'rel_seq': rel_seq,
                                                                 'ent_seq': ent_seq,
                                                                 'seq_len': path_len,
                                                                 'rel_len': rel_len,
                                                                 'ent_len': ent_len,
                                                                 'target_rel': target_rel},
                                                    batch_size=batch_size,
                                                    positive_prop=positive_prop,
                                                    permute=True)
    print(batch_gen.positive_idx)
    print(batch_gen.negative_idx)

    assert int(batch_size * positive_prop) == batch_gen.positive_batch_size
    assert (batch_size - int(batch_size * positive_prop)) == batch_gen.negative_batch_size
    while batch_gen.epochs_completed < 3:
        batch_gen.get_batch(debug=True)

    batch_gen = ProportionedPartitionBatchGenerator(partition,
                                                    label,
                                                    tensor_dict={'rel_seq': rel_seq,
                                                                 'ent_seq': ent_seq,
                                                                 'seq_len': path_len,
                                                                 'rel_len': rel_len,
                                                                 'ent_len': ent_len,
                                                                 'target_rel': target_rel},
                                                    batch_size=batch_size,
                                                    positive_prop=0.2,
                                                    permute=True)
    print(batch_gen.positive_idx)
    print(batch_gen.negative_idx)

    while batch_gen.epochs_completed < 3:
        batch_gen.get_batch(debug=True)

    print('\n\nSubsampling batch generator')
    train = pd.read_json('./data/sentwise=F_cutoff=4_limit=500_method=all_tokenizer=punkt_medhop_train_mini.json')
    batch_gen = ProportionedSubsamplingPartitionBatchGenerator(partition,
                                                               label,
                                                               tensor_dict={'rel_seq': rel_seq,
                                                                            'ent_seq': ent_seq,
                                                                            'seq_len': path_len,
                                                                            'rel_len': rel_len,
                                                                            'ent_len': ent_len,
                                                                            'target_rel': target_rel},
                                                               batch_size=batch_size,
                                                               positive_prop=0.2,
                                                               partition_limit=3,
                                                               permute=True)
    while batch_gen.epochs_completed < 3:
        batch_gen.get_batch(debug=True)
