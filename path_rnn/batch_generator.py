import numpy as np
from itertools import chain


class BatchGenerator:
    TRAIN = 'train'
    TEST = 'test'
    TRAIN_EVAL = 'train_eval'

    def __init__(self,
                 indexed_relation_paths,
                 indexed_entity_paths,
                 indexed_target_relations,
                 path_partitions,
                 path_lengths,
                 num_words,
                 labels,
                 batch_size,
                 test_prop=0.0,
                 test_batch_size=None,
                 train_eval_prop=0.0,
                 train_eval_batch_size=None
                 ):
        self.dataset_size = len(path_partitions)
        print('Dataset size: {}'.format(self.dataset_size))

        self.train_size = int((1 - test_prop) * self.dataset_size)
        self.test_size = self.dataset_size - self.train_size
        self.train_eval_size = int(train_eval_prop * self.train_size)

        self.batch_size = batch_size
        self.train_eval_batch_size = (train_eval_batch_size if train_eval_batch_size is not None
                                      else self.train_eval_size)
        self.test_batch_size = (test_batch_size if test_batch_size is not None
                                else self.test_size)

        train_eval_idxs = np.random.choice(np.arange(0,
                                                     self.train_size),
                                           size=self.train_eval_size,
                                           replace=False)
        train_idxs_generator = IndexGenerator(np.arange(0,
                                                        self.train_size),
                                              self.batch_size,
                                              permute=True
                                              )
        train_eval_idxs_generator = IndexGenerator(train_eval_idxs,
                                                   self.train_eval_batch_size,
                                                   permute=False)
        test_idxs_generator = IndexGenerator(np.arange(self.train_size,
                                                       self.dataset_size),
                                             self.test_batch_size,
                                             permute=False)

        self.idx_generators = {
            BatchGenerator.TRAIN: train_idxs_generator,
            BatchGenerator.TEST: test_idxs_generator,
            BatchGenerator.TRAIN_EVAL: train_eval_idxs_generator
        }

        (self.indexed_relation_paths,
         self.indexed_entity_paths,
         self.indexed_target_relations,
         self.path_partitions,
         self.path_lengths,
         self.num_words,
         self.labels) = (indexed_relation_paths,
                         indexed_entity_paths,
                         indexed_target_relations,
                         path_partitions,
                         path_lengths,
                         num_words,
                         labels)

        print('Train queries: {} in {} batches,\n'
              'Test queries: {} in {} batches,\n'
              'Train evaluation queries: {} in {} batches\n'.format(self.train_size,
                                                                  train_idxs_generator.total_batches,
                                                                  self.test_size,
                                                                  test_idxs_generator.total_batches,
                                                                  self.train_eval_size,
                                                                  train_eval_idxs_generator.total_batches))

    def get_batch(self, batch_type, debug=False):
        idxs = self.idx_generators[batch_type].get_batch_idxs(debug)
        return self._get_data(idxs)

    def get_batch_count(self, batch_type):
        return self.idx_generators[batch_type].total_batches

    def get_epochs_completed(self, batch_type):
        return self.idx_generators[batch_type].epochs_completed

    def _get_data(self, idxs):
        path_ranges = self.path_partitions[idxs]
        path_idxs = list(chain.from_iterable([list(range(path_ranges[i, 0], path_ranges[i, 1]))
                                              for i in range(len(idxs))]))

        rel_seq = self.indexed_relation_paths[path_idxs]
        ent_seq = self.indexed_entity_paths[path_idxs]
        target_relations = self.indexed_target_relations[path_idxs]
        partitions = self._generate_partition(path_ranges)
        path_lengths = self.path_lengths[path_idxs]
        num_words = self.num_words[path_idxs]
        labels = self.labels[idxs]

        return (rel_seq,
                ent_seq,
                target_relations,
                partitions,
                path_lengths,
                num_words,
                labels)

    def _generate_partition(self, path_ranges):
        partition = np.zeros(shape=path_ranges.shape)
        start = 0
        for idx, path_range in enumerate(path_ranges):
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

    def get_batch_idxs(self, debug):
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
