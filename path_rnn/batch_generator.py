import numpy as np
from itertools import chain
from path_rnn.tensor_generator_unstacked import get_indexed_paths, get_labels


class BatchGenerator:

    def __init__(self,
                 dataset,
                 relation_token_vocab,
                 entity_vocab,
                 target_relation_vocab,
                 max_path_length,
                 max_relation_length,
                 batch_size,
                 test_prop=0.0
                 ):
        self.dataset = dataset
        self.relation_token_vocab = relation_token_vocab
        self.entity_vocab = entity_vocab
        self.target_relation_vocab = target_relation_vocab
        self.max_path_length = max_path_length
        self.max_relation_length = max_relation_length

        self.batch_size = batch_size
        self.train_size = int((1 - test_prop) * len(dataset))
        self.train_idxs_generator = IndexGenerator(np.arange(0,
                                                             self.train_size),
                                                   self.batch_size,
                                                   permute=True
                                                   )
        self.test_idxs_generator = IndexGenerator(np.arange(self.train_size,
                                                            len(self.dataset)),
                                                  self.batch_size,
                                                  permute=True)
        self.train_set_size = self.train_idxs_generator.size
        self.test_set_size = self.test_idxs_generator.size

        (self.indexed_relation_paths,
         self.indexed_entity_paths,
         self.indexed_target_relations,
         self.path_partitions,
         self.path_lengths,
         self.num_words) = get_indexed_paths(q_relation_paths=self.dataset['relation_paths'],
                                             q_entity_paths=self.dataset['entity_paths'],
                                             target_relations=self.dataset['relation'],
                                             relation_token_vocab=self.relation_token_vocab,
                                             entity_vocab=self.entity_vocab,
                                             target_relation_vocab=self.target_relation_vocab,
                                             max_path_length=self.max_path_length,
                                             max_relation_length=self.max_relation_length)
        self.labels = get_labels(self.dataset['label'])
        print('Train queries: {}, Test queries: {}'.format(self.train_set_size, self.test_set_size))

    def get_batch(self, test=False):
        if test:
            idxs = self.test_idxs_generator.get_batch_idxs()
        else:
            idxs = self.train_idxs_generator.get_batch_idxs()

        path_ranges = self.path_partitions[idxs]
        path_idxs = list(chain.from_iterable([list(range(path_ranges[i, 0], path_ranges[i, 1]))
                                              for i in range(self.batch_size)]))

        batch_rel_seq = self.indexed_relation_paths[path_idxs]
        batch_ent_seq = self.indexed_entity_paths[path_idxs]
        batch_target_relations = self.indexed_target_relations[path_idxs]
        batch_partitions = self._generate_partition(path_ranges)
        batch_path_lengths = self.path_lengths[path_idxs]
        batch_num_words = self.num_words[path_idxs]
        batch_labels = self.labels[idxs]

        return (batch_rel_seq,
                batch_ent_seq,
                batch_target_relations,
                batch_partitions,
                batch_path_lengths,
                batch_num_words,
                batch_labels)

    def _generate_partition(self, path_ranges):
        total_batch_size = sum([path_ranges[i][1] - path_ranges[i][0] for i in range(self.batch_size)])
        batch_partition = np.zeros([total_batch_size], dtype=int)
        idx = 0
        pos = 0
        for path_range in path_ranges:
            start = pos
            end = pos + (path_range[1] - path_range[0])
            batch_partition[start:end] = idx
            idx += 1
            pos = end
        return batch_partition


class IndexGenerator:

    def __init__(self,
                 idxs,
                 batch_size,
                 permute):
        self.idxs = idxs
        self.batch_size = batch_size
        self.total_batches = np.floor(len(self.idxs) / self.batch_size)
        self.size = self.batch_size * self.total_batches
        self.permute = permute
        self.epochs_completed = 0
        self.current_batch = 0

        if self.permute:
            self.idxs = np.random.permutation(idxs)

    def get_batch_idxs(self):
        if self.current_batch < self.total_batches:
            batch_idxs = self.idxs[self.current_batch * self.batch_size:(self.current_batch + 1) * self.batch_size]
            self.current_batch += 1
        else:
            if self.permute:
                self.idxs = np.random.permutation(self.idxs)
            self.current_batch = 0
            self.epochs_completed += 1
            batch_idxs = self.idxs[self.current_batch * self.batch_size:(self.current_batch + 1) * self.batch_size]

        return batch_idxs
