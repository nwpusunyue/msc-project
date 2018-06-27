import numpy as np

from itertools import chain


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
