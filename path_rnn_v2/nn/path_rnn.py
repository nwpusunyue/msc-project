import tensorflow as tf

from path_rnn_v2.util.sequence_encoder import encoder


class PathRnn:

    def aggregate_scores(self, scores, aggregator='logsumexp', k=1):
        '''

        :param scores: 1D tensor with scores to aggregate
        :return: a scalar - the agg score
        '''
        if aggregator == 'logsumexp':
            return tf.reduce_logsumexp(scores)
        elif aggregator == 'max':
            return tf.reduce_max(scores)
        elif aggregator == 'average':
            return tf.reduce_mean(scores)
        elif aggregator == 'topk':
            return tf.reduce_mean(tf.nn.top_k(scores, k=k)[0])

    def encode_path(self, rel_seq, seq_len, encoder_params, ent_seq=None):
        '''

        :param rel_seq: [batch_size, max_path_length, rel_repr_dim]
        :param seq_len: [batch_size]
        :param encoder_params: dict of params to pass to the sequence encoder
        :param ent_seq: if not None, then each element in the path will be formed from the concatenation of
        the entity_repr and relation_repr. The ent_seq  must have same batch_size, max_path_length dimensions
        [batch_size, max_path_length, ent_repr_dim]
        :return: [batch_size, repr_dim]
        '''

        if ent_seq is not None:
            seq = tf.concat([rel_seq, ent_seq], axis=1)
        else:
            seq = rel_seq

        # [batch_size, repr_dim]
        enc_seq = encoder(seq, seq_len, **encoder_params)

        return enc_seq

    def get_output(self, rel_seq, seq_len, path_partitions, target_rel, encoder_params, ent_seq=None,
                   aggregator='logsumexp', k=1):
        '''

        :param rel_seq: [batch_size, max_path_len, rel_repr_dim]
        :param seq_len: [batch_size]
        :param path_partitions: [num_queries, 2] - each path in this batch corresponds to a certain candidate
        triple. This tensor indicates the start idx and end idx (exclusive) of the partition of paths for each query,
        so that the scores of the correct paths can be aggregated for a certain triplet.
        :param target_rel: [batch_size, target_rel_repr_dim]
        :param encoder_params: dict of params for the path encoder
        :param ent_seq: if not None, then each element in the path will be formed from the concatenation of
        the entity_repr and relation_repr. The ent_seq  must have same batch_size, max_path_length dimensions
        [batch_size, max_path_length, ent_repr_dim]
        :param aggregator: 'logsumexp', 'max', 'avg', 'topk' - how to aggreagate the path scores for a certain triplet
        :return:
        [num_queries] - the aggregated score for each query in the batch
        '''
        # [batch_size, repr_dim]
        enc_path = self.encode_path(rel_seq, seq_len, encoder_params, ent_seq)

        # [batch_size]
        path_score = tf.reduce_sum(tf.multiply(enc_path, target_rel), axis=1)
        # [num_queries_in_batch]
        query_score = tf.map_fn(
            fn=lambda part: self.aggregate_scores(tf.gather(path_score, tf.range(part[0], part[1])), aggregator, k),
            elems=path_partitions,
            dtype=tf.float32)

        return query_score
