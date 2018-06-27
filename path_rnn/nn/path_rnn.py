import tensorflow as tf

# TODO: Add a top-k aggregator
aggregators = {
    'log_sum_exp': tf.reduce_logsumexp,
    'average': tf.reduce_mean,
    'max': tf.reduce_max,
}


class PathRNN:
    LOG_SUM_EXP = 'log_sum_exp'
    AVERAGE = 'average'
    MAX = 'max'

    def __init__(self,
                 max_path_length,
                 max_relation_length,
                 relation_token_embedding,
                 entity_embedding,
                 target_relation_embedding,
                 rnn_cell,
                 aggregator,
                 relation_only=False,
                 recurrent_relation_embedder=False,
                 relation_rnn_cell=None,
                 label_smoothing=0.0):
        self.max_path_length = max_path_length
        self.max_relation_length = max_relation_length

        self.relation_token_embedding = relation_token_embedding
        self.entity_embedding = entity_embedding
        self.target_relation_embedding = target_relation_embedding

        self.rnn_cell = rnn_cell
        self.aggregator = aggregator
        self.relation_only = relation_only
        self.recurrent_relation_embedder = recurrent_relation_embedder
        self.relation_rnn_cell = relation_rnn_cell
        self.label_smoothing = label_smoothing
        self.initialize_placeholders()

    def _get_embedding(self,
                       input,
                       embedding_matrix):
        return tf.nn.embedding_lookup(embedding_matrix,
                                      input)

    def _aggregate(self, path_scores):
        return aggregators[self.aggregator](path_scores)

    def _embed_relations(self):
        # embed & project relation paths
        relation_token_emb_dim = self.relation_token_embedding.get_shape()[1].value
        # [batch_size, max_path_length, max_relation_length, embedding_dim]
        self.relation_seq_token_embedded = self._get_embedding(embedding_matrix=self.relation_token_embedding,
                                                               input=self.relation_seq)
        # [batch_size, max_path_length, max_relation_length, embedding_dim]
        self.relation_seq_token_projected = tf.layers.dense(self.relation_seq_token_embedded,
                                                            relation_token_emb_dim,
                                                            activation=tf.tanh,
                                                            use_bias=False,
                                                            name='projection_rel_tok')

        if self.recurrent_relation_embedder:
            relation_rnn_state_size = (self.relation_rnn_cell.state_size.c
                                       if type(self.relation_rnn_cell.state_size) is tf.contrib.rnn.LSTMStateTuple
                                       else self.relation_rnn_cell.state_size)
            # [batch_size x max_path_length, max_relation_length, embedding_dim]
            relation_rnn_in = tf.reshape(self.relation_seq_token_projected,
                                         shape=[-1,
                                                self.max_relation_length,
                                                relation_token_emb_dim])
            # [batch_size x max_path_length]
            relation_rnn_seq_lengths = tf.reshape(self.num_words, shape=[-1])

            _, relation_state = tf.nn.dynamic_rnn(cell=self.relation_rnn_cell,
                                                  inputs=relation_rnn_in,
                                                  sequence_length=relation_rnn_seq_lengths,
                                                  dtype=tf.float32)
            if type(relation_state) is tf.contrib.rnn.LSTMStateTuple:
                # [batch_size x max_path_length, relation_rnn_state_size]
                relation_state = relation_state.c

            # [batch_size, max_path_length, relation_rnn_state_size]
            relation_seq_embedded = tf.reshape(relation_state, shape=[-1,
                                                                      self.max_path_length,
                                                                      relation_rnn_state_size])
        else:
            # [batch_size, max_path_length, embedding_dim]
            relation_seq_embedded = tf.reduce_sum(self.relation_seq_token_projected, axis=2) / tf.expand_dims(
                self.num_words,
                axis=2)
        return relation_seq_embedded

    def _embed_entities(self):
        # embed & project entity paths
        entity_emb_dim = self.entity_embedding.get_shape()[1].value
        # [batch_size, max_path_length, embedding_dim]
        self.entity_seq_embedded = self._get_embedding(embedding_matrix=self.entity_embedding,
                                                       input=self.entity_seq)
        # [batch_size, max_path_length, embedding_dim]
        entity_seq_projected = tf.layers.dense(self.entity_seq_embedded,
                                               entity_emb_dim,
                                               activation=tf.tanh,
                                               use_bias=False,
                                               name='projection_ent')
        return entity_seq_projected

    def build_model(self):
        # embed & project relation paths
        # [batch_size, max_path_length, relation_embedding_dim]
        self.relation_seq_embedded = self._embed_relations()

        # embed & project entity paths
        # [bathc_size, max_path_length, entity_embedding_dim]
        self.entity_seq_projected = self._embed_entities()

        # embed target relations
        # [batch_size, embedding_dim]
        self.target_relation_embedded = self._get_embedding(embedding_matrix=self.target_relation_embedding,
                                                            input=self.target_rel)

        # embed each path
        if self.relation_only:
            # [batch_size, max_path_length, relation_embedding_dim]
            self.rnn_in = self.relation_seq_embedded
        else:
            # [batch_size, max_path_length, relation_embedding_dim + entity_embedding_dim]
            self.rnn_in = tf.concat([self.relation_seq_embedded,
                                     self.entity_seq_projected],
                                    axis=2)

        # encode each path by passing each of its rel-entity pairs through an RNN
        # [batch_size, hidden_size]
        _, self.state = tf.nn.dynamic_rnn(cell=self.rnn_cell,
                                          inputs=self.rnn_in,
                                          sequence_length=self.path_lengths,
                                          dtype=tf.float32)

        if type(self.state) is tf.contrib.rnn.LSTMStateTuple:
            self.state = self.state.c

        # compute dot product of each path-encoding and the corresponding target-relation
        # [batch_size]
        self.path_scores = tf.reduce_sum(tf.multiply(self.state,
                                                     self.target_relation_embedded),
                                         axis=1)
        # for each partitioning range (i.e. range of paths corresponding to a certain query),
        # select only those paths in that range and aggregate their scores.
        # [num_queries]
        self.query_scores = tf.map_fn(fn=lambda part: self._aggregate(tf.gather(self.path_scores,
                                                                                tf.range(part[0], part[1]))),
                                      elems=self.path_partitions,
                                      dtype=tf.float32)

        self._label = tf.cast(self.label, tf.float32)
        if self.label_smoothing > 0:
            # [num_queries]
            self._label = self._label * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # triplet probability
        # [num_queries]
        self.prob = tf.sigmoid(self.query_scores)
        # scalar
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._label,
                                                                           logits=self.query_scores))

    def initialize_placeholders(self):
        # [num_queries, 2] - start index (inclusive) and end index(exclusive) for each query partition
        self.path_partitions = tf.placeholder(tf.int32,
                                              shape=[None,
                                                     2])
        # [batch_size] - how many relations per each paths - can be passed to dynamic rnn
        self.path_lengths = tf.placeholder(tf.int32,
                                           shape=[None])
        # [batch_size, max_path_length] - how many words per each relation
        self.num_words = tf.placeholder(tf.float32,
                                        shape=[None,
                                               self.max_path_length])
        # [batch_size, max_path_length, max_relation_length]
        self.relation_seq = tf.placeholder(tf.int64,
                                           shape=[None,
                                                  self.max_path_length,
                                                  self.max_relation_length])
        # [batch_size, max_path_length]
        self.entity_seq = tf.placeholder(tf.int64,
                                         shape=[None,
                                                self.max_path_length])
        # [batch_size]
        self.target_rel = tf.placeholder(tf.int64,
                                         shape=[None])
        # [num_queries]
        self.label = tf.placeholder(tf.int64,
                                    shape=[None])

        self.placeholders = {
            'path_partitions': self.path_partitions,
            'path_lengths': self.path_lengths,
            'num_words': self.num_words,
            'relation_seq': self.relation_seq,
            'entity_seq': self.entity_seq,
            'target_rel': self.target_rel,
            'label': self.label
        }
