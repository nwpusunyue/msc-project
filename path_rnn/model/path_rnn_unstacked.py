import tensorflow as tf
from path_rnn.tfutil import loss_function


class PathRNN:

    def __init__(self,
                 num_partitions,
                 max_path_length,
                 max_relation_length,
                 relation_token_embedding,
                 entity_embedding,
                 target_relation_embedding,
                 rnn_cell):
        self.num_partitions = num_partitions
        self.max_path_length = max_path_length
        self.max_relation_length = max_relation_length
        self.relation_token_embedding = relation_token_embedding
        self.entity_embedding = entity_embedding
        self.target_relation_embedding = target_relation_embedding
        self.rnn_cell = rnn_cell
        self.initialize_placeholders()

    def _get_embedding(self,
                       input,
                       embedding_matrix):
        return tf.nn.embedding_lookup(embedding_matrix,
                                      input)

    def _aggregate(self, path_scores):
        return tf.reduce_logsumexp(path_scores)

    def build_model(self):
        # embed & project relation paths
        relation_token_emb_dim = self.relation_token_embedding.get_shape()[1].value
        # [batch_size, max_path_length, max_relation_length, embedding_dim]
        relation_seq_token_embedded = self._get_embedding(embedding_matrix=self.relation_token_embedding,
                                                          input=self.relation_seq)
        # [batch_size, max_path_length, max_relation_length, embedding_dim]
        relation_seq_token_projected = tf.contrib.layers.fully_connected(relation_seq_token_embedded,
                                                                         relation_token_emb_dim,
                                                                         biases_initializer=None,
                                                                         activation_fn=None)
        # [batch_size, max_path_length, embedding_dim]
        relation_seq_embedded = tf.reduce_sum(relation_seq_token_projected, axis=2) / tf.expand_dims(self.num_words, 2)

        # embed & project entity paths
        entity_emb_dim = self.entity_embedding.get_shape()[1].value
        # [batch_size, max_path_length, embedding_dim]
        entity_seq_embedded = self._get_embedding(embedding_matrix=self.entity_embedding,
                                                  input=self.entity_seq)
        # [batch_size, max_path_length, embedding_dim]
        entity_seq_projected = tf.contrib.layers.fully_connected(entity_seq_embedded,
                                                                 entity_emb_dim,
                                                                 biases_initializer=None,
                                                                 activation_fn=None)

        # embed target relations
        # [batch_size, embedding_dim]
        target_relation_embedded = self._get_embedding(embedding_matrix=self.target_relation_embedding,
                                                       input=self.target_rel)

        # embed each path
        # [batch_size, max_path_length, rel_embedding_dim + entity_embedding_dim]
        rnn_in = tf.concat([relation_seq_embedded,
                            entity_seq_projected],
                           2)

        # encode each path by passing each of its rel-entity pairs through an RNN
        # [batch_size, hidden_size]
        outputs, state = tf.nn.dynamic_rnn(cell=self.rnn_cell,
                                           inputs=rnn_in,
                                           sequence_length=self.path_lengths,
                                           dtype=tf.float64)

        if type(state) is tf.contrib.rnn.LSTMStateTuple:
            state = state.c

        # compute dot product of each path-encoding and the corresponding target-relation
        # [batch_size]
        path_scores = tf.reduce_sum(tf.multiply(state,
                                                target_relation_embedded),
                                    axis=1)

        partitioned_path_scores = tf.dynamic_partition(data=path_scores,
                                                       partitions=self.path_partitions,
                                                       num_partitions=self.num_partitions)

        agg_scores = [self._aggregate(scores) for scores in partitioned_path_scores]
        query_scores = tf.stack(agg_scores)

        # triplet probability
        self.prob = tf.sigmoid(query_scores)

        self.loss = tf.reduce_mean(loss_function(self.prob, tf.cast(self.label, tf.float64)))

    def initialize_placeholders(self):
        # [batch_size] - indices used to partition the paths by the example they belong to: 0 will belong to query 0,
        # 1 to query 1 and so on.
        self.path_partitions = tf.placeholder(tf.int32,
                                              shape=[None])
        # [batch_size] - how many relations per each paths - can be passed to dynamic rnn
        self.path_lengths = tf.placeholder(tf.int64,
                                           shape=[None])

        # [batch_size, max_path_length] - how many words per each relation
        self.num_words = tf.placeholder(tf.float64,
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

        # [batch_size]
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
