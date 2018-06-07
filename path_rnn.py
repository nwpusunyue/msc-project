import tensorflow as tf
from tfutil import loss_function


class PathRNN:

    def __init__(self,
                 max_paths,
                 max_path_length,
                 max_relation_length,
                 relation_token_embedding,
                 entity_embedding,
                 target_relation_embedding,
                 rnn_cell):
        self.max_paths = max_paths
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
        return tf.reduce_logsumexp(path_scores, axis=1)

    def build_model(self):
        # embed & project relation paths
        relation_token_emb_dim = self.relation_token_embedding.get_shape()[1].value
        # [batch_size, num_paths, max_relations, max_relation_length, embedding_dim]
        relation_seq_token_embedded = self._get_embedding(embedding_matrix=self.relation_token_embedding,
                                                          input=self.relation_seq)
        # [batch_size, num_paths, max_path_length, max_relation_length, embedding_dim]
        relation_seq_token_projected = tf.contrib.layers.fully_connected(relation_seq_token_embedded,
                                                                         relation_token_emb_dim,
                                                                         biases_initializer=None,
                                                                         activation_fn=None)
        # [batch_size, num_paths, max_path_length, embedding_dim]
        relation_seq_embedded = tf.reduce_sum(relation_seq_token_projected, axis=3) / tf.expand_dims(self.num_words, 3)

        # embed & project entity paths
        entity_emb_dim = self.entity_embedding.get_shape()[1].value
        # [batch_size, num_paths, max_path_length, embedding_dim]
        entity_seq_embedded = self._get_embedding(embedding_matrix=self.entity_embedding,
                                                  input=self.entity_seq)
        # [batch_size, num_paths, max_path_length, embedding_dim]
        entity_seq_projected = tf.contrib.layers.fully_connected(entity_seq_embedded,
                                                                 entity_emb_dim,
                                                                 biases_initializer=None,
                                                                 activation_fn=None)

        # embed target relations
        # [batch_size,embedding_dim]
        target_relation_embedded = self._get_embedding(embedding_matrix=self.target_relation_embedding,
                                                       input=self.target_rel)

        # embed each path
        # [batch_size * max_paths, max_path_length, rel_embedding_dim + entity_embedding_dim]
        rnn_in = tf.reshape(tf.concat([relation_seq_embedded,
                                       entity_seq_projected],
                                      3),
                            [-1,
                             self.max_path_length,
                             relation_token_emb_dim + entity_emb_dim])

        # [batch_size * num_paths]
        rnn_sequence_lengths = tf.reshape(self.path_lengths,
                                          [-1])

        # encode each path by passing each of its rel-entity pairs through an RNN
        # [batch_size * num_paths, hidden_size]
        outputs, state = tf.nn.dynamic_rnn(cell=self.rnn_cell,
                                           inputs=rnn_in,
                                           sequence_length=rnn_sequence_lengths,
                                           dtype=tf.float64)

        if type(state) is tf.contrib.rnn.LSTMStateTuple:
            state = state.c
            state_size = self.rnn_cell.state_size.c
        else:
            state_size = self.rnn_cell.state_size

        # [batch_size, num_paths, hidden_size]
        path_emb = tf.reshape(state,
                              [-1,
                               self.max_paths,
                               state_size])

        # compute dot product of each path-encoding and the corresponding target-relation
        # [batch_size, num_paths]
        path_scores = tf.reduce_sum(tf.multiply(path_emb,
                                                tf.expand_dims(target_relation_embedded,
                                                               axis=1)),
                                    axis=2)

        # aggregate scores per query
        # query_scores = self._aggregate(path_scores)
        # TODO: for now use the Mean instead of other aggregations for test purpose only!
        summed_path_scores = tf.reduce_sum(path_scores, axis=1)
        query_scores = summed_path_scores / tf.cast(self.num_paths, tf.float64)

        # triplet probability
        self.prob = tf.sigmoid(query_scores)

        self.loss = tf.reduce_mean(loss_function(self.prob, tf.cast(self.label, tf.float64)))

    def initialize_placeholders(self):
        # [batch_size] - how many paths per each instance
        self.num_paths = tf.placeholder(tf.int64,
                                        shape=[None])
        # [batch_size, num_paths] - how many relations per each paths - can be passed to dynamic rnn
        self.path_lengths = tf.placeholder(tf.int64,
                                           shape=[None,
                                                  self.max_paths])

        # [batch_size, num_paths, max_relations] - how many words per each relation
        self.num_words = tf.placeholder(tf.float64,
                                        shape=[None,
                                               self.max_paths,
                                               self.max_path_length])

        # [batch_size, num_paths, max_relations, max_relation_length]
        self.relation_seq = tf.placeholder(tf.int64,
                                           shape=[None,
                                                  self.max_paths,
                                                  self.max_path_length,
                                                  self.max_relation_length])

        # [batch_size, num_paths, max_relations]
        self.entity_seq = tf.placeholder(tf.int64,
                                         shape=[None,
                                                self.max_paths,
                                                self.max_path_length])

        # [batch_size]
        self.target_rel = tf.placeholder(tf.int64,
                                         shape=[None])

        # [batch_size]
        self.label = tf.placeholder(tf.int64,
                                    shape=[None])

        self.placeholders = {
            'num_paths': self.num_paths,
            'path_lengths': self.path_lengths,
            'num_words': self.num_words,
            'relation_seq': self.relation_seq,
            'entity_seq': self.entity_seq,
            'target_rel': self.target_rel,
            'label': self.label
        }
