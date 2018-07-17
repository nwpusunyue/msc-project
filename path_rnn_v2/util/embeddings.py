import logging

import gensim
import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod
from path_rnn_v2.util.activations import activation_from_string
from parsing.special_tokens import *

logger = logging.getLogger(__name__)


def load_embedding_matrix(word2vec_path):
    '''
    :param embedding_size:
    :param word2vec_path: if not None, embeddings are loaded via Gensim from a pre-trained word2vec nn; the
    additional special tokens will be added subsequently, with randomly initialised embeddings
    :param seed: seed for the random vectors of the special tokens
    :return:
    '''
    model = gensim.models.Word2Vec.load(word2vec_path)
    # convert the wv word vectors into a numpy matrix that is suitable for insertion
    # into TensorFlow
    embedding_matrix = np.zeros((len(model.wv.vocab), model.wv.vector_size), dtype=np.float32)
    vocab = dict()
    for i in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        vocab[model.wv.index2word[i]] = i
    return vocab, embedding_matrix


def add_special_tokens(vocab, embedding_matrix, special_tokens, seed=None):
    # TODO: use the seed param
    count = len(vocab)
    random_embeddings = np.random.normal(size=(len(special_tokens), embedding_matrix.shape[1]))

    for i, (token, is_zero) in enumerate(special_tokens):
        vocab[token] = count
        count += 1
        if is_zero:
            random_embeddings[i, :] = 0

    return vocab, np.append(embedding_matrix, random_embeddings, axis=0)


class Embeddings(ABC):

    @abstractmethod
    def get_idx(self, token):
        pass

    @abstractmethod
    def get_embedding_matrix_tensor(self):
        pass

    @property
    @abstractmethod
    def config_str(self):
        pass

    def embed_sequence(self, seq, name='embedder', max_norm=None, with_projection=False, projection_activation=None,
                       projection_dim=None, reuse=None):
        '''

        :param seq: [batch_size, ..., max_seq_length] tensor
        If seq is None, then embedding matrix is returned instead, with a potential projection
        :param embd: An instantiation of the Embeddings abstract class
        :param name: name of the scope
        :param max_norm: if not None, the embeddings are l2-normalised to max norm
        :param with_projection: If True the embeddings are projected through a dense layer
        :param projection_activation:  used only when project is True. If not None, the specific
        activation is used in the dense layer
        :param projection_dim: if None, the projection dim is the embedding dim, else the provided
        projection dim is used.
        :return:
        [batch_size, ..., max_seq_length, embd_dim]
        '''
        with tf.variable_scope(name, reuse=reuse):
            embd_matrix = self.get_embedding_matrix_tensor()

            if seq is not None:
                embd_seq = tf.nn.embedding_lookup(embd_matrix,
                                                  seq,
                                                  max_norm=max_norm)
            else:
                embd_seq = embd_matrix
            if with_projection:
                if projection_dim is None:
                    projection_dim = embd_matrix.get_shape()[1]
                if projection_activation is not None:
                    projection_activation = activation_from_string(projection_activation)
                embd_seq = tf.layers.dense(embd_seq,
                                           projection_dim,
                                           activation=projection_activation,
                                           use_bias=False)
        return embd_seq


class Word2VecEmbeddings(Embeddings):

    def __init__(self, word2vec_path, name, unk_token, trainable=False, special_tokens=None):
        self.name = name
        self.trainable = trainable
        self.unk_token = unk_token
        self.special_tokens = special_tokens

        self.vocab, self.embedding_matrix = load_embedding_matrix(word2vec_path)
        if self.special_tokens is not None:
            self.vocab, self.embedding_matrix = add_special_tokens(self.vocab,
                                                                   self.embedding_matrix,
                                                                   self.special_tokens)

    def get_idx(self, token):
        return self.vocab[token] if token in self.vocab else self.vocab[self.unk_token]

    def get_embedding_matrix_tensor(self):
        return tf.get_variable(name=self.name,
                               initializer=self.embedding_matrix.astype(np.float32),
                               dtype=tf.float32,
                               trainable=self.trainable)

    @property
    def config_str(self):
        return ('=====Embedding params=====\n'
                'Name: {}\n'
                'Embedding size: {}\n'
                'Vocab size: {}\n'
                'Is Trainable: {}\n'
                'Unknown token: {}\n'
                'Special tokens: {}\n'
                '========================='.format(self.name, self.embedding_matrix.shape[0],
                                                   self.embedding_matrix.shape[1],
                                                   self.trainable, self.unk_token, self.special_tokens))


class RandomEmbeddings(Embeddings):

    def __init__(self, tokens, embedding_size, name, unk_token, initializer):
        self.name = name
        self.initializer = initializer
        self.embedding_size = embedding_size
        self.unk_token = unk_token
        self.vocab = {t: i for i, t in enumerate(set(tokens))}

    def get_idx(self, token):
        return self.vocab[token] if token in self.vocab else self.vocab[self.unk_token]

    def get_embedding_matrix_tensor(self):
        return tf.get_variable(name=self.name,
                               shape=[len(self.vocab), self.embedding_size],
                               initializer=self.initializer,
                               trainable=True)

    @property
    def config_str(self):
        return ('=====Embedding params===== \n'
                'Name: {}\n'
                'Embedding size: {}\n'
                'Vocab size: {}\n'
                'Initializer: {}\n'
                'Unknown token: {}\n'
                '========================='.format(self.name, self.embedding_size, len(self.vocab), self.initializer,
                                                   self.unk_token))


class EntityTypeEmbeddings(RandomEmbeddings):

    def __init__(self, entity2type_path, embedding_size, name, unk_token, initializer):
        self.entity2type = self._read_entity2type(entity2type_path)

        super(EntityTypeEmbeddings, self).__init__(tokens=self.entity2type.values(), embedding_size=embedding_size,
                                                   name=name,
                                                   unk_token=unk_token, initializer=initializer)

    def _read_entity2type(self, entity2type_path):
        with open(entity2type_path, 'r') as f:
            content = f.readlines()
        entity2type = {l.split(' ')[0].strip(): l.split(' ')[1].strip() for l in content}
        return entity2type

    def get_idx(self, token):
        return self.vocab[self.entity2type[token]] if token in self.entity2type else self.vocab[self.unk_token]

    @property
    def config_str(self):
        return ('=====Embedding params===== \n'
                'Name: {}\n'
                'Embedding size: {}\n'
                'Entities mapped: {}\n'
                'Entity types: {}\n'
                'Initializer: {}\n'
                'Unknown token: {}\n'
                '========================='.format(self.name, self.embedding_size, len(self.vocab), self.vocab,
                                                   self.initializer,
                                                   self.unk_token))


if __name__ == '__main__':
    embd = Word2VecEmbeddings('medhop_word2vec_punkt',
                              name='token_embd',
                              unk_token=UNK,
                              trainable=False,
                              special_tokens=[(UNK, False), (PAD, True)])
    print(embd.config_str)
    batch_size = 2
    seq_len = 3
    test_seq = np.zeros([batch_size, seq_len])
    test_seq[:, 0] = embd.get_idx('the')
    test_seq[0, 1] = embd.get_idx('man')
    test_seq[1, 1] = embd.get_idx('woman')
    test_seq[:, 2] = embd.get_idx(PAD)

    max_norm = 0.0001
    activation = 'relu'
    projection_dim = 50
    seq = tf.placeholder(tf.int32, shape=[None, None], name='seq')
    seq_embd = embd.embed_sequence(seq, name='test_emb')
    seq_embd_normed = embd.embed_sequence(seq, name='test_emb_normed', max_norm=max_norm)
    seq_embd_projected = embd.embed_sequence(seq, name='test_emb_proj', with_projection=True,
                                             projection_activation=activation)
    seq_embd_projected_resize = embd.embed_sequence(seq, name='test_embd_proj_resize', with_projection=True,
                                                    projection_activation=activation, projection_dim=projection_dim)

    for op in tf.get_default_graph().get_operations():
        print(str(op.name))

    with tf.train.MonitoredTrainingSession() as sess:
        test_seq_embd = sess.run(seq_embd, feed_dict={
            seq: test_seq
        })

        assert test_seq_embd.shape == (batch_size, seq_len, embd.embedding_matrix.shape[1])
        assert np.array_equal(test_seq_embd[0, 2, :], embd.embedding_matrix[embd.get_idx(PAD)])
        assert np.array_equal(test_seq_embd[1, 2, :], embd.embedding_matrix[embd.get_idx(PAD)])
        assert np.array_equal(test_seq_embd[0, 0, :], embd.embedding_matrix[embd.get_idx('the')])
        assert np.array_equal(test_seq_embd[1, 0, :], embd.embedding_matrix[embd.get_idx('the')])
        assert np.array_equal(test_seq_embd[0, 1, :], embd.embedding_matrix[embd.get_idx('man')])
        assert np.array_equal(test_seq_embd[1, 1, :], embd.embedding_matrix[embd.get_idx('woman')])

        test_seq_embd_normed = sess.run(seq_embd_normed, feed_dict={
            seq: test_seq
        })

        assert (test_seq_embd_normed <= max_norm).all()

        test_seq_embd_projected = sess.run(seq_embd_projected, feed_dict={
            seq: test_seq
        })

        assert test_seq_embd_projected.shape == (batch_size, seq_len, embd.embedding_matrix.shape[1])
        assert (test_seq_embd_projected >= 0.0).all()

        test_seq_embd_projected_resized = sess.run(seq_embd_projected_resize, feed_dict={
            seq: test_seq
        })

        assert test_seq_embd_projected_resized.shape == (batch_size, seq_len, projection_dim)
        assert (test_seq_embd_projected_resized >= 0.0).all()
