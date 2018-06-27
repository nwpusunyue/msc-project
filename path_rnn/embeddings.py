import logging

import gensim
import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod

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
        self.embedding_matrix_tensor = tf.get_variable(name=self.name,
                                                       initializer=self.embedding_matrix.astype(np.float32),
                                                       dtype=tf.float32,
                                                       trainable=self.trainable)

    def get_idx(self, token):
        return self.vocab[token] if token in self.vocab else self.vocab[self.unk_token]

    def get_embedding_matrix_tensor(self):
        return self.embedding_matrix_tensor


class RandomEmbeddings(Embeddings):

    def __init__(self, tokens, embedding_size, name, unk_token, initializer):
        self.name = name
        self.initializer = initializer
        self.embedding_size = embedding_size
        self.unk_token = unk_token
        self.vocab = {t: i for i, t in enumerate(set(tokens))}

        self.embedding_matrix_tensor = tf.get_variable(name=self.name,
                                                       shape=[len(self.vocab), self.embedding_size],
                                                       initializer=self.initializer,
                                                       trainable=True)

    def get_idx(self, token):
        return self.vocab[token] if token in self.vocab else self.vocab[self.unk_token]

    def get_embedding_matrix_tensor(self):
        return self.embedding_matrix_tensor


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
