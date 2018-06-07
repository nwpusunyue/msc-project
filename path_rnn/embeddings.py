import gensim
import logging
import numpy as np
from parsing.special_tokens import *

logger = logging.getLogger(__name__)


def load_embedding_matrix(embedding_size,
                          word2vec_path=None,
                          np_path=None):
    '''
    :param embedding_size:
    :param word2vec_path: if not None, embeddings are loaded via Gensim from a pre-trained word2vec model; the
    additional special tokens will be added subsequently, with randomly initialised embeddings
    :param np_path: if not None, embeddings are loaded directly from
    :return:
    '''
    if word2vec_path is not None:
        model = gensim.models.Word2Vec.load(word2vec_path)
        # convert the wv word vectors into a numpy matrix that is suitable for insertion
        # into TensorFlow
        embedding_matrix = np.zeros((len(model.wv.vocab), embedding_size))
        vocab = dict()
        for i in range(len(model.wv.vocab)):
            embedding_vector = model.wv[model.wv.index2word[i]]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            vocab[model.wv.index2word[i]] = i
        vocab, embedding_matrix = add_special_tokens(vocab, embedding_matrix)
    else:
        # TODO: add this bit once establishing how the numpy version of the emb matr will be stored.
        # alternative is to just use the previous approach but with a fixed random seed so the result is reproducible
        pass

    return vocab, embedding_matrix


def add_special_tokens(vocab, embedding_matrix):
    # If these stay randomly initialised, the whole emb. matrix + vocab must
    # be stored like this, so that the random vectors to not change at each run.
    count = len(vocab)
    vocab[ENT_1] = count
    vocab[ENT_2] = count + 1
    vocab[ENT_X] = count + 2
    vocab[UNK] = count + 3
    vocab[END] = count + 4
    vocab[PAD] = count + 5

    random_embeddings = np.random.normal(size=(6, embedding_matrix.shape[1]))
    random_embeddings[5, :] = 0

    return vocab, np.append(embedding_matrix, random_embeddings, axis=0)
