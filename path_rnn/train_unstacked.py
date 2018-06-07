import pandas as pd
import tensorflow as tf
from path_rnn.embeddings import load_embedding_matrix
from path_rnn.model.path_rnn_unstacked import PathRNN
from path_rnn.tensor_generator_unstacked import get_indexed_paths, get_labels
from util import get_target_relations_vocab


def build_graph(emb_matrix,
                target_relation_vocab,
                num_partitions,
                max_path_length,
                max_relation_length,
                hidden_size,
                learning_rate):
    relation_token_embedding = tf.get_variable('relation_token_embedding',
                                               initializer=emb_matrix,
                                               trainable=False)
    entity_embedding = tf.get_variable('entity_embedding',
                                       initializer=emb_matrix,
                                       trainable=False)
    target_relation_embedding = tf.get_variable(name="target_rel_emb",
                                                shape=[len(target_relation_vocab), hidden_size],
                                                initializer=tf.random_uniform_initializer(-0.04, 0.04,
                                                                                          dtype=tf.float64),
                                                dtype=tf.float64,
                                                trainable=True)
    model = PathRNN(num_partitions=num_partitions,
                    max_path_length=max_path_length,
                    max_relation_length=max_relation_length,
                    rnn_cell=tf.contrib.rnn.LSTMCell(hidden_size),
                    relation_token_embedding=relation_token_embedding,
                    entity_embedding=entity_embedding,
                    target_relation_embedding=target_relation_embedding)

    model.build_model()

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(model.loss)

    return model, train_step


def build_feed_dict(model,
                    relation_seq,
                    entity_seq,
                    path_partitions,
                    path_lengths,
                    num_words,
                    target_relations,
                    labels):
    return {
        model.placeholders['relation_seq']: relation_seq,
        model.placeholders['entity_seq']: entity_seq,
        model.placeholders['path_partitions']: path_partitions,
        model.placeholders['path_lengths']: path_lengths,
        model.placeholders['num_words']: num_words,
        model.placeholders['target_rel']: target_relations,
        model.placeholders['label']: labels
    }


def train(dataset,
          num_partitions,
          max_path_length,
          max_relation_length,
          hidden_size,
          learning_rate,
          batch_size,
          epochs):
    vocab, emb_matrix = load_embedding_matrix(embedding_size=100,
                                              word2vec_path='medhop_word2vec')
    # TODO: this vocab must be subsequently stored for test time
    target_relation_vocab = get_target_relations_vocab(dataset['relation'])

    (rel_seq,
     entity_seq,
     target_relations,
     path_partitions,
     path_lengths,
     num_words) = get_indexed_paths(q_relation_paths=dataset['relation_paths'],
                                    q_entity_paths=dataset['entity_paths'],
                                    target_relations=dataset['relation'],
                                    relation_token_vocab=vocab,
                                    entity_vocab=vocab,
                                    target_relation_vocab=target_relation_vocab,
                                    max_path_length=max_path_length,
                                    max_relation_length=max_relation_length)
    labels = get_labels(dataset['label'])

    model, train_step = build_graph(emb_matrix=emb_matrix,
                                    target_relation_vocab=target_relation_vocab,
                                    num_partitions=num_partitions,
                                    max_path_length=max_path_length,
                                    max_relation_length=max_relation_length,
                                    hidden_size=hidden_size,
                                    learning_rate=learning_rate)

    with tf.train.MonitoredTrainingSession() as sess:
        for i in range(epochs):
            train_loss, train_prob, _ = sess.run([model.loss,
                                                  model.prob,
                                                  train_step],
                                                 feed_dict=build_feed_dict(model=model,
                                                                           relation_seq=rel_seq,
                                                                           entity_seq=entity_seq,
                                                                           path_partitions=path_partitions,
                                                                           path_lengths=path_lengths,
                                                                           num_words=num_words,
                                                                           target_relations=target_relations,
                                                                           labels=labels))
            print('Epoch: {} Loss={}'.format(i, train_loss, train_prob, labels))


if __name__ == '__main__':
    dataset = pd.read_json('dummy_dataset.json').loc[:10]
    train(dataset=dataset,
          num_partitions=len(dataset),
          max_path_length=9,
          max_relation_length=200,
          hidden_size=150,
          learning_rate=1e-2,
          epochs=100)
