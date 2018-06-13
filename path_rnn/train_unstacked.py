import pandas as pd
import tensorflow as tf
from path_rnn.embeddings import load_embedding_matrix
from path_rnn.model.path_rnn_unstacked import PathRNN
from path_rnn.batch_generator import BatchGenerator
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
          max_path_length,
          max_relation_length,
          hidden_size,
          learning_rate,
          batch_size,
          epochs,
          test_prop,
          sess_config):
    vocab, emb_matrix = load_embedding_matrix(embedding_size=100,
                                              word2vec_path='medhop_word2vec')
    # TODO: this vocab must be subsequently stored for test time
    target_relation_vocab = get_target_relations_vocab(dataset['relation'])

    batch_generator = BatchGenerator(dataset=dataset,
                                     relation_token_vocab=vocab,
                                     entity_vocab=vocab,
                                     target_relation_vocab=target_relation_vocab,
                                     max_path_length=max_path_length,
                                     max_relation_length=max_relation_length,
                                     batch_size=batch_size,
                                     test_prop=test_prop)

    model, train_step = build_graph(emb_matrix=emb_matrix,
                                    target_relation_vocab=target_relation_vocab,
                                    num_partitions=batch_size,
                                    max_path_length=max_path_length,
                                    max_relation_length=max_relation_length,
                                    hidden_size=hidden_size,
                                    learning_rate=learning_rate)
    print('Total train batches: {} Total test batches: {}'.format(batch_generator.train_idxs_generator.total_batches,
                                                                  batch_generator.test_idxs_generator.total_batches))

    with tf.train.MonitoredTrainingSession(config=sess_config) as sess:
        step = 0
        check_period = batch_generator.train_idxs_generator.total_batches
        total_epoch_loss = 0

        while batch_generator.train_idxs_generator.epochs_completed < epochs:
            step += 1
            (batch_rel_seq,
             batch_ent_seq,
             batch_target_relations,
             batch_partitions,
             batch_path_lengths,
             batch_num_words,
             batch_labels) = batch_generator.get_batch()
            train_loss, _ = sess.run([model.loss,
                                      train_step],
                                     feed_dict=build_feed_dict(model=model,
                                                               relation_seq=batch_rel_seq,
                                                               entity_seq=batch_ent_seq,
                                                               path_partitions=batch_partitions,
                                                               path_lengths=batch_path_lengths,
                                                               num_words=batch_num_words,
                                                               target_relations=batch_target_relations,
                                                               labels=batch_labels))
            total_epoch_loss += train_loss

            if step % batch_generator.train_idxs_generator.total_batches == 0:
                print('Epoch: {} Step: {} Loss={}'.format(batch_generator.train_idxs_generator.epochs_completed,
                                                          step,
                                                          total_epoch_loss
                                                          / batch_generator.train_idxs_generator.total_batches))
                total_epoch_loss = 0
            if step % check_period == 0 and batch_generator.test_idxs_generator.total_batches > 0:
                total_test_loss = 0
                for _ in range(int(batch_generator.test_idxs_generator.total_batches)):
                    (test_rel_seq,
                     test_ent_seq,
                     test_target_relations,
                     test_partitions,
                     test_path_lengths,
                     test_num_words,
                     test_labels) = batch_generator.get_batch(test=True)
                    test_loss = sess.run([model.loss],
                                         feed_dict=build_feed_dict(model=model,
                                                                   relation_seq=test_rel_seq,
                                                                   entity_seq=test_ent_seq,
                                                                   path_partitions=test_partitions,
                                                                   path_lengths=test_path_lengths,
                                                                   num_words=test_num_words,
                                                                   target_relations=test_target_relations,
                                                                   labels=test_labels)
                                         )
                    total_test_loss += test_loss[0]

                print(
                    'Epoch: {} Step: {} Test Loss={}\n'.format(batch_generator.train_idxs_generator.epochs_completed,
                                                               step,
                                                               total_test_loss
                                                               / batch_generator.test_idxs_generator.total_batches))


if __name__ == '__main__':
    dataset = pd.read_json('dummy_docs.json')
    print('Tuples count: {}'.format(len(dataset)))
    #tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list='0'))
    train(dataset=dataset,
          max_path_length=5,
          max_relation_length=500,
          hidden_size=50,
          learning_rate=1e-3,
          epochs=100,
          batch_size=10,
          test_prop=0.1,
          sess_config=None)
