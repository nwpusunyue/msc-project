import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from path_rnn.batch_generator import BatchGenerator
from path_rnn.embeddings import load_embedding_matrix
from path_rnn.model.path_rnn import PathRNN
from path_rnn.parameters.model_parameters import ModelParameters
from path_rnn.parameters.train_parameters import Optimizer, TrainParameters
from path_rnn.tensor_generator import get_indexed_paths, get_labels, get_synthetic_dataset
from util import get_target_relations_vocab

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def build_graph(emb_matrix,
                target_relation_vocab,
                max_path_length,
                max_relation_length,
                path_hidden_size,
                aggregator,
                relation_only,
                recurrent_relation_embedder,
                relation_hidden_size,
                optimizer,
                learning_rate,
                label_smoothing):
    relation_token_embedding = tf.get_variable('relation_token_embedding',
                                               initializer=emb_matrix.astype(np.float32),
                                               dtype=tf.float32,
                                               trainable=False)
    entity_embedding = tf.get_variable('entity_embedding',
                                       initializer=emb_matrix.astype(np.float32),
                                       dtype=tf.float32,
                                       trainable=False)
    target_relation_embedding = tf.get_variable(name="target_rel_emb",
                                                shape=[len(target_relation_vocab), path_hidden_size],
                                                initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                            stddev=1.0,
                                                                                            dtype=tf.float64),
                                                dtype=tf.float32,
                                                trainable=True)

    model = PathRNN(max_path_length=max_path_length,
                    max_relation_length=max_relation_length,
                    relation_token_embedding=relation_token_embedding,
                    entity_embedding=entity_embedding,
                    target_relation_embedding=target_relation_embedding,
                    aggregator=aggregator,
                    relation_only=relation_only,
                    recurrent_relation_embedder=recurrent_relation_embedder,
                    rnn_cell=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(path_hidden_size, name='path_lstm'),
                                                           input_keep_prob=model_params.path_keep_prob,
                                                           state_keep_prob=model_params.path_keep_prob),
                    relation_rnn_cell=tf.contrib.rnn.DropoutWrapper(
                        tf.contrib.rnn.LSTMCell(relation_hidden_size, name='relation_lstm'),
                        input_keep_prob=model_params.relation_keep_prob,
                        state_keep_prob=model_params.relation_keep_prob),
                    label_smoothing=label_smoothing)

    model.build_model()
    train_step = Optimizer.get_optimizer(optimizer)(learning_rate).minimize(model.loss)

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


def evaluate(sess,
             model,
             batch_generator,
             batch_type):
    step = 0
    total_loss = 0

    while step < batch_generator.get_batch_count(batch_type):
        (rel_seq,
         ent_seq,
         target_relations,
         partitions,
         path_lengths,
         num_words,
         labels) = batch_generator.get_batch(batch_type)
        loss = sess.run([model.loss],
                        feed_dict=build_feed_dict(model=model,
                                                  relation_seq=rel_seq,
                                                  entity_seq=ent_seq,
                                                  path_partitions=partitions,
                                                  path_lengths=path_lengths,
                                                  num_words=num_words,
                                                  target_relations=target_relations,
                                                  labels=labels))
        total_loss += loss[0]
        step += 1
    return total_loss / batch_generator.get_batch_count(batch_type)


def train(dataset,
          document_store,
          model_params,
          train_params,
          sess_config):
    vocab, emb_matrix = load_embedding_matrix(embedding_size=100,
                                              word2vec_path=train_params.word_embd_path)
    # TODO: this vocab must be subsequently stored for test time
    target_relation_vocab = get_target_relations_vocab(dataset['relation'])

    num_words_filler = 0 if model_params.recurrent_relation_embedder else 1
    (indexed_relation_paths,
     indexed_entity_paths,
     indexed_target_relations,
     path_partitions,
     path_lengths,
     num_words) = get_indexed_paths(q_relation_paths=dataset['relation_paths'],
                                    q_entity_paths=dataset['entity_paths'],
                                    target_relations=dataset['relation'],
                                    document_store=document_store,
                                    relation_token_vocab=vocab,
                                    entity_vocab=vocab,
                                    target_relation_vocab=target_relation_vocab,
                                    max_path_length=model_params.max_path_length,
                                    max_relation_length=model_params.max_relation_length,
                                    num_words_filler=num_words_filler,
                                    replace_in_doc=(not train_params.debug_mode),
                                    truncate_doc=model_params.truncate_documents)
    labels = get_labels(dataset['label'])

    batch_generator = BatchGenerator(indexed_relation_paths=indexed_relation_paths,
                                     indexed_entity_paths=indexed_entity_paths,
                                     indexed_target_relations=indexed_target_relations,
                                     path_partitions=path_partitions,
                                     path_lengths=path_lengths,
                                     num_words=num_words,
                                     labels=labels,
                                     batch_size=train_params.batch_size,
                                     test_prop=train_params.test_prop,
                                     train_eval_prop=train_params.train_eval_prop,
                                     train_eval_batch_size=train_params.train_eval_batch_size)

    model, train_step = build_graph(emb_matrix=emb_matrix,
                                    target_relation_vocab=target_relation_vocab,
                                    max_path_length=model_params.max_path_length,
                                    max_relation_length=model_params.max_relation_length,
                                    path_hidden_size=model_params.path_rnn_hidden_size,
                                    relation_hidden_size=model_params.relation_rnn_hidden_size,
                                    relation_only=model_params.relation_only,
                                    recurrent_relation_embedder=model_params.recurrent_relation_embedder,
                                    aggregator=model_params.aggregator,
                                    optimizer=train_params.optimizer,
                                    learning_rate=train_params.learning_rate,
                                    label_smoothing=model_params.label_smoothing)

    with tf.train.MonitoredTrainingSession(config=sess_config) as sess:
        step = 0
        total_train_loss = 0
        while batch_generator.get_epochs_completed(BatchGenerator.TRAIN) < train_params.epochs:
            step += 1
            (batch_rel_seq,
             batch_ent_seq,
             batch_target_relations,
             batch_partitions,
             batch_path_lengths,
             batch_num_words,
             batch_labels) = batch_generator.get_batch(BatchGenerator.TRAIN)

            train_loss, train_prob, _ = sess.run([model.loss,
                                                  model.prob,
                                                  train_step],
                                                 feed_dict=build_feed_dict(model=model,
                                                                           relation_seq=batch_rel_seq,
                                                                           entity_seq=batch_ent_seq,
                                                                           path_partitions=batch_partitions,
                                                                           path_lengths=batch_path_lengths,
                                                                           num_words=batch_num_words,
                                                                           target_relations=batch_target_relations,
                                                                           labels=batch_labels))

            total_train_loss += train_loss
            if step % batch_generator.get_batch_count(BatchGenerator.TRAIN) == 0:
                print('Epoch {} train loss: {}'.format(batch_generator.get_epochs_completed(BatchGenerator.TRAIN),
                                                       total_train_loss / batch_generator.get_batch_count(
                                                           BatchGenerator.TRAIN)))
                total_train_loss = 0

            if step % train_params.check_period == 0:
                if batch_generator.train_eval_size > 0:
                    loss = evaluate(sess, model, batch_generator, BatchGenerator.TRAIN_EVAL)
                    print(
                        'Epoch: {} Step: {} Train evaluation Loss={}'.format(
                            batch_generator.get_epochs_completed(BatchGenerator.TRAIN),
                            step,
                            loss))
                if batch_generator.test_size > 0:
                    loss = evaluate(sess, model, batch_generator, BatchGenerator.TEST)
                    print(
                        'Epoch: {} Step: {} Test Loss={}'.format(
                            batch_generator.get_epochs_completed(BatchGenerator.TRAIN),
                            step,
                            loss))
                print()


if __name__ == '__main__':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list='0',
                                                      per_process_gpu_memory_fraction=1.0))
    # train_dataset = get_synthetic_dataset(num_positive=500,
    #                                       num_negative=1000,
    #                                       path_length=5,
    #                                       paths_per_query=3,
    #                                       doc_range_positive=(0, 30),
    #                                       doc_range_negative=(30, 60),
    #                                       positive_entities=['DB00338', 'Q04656', 'O75030', 'DB00133',
    #                                                          'DB00773', 'DB00338', 'DB00133',
    #                                                          'P11388'],
    #                                       negative_entities=['DB00341', 'P35367', 'P05231', 'P15692', 'P16104',
    #                                                          'DB00773'],
    #                                       target_relation='interacts_with')

    train_document_store = pickle.load(open('./data/train_doc_store_punkt.pickle',
                                            'rb'))
    train_dataset = pd.read_json(
        './data/sentwise=F_cutoff=4_limit=100_method=shortest_tokenizer=punkt_medhop_train.json')
    print('Tuples count: {}'.format(len(train_dataset)))
    print('Maximum document length: {}'.format(train_document_store.max_tokens))

    # train_dataset = pd.concat([train_dataset.loc[train_dataset['label'] == 1].sample(100, random_state=10),
    #                            train_dataset.loc[train_dataset['label'] == 0].sample(100, random_state=10)])
    # train_dataset = train_dataset.sample(100, random_state=10)

    train_params = TrainParameters(optimizer=Optimizer.ADAM,
                                   learning_rate=5e-4,
                                   epochs=100,
                                   batch_size=32,
                                   test_prop=0.1,
                                   train_eval_prop=1.0,
                                   check_period=50,
                                   word_embd_path='medhop_word2vec_punkt',
                                   debug_mode=False,
                                   train_eval_batch_size=1000)
    print('Train params:')
    train_params.print()
    print()

    model_params = ModelParameters(
        max_path_length=5,
        max_relation_length=train_document_store.max_tokens,
        path_rnn_hidden_size=150,
        path_keep_prob=0.9,
        aggregator=PathRNN.LOG_SUM_EXP,
        relation_only=False,
        recurrent_relation_embedder=True,
        relation_rnn_hidden_size=150,
        relation_keep_prob=0.9,
        truncate_documents=True,
        label_smoothing=0.1)
    print('Model params:')
    model_params.print()
    print()

    train(dataset=train_dataset,
          document_store=train_document_store,
          model_params=model_params,
          train_params=train_params,
          sess_config=config)
