import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from path_rnn.batch_generator import BatchGenerator
from path_rnn.embeddings import Word2VecEmbeddings, RandomEmbeddings, EntityTypeEmbeddings
from path_rnn.nn.path_rnn import PathRNN
from path_rnn.parameters.model_parameters import ModelParameters
from path_rnn.parameters.train_parameters import Optimizer, TrainParameters
from path_rnn.tensor_generator import get_indexed_paths, get_labels, get_synthetic_dataset
from parsing.special_tokens import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def build_graph(relation_token_embeddings_tensor,
                entity_embeddings_tensor,
                target_relation_embeddings_tensor,
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
    model = PathRNN(max_path_length=max_path_length,
                    max_relation_length=max_relation_length,
                    relation_token_embedding=relation_token_embeddings_tensor,
                    entity_embedding=entity_embeddings_tensor,
                    target_relation_embedding=target_relation_embeddings_tensor,
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
    total_loss = 0

    for step in range(batch_generator.get_batch_count(batch_type)):
        (rel_seq,
         ent_seq,
         target_relations,
         partitions,
         path_lengths,
         num_words,
         labels) = batch_generator.get_batch(batch_type)
        loss, prob = sess.run([model.loss, model.prob],
                              feed_dict=build_feed_dict(model=model,
                                                        relation_seq=rel_seq,
                                                        entity_seq=ent_seq,
                                                        path_partitions=partitions,
                                                        path_lengths=path_lengths,
                                                        num_words=num_words,
                                                        target_relations=target_relations,
                                                        labels=labels))
        total_loss += loss

    return total_loss / batch_generator.get_batch_count(batch_type)


def evaluate_medhop_accuracy(dataset,
                             sess,
                             model,
                             batch_generator,
                             batch_type):
    probs = []
    idxs = []

    for step in range(batch_generator.get_batch_count(batch_type)):
        (rel_seq,
         ent_seq,
         target_relations,
         partitions,
         path_lengths,
         num_words,
         labels,
         batch_idxs) = batch_generator.get_batch(batch_type, return_idxs=True)
        batch_prob = sess.run([model.prob],
                              feed_dict=build_feed_dict(model=model,
                                                        relation_seq=rel_seq,
                                                        entity_seq=ent_seq,
                                                        path_partitions=partitions,
                                                        path_lengths=path_lengths,
                                                        num_words=num_words,
                                                        target_relations=target_relations,
                                                        labels=labels))
        probs += batch_prob[0].tolist()
        idxs += batch_idxs.tolist()

    pd.options.mode.chained_assignment = None
    selected = dataset.iloc[idxs]
    selected['prob'] = probs
    grouped = selected.groupby(['id']).agg({'label': list, 'prob': list})
    grouped['correct'] = grouped.apply(lambda row: int(np.argmax(row['label']) == np.argmax(row['prob'])), axis=1)
    print(grouped.loc[:10, ['label', 'prob']])
    accuracy = grouped['correct'].sum() / len(grouped)
    return accuracy


def train(dataset,
          document_store,
          model_params,
          train_params,
          sess_config):
    relation_token_embeddings = Word2VecEmbeddings(train_params.word_embd_path,
                                                   name='token_embd',
                                                   unk_token=UNK,
                                                   trainable=False,
                                                   special_tokens=[(ENT_1, False),
                                                                   (ENT_2, False),
                                                                   (ENT_X, False),
                                                                   (UNK, False),
                                                                   (END, False),
                                                                   (PAD, True)])
    target_relation_embeddings = RandomEmbeddings(dataset['relation'],
                                                  name='target_rel_emb',
                                                  embedding_size=model_params.path_rnn_hidden_size,
                                                  unk_token=None,
                                                  initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                              stddev=1.0,
                                                                                              dtype=tf.float64))
    if model_params.use_entity_type:
        entity_embeddings = EntityTypeEmbeddings(entity2type_path=train_params.entity2type_path,
                                                 name='entity_type_embd',
                                                 embedding_size=model_params.entity_type_embd_size,
                                                 unk_token=PROTEIN,
                                                 initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                             stddev=1.0,
                                                                                             dtype=tf.float64))
    else:
        entity_embeddings = relation_token_embeddings

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
                                    relation_token_embeddings=relation_token_embeddings,
                                    entity_embeddings=entity_embeddings,
                                    target_relation_embeddings=target_relation_embeddings,
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

    model, train_step = build_graph(
        relation_token_embeddings_tensor=relation_token_embeddings.get_embedding_matrix_tensor(),
        entity_embeddings_tensor=entity_embeddings.get_embedding_matrix_tensor(),
        target_relation_embeddings_tensor=target_relation_embeddings.get_embedding_matrix_tensor(),
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

        acc = evaluate_medhop_accuracy(dataset, sess, model, batch_generator, BatchGenerator.TRAIN_EVAL)
        print('Initial train medhop accuracy: {}'.format(acc))
        loss = evaluate(sess, model, batch_generator, BatchGenerator.TRAIN_EVAL)
        print('Initial train evaluation loss: {}'.format(loss))

        acc = evaluate_medhop_accuracy(dataset, sess, model, batch_generator, BatchGenerator.TEST)
        print('Initial test medhop accuracy: {}'.format(acc))
        loss = evaluate(sess, model, batch_generator, BatchGenerator.TEST)
        print('Initial test loss: {}'.format(loss))

        while batch_generator.get_epochs_completed(BatchGenerator.TRAIN) < train_params.epochs:
            epoch = batch_generator.get_epochs_completed(BatchGenerator.TRAIN)
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
                print('Epoch {} mean train loss: {}'.format(epoch,
                                                            total_train_loss / batch_generator.get_batch_count(
                                                                BatchGenerator.TRAIN)))

                acc = evaluate_medhop_accuracy(dataset, sess, model, batch_generator, BatchGenerator.TRAIN_EVAL)
                print('Epoch {} train medhop accuracy: {}'.format(epoch, acc))
                loss = evaluate(sess, model, batch_generator, BatchGenerator.TRAIN_EVAL)
                print('Epoch {} train evaluation loss: {}'.format(epoch, loss))

                acc = evaluate_medhop_accuracy(dataset, sess, model, batch_generator, BatchGenerator.TEST)
                print('Epoch {} test medhop accuracy: {}'.format(epoch, acc))
                loss = evaluate(sess, model, batch_generator, BatchGenerator.TEST)
                print('Epoch {} test loss: {}'.format(epoch, loss))

                total_train_loss = 0
                print()

            if step % train_params.check_period == 0:
                if batch_generator.train_eval_size > 0:
                    loss = evaluate(sess, model, batch_generator, BatchGenerator.TRAIN_EVAL)
                    print(
                        'Epoch: {} Step: {} Train evaluation Loss={}'.format(
                            epoch,
                            step,
                            loss))
                if batch_generator.test_size > 0:
                    loss = evaluate(sess, model, batch_generator, BatchGenerator.TEST)
                    print(
                        'Epoch: {} Step: {} Test Loss={}'.format(
                            epoch,
                            step,
                            loss))
                print()


if __name__ == '__main__':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list='0',
                                                      per_process_gpu_memory_fraction=1.0))

    train_document_store = pickle.load(open('./data/train_doc_store_punkt.pickle',
                                            'rb'))
    train_dataset = pd.read_json(
        './data/sentwise=F_cutoff=4_limit=100_method=shortest_tokenizer=punkt_medhop_train.json')
    print('Tuples count: {}'.format(len(train_dataset)))
    print('Maximum document length: {}'.format(train_document_store.max_tokens))

    #train_dataset = train_dataset.sample(100, random_state=10)

    train_params = TrainParameters(optimizer=Optimizer.ADAM,
                                   learning_rate=5e-4,
                                   epochs=1,
                                   batch_size=100,
                                   test_prop=0.1,
                                   train_eval_prop=1.0,
                                   check_period=50,
                                   word_embd_path='medhop_word2vec_punkt',
                                   debug_mode=False,
                                   train_eval_batch_size=450)

    print('Train params:')
    train_params.print()
    print()

    model_params = ModelParameters(
        max_path_length=5,
        max_relation_length=train_document_store.max_tokens,
        path_rnn_hidden_size=150,
        path_keep_prob=1.0,
        aggregator=PathRNN.LOG_SUM_EXP,
        relation_only=False,
        recurrent_relation_embedder=True,
        relation_rnn_hidden_size=150,
        relation_keep_prob=1.0,
        truncate_documents=True,
        label_smoothing=0.0,
        use_entity_type=True,
        entity_type_embd_size=50)

    print('Model params:')
    model_params.print()
    print()

    train(dataset=train_dataset,
          document_store=train_document_store,
          model_params=model_params,
          train_params=train_params,
          sess_config=config)
