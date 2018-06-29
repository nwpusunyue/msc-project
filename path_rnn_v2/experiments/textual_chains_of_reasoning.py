import os
import pickle
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from parsing.special_tokens import *
from path_rnn_v2.nn.textual_chains_of_reasoning_model import TextualChainsOfReasoningModel
from path_rnn_v2.util.batch_generator import PartitionBatchGenerator
from path_rnn_v2.util.embeddings import RandomEmbeddings, Word2VecEmbeddings
from path_rnn_v2.util.tensor_generator import get_medhop_tensors
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def medhop_accuracy(dataset, probs):
    dataset['prob'] = probs
    grouped = dataset.groupby(['id']).agg({'label': list, 'prob': list})
    grouped['correct'] = grouped.apply(lambda row: int(np.argmax(row['label']) == np.argmax(row['prob'])), axis=1)
    accuracy = grouped['correct'].sum() / len(grouped)
    return accuracy


if __name__ == '__main__':
    emb_dim = 200
    max_path_len = 5
    max_ent_len = 1
    batch_size = 20
    path = './data'
    num_epochs = 50

    train = pd.read_json(
        '{}/sentwise=F_cutoff=4_limit=100_method=shortest_tokenizer=punkt_medhop_train.json'.format(path))
    dev = pd.read_json(
        '{}/sentwise=F_cutoff=4_limit=100_method=shortest_tokenizer=punkt_medhop_dev.json'.format(path))

    train_document_store = pickle.load(open('{}/train_doc_store_punkt.pickle'.format(path), 'rb'))
    dev_document_store = pickle.load(open('{}/dev_doc_store_punkt.pickle'.format(path), 'rb'))

    max_rel_len = train_document_store.max_tokens

    word2vec_embeddings = Word2VecEmbeddings('./medhop_word2vec_punkt',
                                             name='token_embd',
                                             unk_token=UNK,
                                             trainable=False,
                                             special_tokens=[(ENT_1, False),
                                                             (ENT_2, False),
                                                             (ENT_X, False),
                                                             (UNK, False),
                                                             (END, False),
                                                             (PAD, True)])
    target_embeddings = RandomEmbeddings(train['relation'],
                                         name='target_rel_emb',
                                         embedding_size=emb_dim,
                                         unk_token=None,
                                         initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                     stddev=1.0,
                                                                                     dtype=tf.float64))

    train_tensors = get_medhop_tensors(train['relation_paths'],
                                       train['entity_paths'],
                                       train['relation'],
                                       train['label'],
                                       train_document_store,
                                       word2vec_embeddings,
                                       word2vec_embeddings,
                                       target_embeddings,
                                       max_path_len=max_path_len,
                                       max_rel_len=max_rel_len,
                                       max_ent_len=max_ent_len,
                                       rel_retrieve_params={
                                           'replacement': (ENT_1, ENT_2),
                                           'truncate': False
                                       },
                                       ent_retrieve_params={
                                           'neighb_size': 0
                                       })
    dev_tensors = get_medhop_tensors(dev['relation_paths'],
                                     dev['entity_paths'],
                                     dev['relation'],
                                     dev['label'],
                                     dev_document_store,
                                     word2vec_embeddings,
                                     word2vec_embeddings,
                                     target_embeddings,
                                     max_path_len=max_path_len,
                                     max_rel_len=max_rel_len,
                                     max_ent_len=max_ent_len,
                                     rel_retrieve_params={
                                         'replacement': (ENT_1, ENT_2),
                                         'truncate': False
                                     },
                                     ent_retrieve_params={
                                         'neighb_size': 0
                                     })

    (rel_seq, ent_seq, path_len, rel_len, ent_len, target_rel, partition, label) = train_tensors
    train_batch_generator = PartitionBatchGenerator(partition, label, batch_size=batch_size, permute=True,
                                                    tensor_dict={'rel_seq': rel_seq,
                                                                 'ent_seq': ent_seq,
                                                                 'seq_len': path_len,
                                                                 'rel_len': rel_len,
                                                                 'ent_len': ent_len,
                                                                 'target_rel': target_rel})
    train_eval_batch_generator = PartitionBatchGenerator(partition, label, batch_size=batch_size, permute=False,
                                                         tensor_dict={'rel_seq': rel_seq,
                                                                      'ent_seq': ent_seq,
                                                                      'seq_len': path_len,
                                                                      'rel_len': rel_len,
                                                                      'ent_len': ent_len,
                                                                      'target_rel': target_rel})
    (rel_seq, ent_seq, path_len, rel_len, ent_len, target_rel, partition, label) = dev_tensors
    dev_batch_generator = PartitionBatchGenerator(partition, label, batch_size=batch_size, permute=False,
                                                  tensor_dict={'rel_seq': rel_seq,
                                                               'ent_seq': ent_seq,
                                                               'seq_len': path_len,
                                                               'rel_len': rel_len,
                                                               'ent_len': ent_len,
                                                               'target_rel': target_rel})

    model_params = {
        'max_path_len': max_path_len,
        'max_rel_len': max_rel_len,
        'max_ent_len': max_ent_len,
        'relation_embedder': word2vec_embeddings,
        'relation_embedder_params': {
            'max_norm': None,
            'with_projection': True,
            'projection_activation': 'tanh',
            'projection_dim': None,
            'name': 'node_embedder',
            'reuse': False
        },
        'relation_encoder_params': {
            'module': 'average',
            'repr_dim': emb_dim,
            'activation': None,
            'dropout': None,
            'extra_args': {
                'with_backward': False,
                'with_projection': False
            }
        },
        'entity_embedder': word2vec_embeddings,
        'entity_embedder_params': {
            'max_norm': None,
            'with_projection': True,
            'projection_activation': 'tanh',
            'projection_dim': None,
            'name': 'node_embedder',
            'reuse': True
        },
        'entity_encoder_params': {
            'module': 'identity',
            'activation': None,
            'dropout': None
        },
        'target_embedder': target_embeddings,
        'target_embedder_params': {
            'max_norm': None,
            'with_projection': True,
            'projection_activation': 'tanh',
            'projection_dim': None,
            'name': 'target_relation_embedder',
            'reuse': False
        },
        'path_encoder_params': {
            'repr_dim': emb_dim,
            'module': 'lstm',
            'activation': None,
            'dropout': None,
            'extra_args': {
                'with_backward': False,
                'with_projection': False
            }
        },
        'path_rnn_params': {
            'aggregator': 'logsumexp',
            'k': None
        }
    }

    train_params = {
        'optimizer': tf.train.AdamOptimizer(learning_rate=1e-3),
        'l2': 0.0,
        'clip_op': None,
        'clip': None
    }

    model = TextualChainsOfReasoningModel(model_params, train_params)
    print(model.params_str)

    steps = train_batch_generator.batch_count * num_epochs
    check_period = 10
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list='0',
                                                      per_process_gpu_memory_fraction=0.5))
    with tf.train.MonitoredTrainingSession(config=config) as sess:
        summ_writer = tf.summary.FileWriter('./textual_chains_of_reasoning_logs/run_{}'.format(time.strftime('%X_%x')))
        summ_writer.add_graph(tf.get_default_graph())
        for i in tqdm(range(steps)):
            batch = train_batch_generator.get_batch()

            batch_loss = model.train_step(batch, sess, summ_writer=summ_writer)

            if i % check_period == 0:
                print('Step: {}'.format(i))
                train_eval_prob = np.array([])

                for j in range(train_eval_batch_generator.batch_count):
                    batch = train_eval_batch_generator.get_batch()
                    model.eval_step(batch, sess)
                    train_eval_prob = np.concatenate((train_eval_prob, model.predict_step(batch, sess)))

                metrics = model.eval_step(batch=None, sess=sess, reset=True, summ_writer=summ_writer,
                                          summ_collection='summary_train_eval')

                print('Train loss: {} Train medhop acc: {}'.format(metrics, medhop_accuracy(train, train_eval_prob)))

                dev_prob = np.array([])

                for j in range(dev_batch_generator.batch_count):
                    batch = dev_batch_generator.get_batch()
                    model.eval_step(batch, sess)
                    dev_prob = np.concatenate((dev_prob, model.predict_step(batch, sess)))

                metrics = model.eval_step(batch=None, sess=sess, reset=True, summ_writer=summ_writer,
                                          summ_collection='summary_test')

                print('Dev loss: {} Dev medhop acc: {}'.format(metrics, medhop_accuracy(dev, dev_prob)))
