import argparse
import os
import pickle
import pprint
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


np.random.seed(0)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# pd.set_option('display.max_colwidth', -1)

# Instantiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('--emb_dim',
                    type=int,
                    default=200,
                    help='Size of path-rnn embeddings')
parser.add_argument('--l2',
                    type=float,
                    default=0.0,
                    help='L2 loss regularization')
parser.add_argument('--dropout',
                    type=float,
                    default=0.0,
                    help='Path rnn dropout prob')


def medhop_accuracy(dataset, probs):
    dataset['prob'] = probs
    grouped = dataset.groupby(['id']).agg({'label': list, 'prob': list})
    grouped['correct'] = grouped.apply(lambda row: int(np.argmax(row['label']) == np.argmax(row['prob'])), axis=1)
    accuracy = grouped['correct'].sum() / len(grouped)
    return accuracy


if __name__ == '__main__':
    # cmd line args
    args = parser.parse_args()
    emb_dim = args.emb_dim
    l2 = args.l2
    dropout = args.dropout

    max_path_len = 5
    max_ent_len = 1
    batch_size = 20
    path = './data'
    num_epochs = 50

    limit = 100
    method = 'shortest'

    train = pd.read_json(
        '{}/sentwise=F_cutoff=4_limit={}_method={}_tokenizer=punkt_medhop_train.json'.format(path, limit, method))
    dev = pd.read_json(
        '{}/sentwise=F_cutoff=4_limit={}_method={}_tokenizer=punkt_medhop_dev.json'.format(path, limit, method))

    train_document_store = pickle.load(open('{}/train_doc_store_punkt.pickle'.format(path), 'rb'))
    dev_document_store = pickle.load(open('{}/dev_doc_store_punkt.pickle'.format(path), 'rb'))

    max_rel_len = max(train_document_store.max_tokens, dev_document_store.max_tokens)

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
            'name': 'relation_average_encoder',
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
            'name': 'entity_identity_encoder',
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
            'name': 'path_encoder',
            'activation': None,
            'dropout': dropout,
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
        'l2': l2,
        'clip_op': None,
        'clip': None
    }

    model = TextualChainsOfReasoningModel(model_params, train_params)
    print(model.params_str)
    print(pprint.pformat(model.train_variables))

    steps = train_batch_generator.batch_count * num_epochs
    check_period = 20
    # config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list='0',
    #                                                   per_process_gpu_memory_fraction=1.0))

    medhop_acc = []
    max_dev_medhop_acc = 0.0

    start_time = time.strftime('%X_%d.%m.%y')
    run_id = 'run_{}_emb_dim={}_l2={}_drop={}'.format(start_time, emb_dim, l2, dropout)
    print('Run id: {}'.format(run_id))
    model_dir = './textual_chains_of_reasoning_models/{}'.format(run_id)
    log_dir = './textual_chains_of_reasoning_logs/{}'.format(run_id)
    acc_dir = './textual_chains_of_reasoning_logs/acc_{}.txt'.format(run_id)

    # make save dir
    os.makedirs(model_dir)
    # make summary writercl
    summ_writer = tf.summary.FileWriter(log_dir)
    summ_writer.add_graph(tf.get_default_graph())
    # make acc file
    acc_file = open(acc_dir, 'w+')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

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
                train_eval_medhop_acc = medhop_accuracy(train, train_eval_prob)

                print('Train loss: {} Train medhop acc: {}'.format(metrics, train_eval_medhop_acc))

                dev_prob = np.array([])

                for j in range(dev_batch_generator.batch_count):
                    batch = dev_batch_generator.get_batch()
                    model.eval_step(batch, sess)
                    dev_prob = np.concatenate((dev_prob, model.predict_step(batch, sess)))

                metrics = model.eval_step(batch=None, sess=sess, reset=True, summ_writer=summ_writer,
                                          summ_collection='summary_test')

                dev_medhop_acc = medhop_accuracy(dev, dev_prob)

                print('Dev loss: {} Dev medhop acc: {}'.format(metrics, dev_medhop_acc))

                medhop_acc.append((train_eval_medhop_acc, dev_medhop_acc))
                acc_file.write('{}: tr: {} dev: {}\n'.format(i, train_eval_medhop_acc, dev_medhop_acc))
                acc_file.flush()

                if dev_medhop_acc > max_dev_medhop_acc:
                    print('Storing model with best dev medhop acc at: {}'.format(model_dir))
                    max_dev_medhop_acc = dev_medhop_acc
                    model.store(sess, '{}/model'.format(model_dir))
