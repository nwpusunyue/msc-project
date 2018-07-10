import argparse
import os
import pickle
import pprint

import numpy as np
import pandas as pd
import tensorflow as tf

from path_rnn_v2.nn.distance_chains_of_reasoning_model import DistanceChainsOfReasoningModel
from path_rnn_v2.util.batch_generator import PartitionBatchGenerator
from path_rnn_v2.util.embeddings import RandomEmbeddings
from path_rnn_v2.util.tensor_generator import get_medhop_distance_tensors

np.random.seed(0)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
pd.set_option('display.max_colwidth', -1)


def medhop_accuracy(dataset, probs):
    dataset['prob'] = probs
    grouped = dataset.groupby(['id']).agg({'label': list, 'prob': list, 'source': list})
    grouped['correct'] = grouped.apply(lambda row: int(np.argmax(row['label']) == np.argmax(row['prob'])), axis=1)
    accuracy = grouped['correct'].sum() / len(grouped)
    return accuracy


# Instantiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('model_path',
                    type=str,
                    help='Path to model')
parser.add_argument('--emb_dim',
                    type=int,
                    default=200,
                    help='Embedding dimension')
parser.add_argument('--l2',
                    type=float,
                    default=0.0,
                    help='L2 loss regularization')
parser.add_argument('--dropout',
                    type=float,
                    default=0.0,
                    help='Path rnn dropout prob')

parser.add_argument('--paths_selection',
                    type=str,
                    default='shortest',
                    help='How the paths in the dataset were generated')

if __name__ == '__main__':
    # cmd line args
    args = parser.parse_args()
    emb_dim = args.emb_dim
    l2 = args.l2
    dropout = args.dropout
    method = args.paths_selection

    max_path_len = 4
    batch_size = 32
    path = './data'
    num_epochs = 50

    limit = 100

    train = pd.read_json(
        '{}/sentwise=F_cutoff=4_limit={}_method={}_tokenizer=punkt_medhop_train.json'.format(path, limit, method))
    dev = pd.read_json(
        '{}/sentwise=F_cutoff=4_limit={}_method={}_tokenizer=punkt_medhop_dev.json'.format(path, limit, method))
    test = pd.read_json(
        '{}/sentwise=F_cutoff=4_limit={}_method={}_tokenizer=punkt_medhop_test.json'.format(path, limit, method))

    train_document_store = pickle.load(open('{}/train_doc_store_punkt.pickle'.format(path), 'rb'))
    dev_document_store = pickle.load(open('{}/dev_doc_store_punkt.pickle'.format(path), 'rb'))
    test_document_store = pickle.load(open('{}/test_doc_store_punkt.pickle'.format(path), 'rb'))

    target_embeddings = RandomEmbeddings(train['relation'],
                                         name='target_rel_emb',
                                         embedding_size=emb_dim,
                                         unk_token=None,
                                         initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                     stddev=1.0,
                                                                                     dtype=tf.float64))

    train_tensors = get_medhop_distance_tensors(query_rel_seq=train['relation_paths'],
                                                query_ent_seq=train['entity_paths'],
                                                query_target_rel=train['relation'],
                                                query_label=train['label'],
                                                max_path_len=max_path_len,
                                                target_rel_embeddings=target_embeddings,
                                                document_store=train_document_store)

    dev_tensors = get_medhop_distance_tensors(query_rel_seq=dev['relation_paths'],
                                              query_ent_seq=dev['entity_paths'],
                                              query_target_rel=dev['relation'],
                                              query_label=dev['label'],
                                              max_path_len=max_path_len,
                                              target_rel_embeddings=target_embeddings,
                                              document_store=dev_document_store)

    test_tensors = get_medhop_distance_tensors(query_rel_seq=test['relation_paths'],
                                               query_ent_seq=test['entity_paths'],
                                               query_target_rel=test['relation'],
                                               query_label=test['label'],
                                               max_path_len=max_path_len,
                                               target_rel_embeddings=target_embeddings,
                                               document_store=test_document_store)

    (rel_seq, path_len, target_rel, partition, label) = train_tensors
    train_eval_batch_generator = PartitionBatchGenerator(partition, label, batch_size=batch_size, permute=False,
                                                         tensor_dict={'rel_seq': rel_seq,
                                                                      'seq_len': path_len,
                                                                      'target_rel': target_rel})
    (rel_seq, path_len, target_rel, partition, label) = dev_tensors
    dev_batch_generator = PartitionBatchGenerator(partition, label, batch_size=batch_size, permute=False,
                                                  tensor_dict={'rel_seq': rel_seq,
                                                               'seq_len': path_len,
                                                               'target_rel': target_rel})
    (rel_seq, path_len, target_rel, partition, label) = test_tensors
    test_batch_generator = PartitionBatchGenerator(partition, label, batch_size=batch_size, permute=False,
                                                  tensor_dict={'rel_seq': rel_seq,
                                                               'seq_len': path_len,
                                                               'target_rel': target_rel})
    model_params = {
        'max_path_len': max_path_len,
        'target_embedder': target_embeddings,
        'target_embedder_params': {
            'max_norm': None,
            'with_projection': True,
            'projection_activation': 'tanh',
            'projection_dim': None,
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
        'optimizer': tf.train.AdamOptimizer(learning_rate=1e-2),
        'l2': l2,
        'clip_op': None,
        'clip': None
    }

    model = DistanceChainsOfReasoningModel(model_params, train_params)
    print(model.params_str)
    print(pprint.pformat(model.train_variables))

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list='0',
                                                      per_process_gpu_memory_fraction=0.3))

    print('Evaluating model: {}'.format(args.model_path))
    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
        model.load(sess, path=args.model_path)

        train_eval_prob = np.array([])

        for j in range(train_eval_batch_generator.batch_count):
            batch = train_eval_batch_generator.get_batch()
            model.eval_step(batch, sess)
            train_eval_prob = np.concatenate((train_eval_prob, model.predict_step(batch, sess)))

        metrics = model.eval_step(batch=None, sess=sess, reset=True)
        train_eval_medhop_acc = medhop_accuracy(train, train_eval_prob)

        print('Train loss: {} Train medhop acc: {}'.format(metrics, train_eval_medhop_acc))

        dev_prob = np.array([])

        for j in range(dev_batch_generator.batch_count):
            batch = dev_batch_generator.get_batch()
            model.eval_step(batch, sess)
            dev_prob = np.concatenate((dev_prob, model.predict_step(batch, sess)))

        metrics = model.eval_step(batch=None, sess=sess, reset=True)

        dev_medhop_acc = medhop_accuracy(dev, dev_prob)

        print('Dev loss: {} Dev medhop acc: {}'.format(metrics, dev_medhop_acc))

        test_prob = np.array([])

        for j in range(test_batch_generator.batch_count):
            batch = test_batch_generator.get_batch()
            model.eval_step(batch, sess)
            test_prob = np.concatenate((test_prob, model.predict_step(batch, sess)))

        metrics = model.eval_step(batch=None, sess=sess, reset=True)

        test_medhop_acc = medhop_accuracy(test, test_prob)

        print('Test loss: {} Test medhop acc: {}'.format(metrics, test_medhop_acc))
