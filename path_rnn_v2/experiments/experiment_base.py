import argparse
import os
import pickle
import pprint

import numpy as np
import pandas as pd
import tensorflow as tf

from parsing.special_tokens import *
from path_rnn_v2.experiments.eval.evaluation import evaluate_model
from path_rnn_v2.experiments.prediction.prediction import generate_prediction
from path_rnn_v2.experiments.training import train_model
from path_rnn_v2.nn.textual_chains_of_reasoning_model import TextualChainsOfReasoningModel
from path_rnn_v2.util.batch_generator import PartitionBatchGenerator, ProportionedPartitionBatchGenerator, \
    ProportionedSubsamplingPartitionBatchGenerator
from path_rnn_v2.util.embeddings import RandomEmbeddings, AugmentedWord2VecEmbeddings, EntityTypeAugmenter
from path_rnn_v2.util.tensor_generator import get_medhop_tensors

np.random.seed(0)


def get_basic_parser():
    # Instantiate the parser
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument('--emb_dim',
                        type=int,
                        default=100,
                        help='Size of path-rnn embeddings')
    parser.add_argument('--l2',
                        type=float,
                        default=0.0,
                        help='L2 loss regularization')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.0,
                        help='Path rnn dropout prob')
    parser.add_argument('--tokenizer',
                        type=str,
                        default='punkt',
                        help='Tokenizer used for the dataset')
    parser.add_argument('--word_embd_path',
                        type=str,
                        default='./medhop_word2vec_punkt_v2',
                        help='Word embedding path')
    parser.add_argument('--entity_augment',
                        action='store_true',
                        help='If this is set, the entity type is concatenated to the word embedding')

    # train params
    parser.add_argument('--batch_size',
                        type=int,
                        default=20,
                        help='Batch size')
    parser.add_argument('--eval_batch_size',
                        type=int,
                        default=20,
                        help='Evaluation batch size')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=25,
                        help='Number of epochs')
    parser.add_argument('--subsample',
                        action='store_true',
                        help='Whether to use path subsampling in the train batches')
    parser.add_argument('--partition_limit',
                        type=int,
                        default=50,
                        help='Max number of paths for an example per epoch')
    parser.add_argument('--check_period',
                        type=int,
                        default=50,
                        help='How many steps between the evaluation of dev metrics')

    # dataset params
    parser.add_argument('--paths_selection',
                        type=str,
                        default='shortest',
                        help='How the paths in the dataset were generated')
    parser.add_argument('--masked',
                        action='store_true',
                        help='If set, the masked datasets are used')
    parser.add_argument('--source_target_only',
                        action='store_true',
                        help='If set, the datasets with source-target only chains will be loaded')
    parser.add_argument('--limit',
                        type=int,
                        default=100,
                        help='Max number of paths')

    # aux
    parser.add_argument('--testing',
                        action='store_true',
                        help='If this is set, testing is run instead of training')
    parser.add_argument('--predicting',
                        action='store_true',
                        help='If this is set, predicting is run instead of training')
    parser.add_argument('--no_gpu_conf',
                        action='store_true',
                        help='If this is set, no gpu options will be passed in')
    parser.add_argument('--model_path',
                        default=None,
                        help='Model path if testing')
    parser.add_argument('--eval_file_path',
                        default=None,
                        help='File to append evaluation results to')
    parser.add_argument('--base_dir',
                        default='.',
                        help='Base directory for saving the model and the logs.')
    parser.add_argument('--no_save',
                        action='store_true',
                        help='If set no data will be saved.')
    return parser


def get_gpu_config(visible_devices="0", visible_device_list="0", memory_fraction=1.0):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list=visible_device_list,
                                                      per_process_gpu_memory_fraction=memory_fraction))
    return config


def read_dataset(path, limit, method, tokenizer, masked, type, source_target_only=False):
    masked = '' if not masked else '_masked'
    source_target_only = '' if not source_target_only else '_stupid'
    dataset_path = '{}/sentwise=F_cutoff=4_limit={}_method={}_tokenizer={}_medhop_{}{}{}.json'.format(path,
                                                                                                      limit,
                                                                                                      method,
                                                                                                      tokenizer,
                                                                                                      type,
                                                                                                      masked,
                                                                                                      source_target_only)
    dataset = pd.read_json(dataset_path)
    document_store = pickle.load(open('{}/{}{}_doc_store_{}.pickle'.format(path, type, masked, tokenizer), 'rb'))
    return dataset, document_store


def get_tensors(df, document_store, relation_embeddings, entity_embeddings, target_embeddings, max_path_len,
                max_rel_len, max_ent_len, rel_retrieve_params, ent_retrieve_params):
    if 'label' in df.columns:
        label = df['label']
    else:
        label = None
    return get_medhop_tensors(query_rel_seq=df['relation_paths'],
                              query_ent_seq=df['entity_paths'],
                              query_target_rel=df['relation'],
                              query_label=label,
                              document_store=document_store,
                              rel_embeddings=relation_embeddings,
                              ent_embeddings=entity_embeddings,
                              target_rel_embeddings=target_embeddings,
                              max_path_len=max_path_len,
                              max_rel_len=max_rel_len,
                              max_ent_len=max_ent_len,
                              rel_retrieve_params=rel_retrieve_params,
                              ent_retrieve_params=ent_retrieve_params)


def get_train_batch(train_tensors, batch_size, tensor_dict_map, subsampled=False, partition_limit=50):
    (rel_seq, ent_seq, seq_len, rel_len, ent_len, target_rel, partition, label) = train_tensors
    pos = len(np.argwhere(label == 1))
    neg = len(np.argwhere(label == 0))
    positive_prop = float(pos) / neg
    tensor_dict = {}
    for k, v in tensor_dict_map.items():
        tensor_dict[k] = locals()[v]
    if not subsampled:
        train_batch_generator = ProportionedPartitionBatchGenerator(partition, label, batch_size=batch_size,
                                                                    permute=True,
                                                                    positive_prop=positive_prop,
                                                                    tensor_dict=tensor_dict)
    else:
        train_batch_generator = ProportionedSubsamplingPartitionBatchGenerator(partition, label, batch_size=batch_size,
                                                                               permute=True,
                                                                               positive_prop=positive_prop,
                                                                               tensor_dict=tensor_dict,
                                                                               partition_limit=partition_limit)
    print('Positives per batch: {} \nNegatives per batch: {}'.format(train_batch_generator.positive_batch_size,
                                                                     train_batch_generator.negative_batch_size))
    return train_batch_generator


def get_batch(tensors, batch_size, tensor_dict_map, with_label=True):
    if with_label:
        (rel_seq, ent_seq, seq_len, rel_len, ent_len, target_rel, partition, label) = tensors
    else:
        (rel_seq, ent_seq, seq_len, rel_len, ent_len, target_rel, partition) = tensors
    tensor_dict = {}
    for k, v in tensor_dict_map.items():
        tensor_dict[k] = locals()[v]
    return PartitionBatchGenerator(partition, label, batch_size=batch_size, permute=False,
                                   tensor_dict=tensor_dict)


def get_word2vec(word_embd_path, name, masked, entity_augment):
    special_tokens = [(ENT_1, False),
                      (ENT_2, False),
                      (ENT_X, False),
                      (UNK, False),
                      (END, False),
                      (NEIGHB_END, False),
                      (PAD, True)]
    if masked:
        special_tokens += [('___MASK{}___'.format(i), False) for i in range(100)]
    if entity_augment:
        augmenter = EntityTypeAugmenter('/home/scimpian/msc-project/parsing/entity_map.txt')
    else:
        augmenter = None
    return AugmentedWord2VecEmbeddings(word_embd_path,
                                       name=name,
                                       unk_token=UNK,
                                       trainable=False,
                                       special_tokens=special_tokens,
                                       augmenter=augmenter)


def get_target_embd(relations, name, embd_dim):
    return RandomEmbeddings(relations,
                            name=name,
                            embedding_size=embd_dim,
                            unk_token=None,
                            initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                        stddev=1.0,
                                                                        dtype=tf.float64))


def run_model(visible_device_list, visible_devices, memory_fraction,
              model_name, extra_parser_args_adder, extra_args_formatter,
              max_ent_len_retrieve, max_rel_len_retrieve, rel_retrieve_params, ent_retrieve_params,
              tensor_dict_map,
              model_params_generator):
    parser = get_basic_parser()
    extra_parser_args_adder(parser)

    args = parser.parse_args()
    emb_dim = args.emb_dim
    l2 = args.l2
    dropout = args.dropout
    tokenizer = args.tokenizer
    word_embd_path = args.word_embd_path
    entity_augment = args.entity_augment

    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    num_epochs = args.num_epochs
    subsample = args.subsample
    partition_limit = args.partition_limit
    check_period = args.check_period

    paths_selection = args.paths_selection
    masked = args.masked
    source_target_only = args.source_target_only
    limit = args.limit

    testing = args.testing
    predicting = args.predicting
    no_gpu_conf = args.no_gpu_conf
    model_path = args.model_path
    eval_file_path = args.eval_file_path
    base_dir = args.base_dir
    no_save = args.no_save

    if not no_gpu_conf:
        config = get_gpu_config(visible_device_list=visible_device_list,
                                visible_devices=visible_devices,
                                memory_fraction=memory_fraction)
    else:
        config = None

    path = './data'
    run_id_params = 'emb_dim={}_l2={}_drop={}_paths={}_tokenizer={}_masked={}_entity={}'.format(emb_dim,
                                                                                                l2,
                                                                                                dropout,
                                                                                                paths_selection,
                                                                                                tokenizer, masked,
                                                                                                entity_augment)
    extra_run_id_params = extra_args_formatter(args)
    run_id_params = run_id_params + '_' + extra_run_id_params

    train, train_document_store = read_dataset(path=path, limit=limit, method=paths_selection,
                                               tokenizer=tokenizer, masked=masked, type='train',
                                               source_target_only=source_target_only)
    dev, dev_document_store = read_dataset(path=path, limit=limit, method=paths_selection,
                                           tokenizer=tokenizer, masked=masked, type='dev',
                                           source_target_only=source_target_only)

    if testing:
        test, test_document_store = read_dataset(path=path, limit=limit, method=paths_selection,
                                                 tokenizer=tokenizer, masked=masked, type='test',
                                                 source_target_only=source_target_only)
    if predicting:
        test_no_label, test_no_label_document_store = read_dataset(path=path, limit=limit, method=paths_selection,
                                                                   tokenizer=tokenizer, masked=masked,
                                                                   type='test_no_label',
                                                                   source_target_only=source_target_only)

    max_path_len = 5 if not source_target_only else 2
    max_ent_len = max_ent_len_retrieve(train_document_store, dev_document_store, args)
    max_rel_len = max_rel_len_retrieve(train_document_store, dev_document_store, args)

    word2vec_embeddings = get_word2vec(word_embd_path=word_embd_path,
                                       name='token_emb',
                                       masked=masked,
                                       entity_augment=entity_augment)
    target_embeddings = get_target_embd(train['relation'],
                                        name='target_rel_embd',
                                        embd_dim=emb_dim)

    if type(ent_retrieve_params) is not dict:
        ent_retrieve_params = ent_retrieve_params(args)
    if type(rel_retrieve_params) is not dict:
        rel_retrieve_params = rel_retrieve_params(args)

    train_tensors = get_tensors(train, train_document_store, word2vec_embeddings, word2vec_embeddings,
                                target_embeddings, max_path_len, max_rel_len, max_ent_len, rel_retrieve_params,
                                ent_retrieve_params)
    dev_tensors = get_tensors(dev, dev_document_store, word2vec_embeddings, word2vec_embeddings, target_embeddings,
                              max_path_len, max_rel_len, max_ent_len, rel_retrieve_params, ent_retrieve_params)

    if testing:
        test_tensors = get_tensors(test, test_document_store, word2vec_embeddings, word2vec_embeddings,
                                   target_embeddings, max_path_len, max_rel_len, max_ent_len, rel_retrieve_params,
                                   ent_retrieve_params)
    if predicting:
        test_no_label_tensors = get_tensors(test_no_label, test_no_label_document_store, word2vec_embeddings,
                                            word2vec_embeddings, target_embeddings, max_path_len, max_rel_len,
                                            max_ent_len, rel_retrieve_params, ent_retrieve_params)

    train_batch_generator = get_train_batch(train_tensors, batch_size, tensor_dict_map, subsampled=subsample,
                                            partition_limit=partition_limit)
    train_eval_batch_generator = get_batch(train_tensors, eval_batch_size, tensor_dict_map, with_label=True)

    dev_batch_generator = get_batch(dev_tensors, eval_batch_size, tensor_dict_map, with_label=True)

    if testing:
        test_batch_generator = get_batch(test_tensors, eval_batch_size, tensor_dict_map, with_label=True)
    if predicting:
        test_no_label_batch_generator = get_batch(test_no_label_tensors, eval_batch_size, tensor_dict_map,
                                                  with_label=False)

    model_params = model_params_generator(max_path_len, max_rel_len, max_ent_len, word2vec_embeddings,
                                          target_embeddings, emb_dim, dropout, args)
    train_params = {
        'optimizer': tf.train.AdamOptimizer(learning_rate=1e-3),
        'l2': l2,
        'clip_op': None,
        'clip': None
    }
    model = TextualChainsOfReasoningModel(model_params, train_params)
    print(model.params_str)
    print(pprint.pformat(model.train_variables))

    if not testing and not predicting:
        train_model(model, train=train, train_batch_generator=train_batch_generator,
                    train_eval_batch_generator=train_eval_batch_generator,
                    dev=dev, dev_batch_generator=dev_batch_generator,
                    model_name=model_name, run_id_params=run_id_params,
                    num_epochs=num_epochs, check_period=check_period, config=config, base_dir=base_dir, no_save=no_save)
    elif not predicting:
        evaluate_model(model, model_path=model_path, train=train, train_eval_batch_generator=train_eval_batch_generator,
                       dev=dev, dev_batch_generator=dev_batch_generator,
                       test=test, test_batch_generator=test_batch_generator,
                       eval_file_path=eval_file_path, word_embd=None)
    else:
        generate_prediction(model, model_path=model_path, test=test_no_label,
                            test_batch_generator=test_no_label_batch_generator, eval_file_path=eval_file_path)
