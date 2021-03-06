import numpy as np

from parsing.special_tokens import *
from path_rnn_v2.experiments.experiment_base import run_model
from path_rnn_v2.experiments.truncated_relation_neighb_models.util import max_entity_mentions

np.random.seed(0)


def model_params_generator(max_path_len, max_rel_len, max_ent_len, word2vec_embeddings,
                           target_embeddings, emb_dim, dropout, args):
    return {
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
        'rel_encoder_name': 'node_encoder',
        'relation_encoder_params': {
            'module': 'additive_attention',
            'name': 'node_lstm_attention_encoder',
            'repr_dim': emb_dim,
            'activation': None,
            'dropout': None,
            'extra_args': {
                'with_backward': True,
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
        'ent_encoder_name': 'node_encoder',
        'entity_encoder_params': {
            'module': 'additive_attention',
            'name': 'node_lstm_attention_encoder',
            'repr_dim': emb_dim,
            'activation': None,
            'dropout': None,
            'extra_args': {
                'with_backward': True,
                'with_projection': False
            }
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


if __name__ == '__main__':
    visible_device_list = '0'
    visible_devices = '1'
    memory_fraction = 0.5

    model_name = 'attention_truncated_relation_neighb'
    extra_parser_args_adder = lambda parser: parser.add_argument('--neighb_dim',
                                                                 type=int,
                                                                 default=3,
                                                                 help='Entity neighborhood dimension')
    extra_args_formatter = lambda args: 'neighb_dim={}'.format(args.neighb_dim)

    max_ent_len_retrieve = lambda train_doc_store, dev_doc_store, args: ((2 * args.neighb_dim + 1)
                                                                         * max(max_entity_mentions(train_doc_store),
                                                                               max_entity_mentions(dev_doc_store)))
    max_rel_len_retrieve = lambda train_doc_store, dev_doc_store, args: 520

    rel_retrieve_params = {
        'replacement': (ENT_1, ENT_2),
        'truncate': True
    }
    ent_retrieve_params = lambda args: {'neighb_size': args.neighb_dim,
                                        'replacement': ENT_1,
                                        'end_token': NEIGHB_END}

    tensor_dict_map = {'rel_seq': 'rel_seq',
                       'ent_seq': 'ent_seq',
                       'ent_len': 'ent_len',
                       'rel_len': 'rel_len',
                       'seq_len': 'seq_len',
                       'target_rel': 'target_rel'}

    run_model(visible_device_list=visible_device_list, visible_devices=visible_devices, memory_fraction=memory_fraction,
              model_name=model_name, extra_parser_args_adder=extra_parser_args_adder,
              extra_args_formatter=extra_args_formatter, max_ent_len_retrieve=max_ent_len_retrieve,
              max_rel_len_retrieve=max_rel_len_retrieve, rel_retrieve_params=rel_retrieve_params,
              ent_retrieve_params=ent_retrieve_params, tensor_dict_map=tensor_dict_map,
              model_params_generator=model_params_generator)
