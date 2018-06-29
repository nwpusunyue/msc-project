import os

import numpy as np
import pandas as pd
import tensorflow as tf

from itertools import chain
from path_rnn_v2.util.batch_generator import PartitionBatchGenerator
from path_rnn_v2.util.embeddings import RandomEmbeddings
from path_rnn_v2.util.ops import create_reset_metric
from path_rnn_v2.nn.path_rnn import PathRnn
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_dataset(pos_path, neg_path, dev_path):
    pos = pd.read_json(pos_path)
    neg = pd.read_json(neg_path)
    train = pd.concat([pos, neg], axis=0)
    dev = pd.read_json(dev_path)

    return train.reset_index(), dev


def get_numpy_arrays(embd, rel_paths, target_rel, label, max_path_len=None):
    path_partition = np.zeros((len(rel_paths), 2), dtype=int)

    total_paths = np.sum([len(rel_paths) for rel_paths in rel_paths])
    if max_path_len is None:
        max_path_len = np.max([np.max([len(path) for path in paths]) for paths in rel_paths])

    print('Max path len: {}'.format(max_path_len))
    print('Total paths: {}'.format(total_paths))

    path_len = np.zeros([total_paths], dtype=np.int)
    indexed_rel_path = np.zeros([total_paths, max_path_len], dtype=np.int)
    indexed_target_rel = np.zeros([total_paths], dtype=np.int)
    label = np.array(label, dtype=np.int)

    path_idx = 0
    for query_idx, (rels, target) in enumerate(zip(rel_paths,
                                                   target_rel)):
        partition_start = path_idx
        for (rel_path) in rels:
            path_len[path_idx] = len(rel_path)
            indexed_target_rel[path_idx] = embd.get_idx(target)

            for rel_idx, rel in enumerate(rel_path):
                indexed_rel_path[path_idx, rel_idx] = embd.get_idx(rel)
            path_idx += 1
        partition_end = path_idx
        path_partition[query_idx, 0] = partition_start
        path_partition[query_idx, 1] = partition_end

    return (path_partition, path_len, indexed_rel_path, indexed_target_rel, label)


def get_model(emb, max_path_len, embedder_params, encoder_params, l2=0.0, clip_op=None, clip=None):
    # [batch_size, max_path_len]
    rel_seq = tf.placeholder(tf.int64,
                             shape=[None, max_path_len])
    # [batch_size]
    seq_len = tf.placeholder(tf.int64,
                             shape=[None])
    # [batch_size]
    target_rel = tf.placeholder(tf.int64,
                                shape=[None])

    # [num_queries_in_batch]
    path_partition = tf.placeholder(tf.int64,
                                    shape=[None, 2])
    # [num_queries_in_batch]
    label = tf.placeholder(tf.int64,
                           shape=[None])

    # [batch_size, max_path_len, emb_dim]
    rel_seq_embd = emb.embed_sequence(rel_seq, name='relation_embedder', **embedder_params)

    # [batch_size, emb_dim]
    target_rel_embd = emb.embed_sequence(target_rel, name='relation_embedder', reuse=True, **embedder_params)

    path_rnn = PathRnn()

    # [num_queries_in_batch]
    score = path_rnn.get_output(rel_seq_embd, seq_len, path_partition, target_rel_embd, encoder_params)

    prob = tf.sigmoid(score)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(label, dtype=score.dtype),
                                                                  logits=score))

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

    if l2:
        vars = tf.trainable_variables()
        loss += tf.add_n([tf.nn.l2_loss(v) for v in vars]) * l2
    if clip:
        gradients = optimizer.compute_gradients(loss)
        if clip_op == tf.clip_by_value:
            gradients = [(tf.clip_by_value(grad, clip[0], clip[1]), var)
                         for grad, var in gradients if grad is not None]
        elif clip_op == tf.clip_by_norm:
            gradients = [(tf.clip_by_norm(grad, clip), var)
                         for grad, var in gradients if grad is not None]
        train_step = optimizer.apply_gradients(gradients)
    else:
        train_step = optimizer.minimize(loss)

    ###### evaluation related ######
    mean_loss, update_op_loss, reset_op_loss = create_reset_metric(tf.metrics.mean, 'mean_loss', values=loss)
    training_loss_summary = tf.summary.scalar('training_loss', mean_loss)
    dev_loss_summary = tf.summary.scalar('dev_loss', mean_loss)

    acc, update_op_acc, reset_op_acc = create_reset_metric(tf.metrics.accuracy, 'mean_acc', labels=label,
                                                           predictions=tf.round(prob))
    training_acc_summary = tf.summary.scalar('training_acc', acc)
    dev_acc_summary = tf.summary.scalar('dev_acc', acc)

    train_merged = tf.summary.merge([training_loss_summary, training_acc_summary])
    dev_merged = tf.summary.merge([dev_loss_summary, dev_acc_summary])

    placeholders = {
        'rel_seq': rel_seq,
        'seq_len': seq_len,
        'target_rel': target_rel,
        'path_partition': path_partition,
        'label': label
    }
    return placeholders, prob, loss, train_step, (mean_loss, update_op_loss, reset_op_loss), (
        acc, update_op_acc, reset_op_acc), train_merged, dev_merged


if __name__ == '__main__':
    train, dev = get_dataset(
        pos_path='./chains_of_reasoning_data/_people_person_nationality/parsed/positive_matrix.json',
        neg_path='./chains_of_reasoning_data/_people_person_nationality/parsed/negative_matrix.json',
        dev_path='./chains_of_reasoning_data/_people_person_nationality/parsed/dev_matrix.json')

    relation_vocab = set(chain.from_iterable(list(chain.from_iterable(train['relation_paths']))))
    relation_vocab.add('#UNK')
    relation_vocab.add('#END')
    embd_dim = 50
    random_embd = RandomEmbeddings(relation_vocab, embd_dim, 'rel_embd', '#UNK', tf.initializers.random_normal())

    (train_path_partition,
     train_path_len,
     train_indexed_rel_path,
     train_indexed_target_rel,
     train_label) = get_numpy_arrays(embd=random_embd,
                                     rel_paths=train[
                                         'relation_paths'],
                                     target_rel=train[
                                         'target_relation'],
                                     label=train['label'])

    (dev_path_partition,
     dev_path_len,
     dev_indexed_rel_path,
     dev_indexed_target_rel,
     dev_label) = get_numpy_arrays(embd=random_embd,
                                   rel_paths=dev[
                                       'relation_paths'],
                                   target_rel=dev[
                                       'target_relation'],
                                   label=dev['label'],
                                   max_path_len=train_indexed_rel_path.shape[1])

    train_batch_generator = PartitionBatchGenerator(partition=train_path_partition,
                                                    label=train_label,
                                                    tensor_dict={
                                                        'seq_len': train_path_len,
                                                        'rel_seq': train_indexed_rel_path,
                                                        'target_rel': train_indexed_target_rel
                                                    },
                                                    batch_size=100)
    train_eval_batch_generator = PartitionBatchGenerator(partition=train_path_partition,
                                                         label=train_label,
                                                         tensor_dict={
                                                             'seq_len': train_path_len,
                                                             'rel_seq': train_indexed_rel_path,
                                                             'target_rel': train_indexed_target_rel
                                                         },
                                                         batch_size=100,
                                                         permute=False)
    dev_batch_generator = PartitionBatchGenerator(partition=dev_path_partition,
                                                  label=dev_label,
                                                  tensor_dict={
                                                      'seq_len': dev_path_len,
                                                      'rel_seq': dev_indexed_rel_path,
                                                      'target_rel': dev_indexed_target_rel
                                                  },
                                                  batch_size=100,
                                                  permute=False)

    (placeholders,
     prob,
     loss,
     train_step,
     stream_loss,
     stream_acc,
     train_summaries,
     dev_summaries) = get_model(
        random_embd,
        max_path_len=train_indexed_rel_path.shape[1],
        embedder_params={
            'max_norm': None,
            'with_projection': False,
        },
        encoder_params={
            'repr_dim': embd_dim,
            'module': 'lstm',
            'extra_args': {
                'with_backward': False,
                'with_projection': False
            }
        },
        l2=0.0)

    check_period = 10
    with tf.train.MonitoredTrainingSession() as sess:
        writer = tf.summary.FileWriter('./chains_of_reasoning_logs/',
                                       sess.graph)
        for i in tqdm(range(1000)):
            tensor_dict = train_batch_generator.get_batch()

            batch_loss, _ = sess.run([loss, train_step],
                                     feed_dict={
                                         placeholders['rel_seq']: tensor_dict['rel_seq'],
                                         placeholders['seq_len']: tensor_dict['seq_len'],
                                         placeholders['target_rel']: tensor_dict['target_rel'],
                                         placeholders['path_partition']: tensor_dict['partition'],
                                         placeholders['label']: tensor_dict['label']
                                     })

            if i % check_period == 0:
                for j in range(train_eval_batch_generator.batch_count):
                    tensor_dict = train_eval_batch_generator.get_batch()
                    sess.run([stream_acc[1], stream_loss[1]],
                             feed_dict={
                                 placeholders['rel_seq']: tensor_dict['rel_seq'],
                                 placeholders['seq_len']: tensor_dict['seq_len'],
                                 placeholders['target_rel']: tensor_dict['target_rel'],
                                 placeholders['path_partition']: tensor_dict['partition'],
                                 placeholders['label']: tensor_dict['label']
                             })
                train_summ, mean_train_loss, mean_train_acc = sess.run([train_summaries, stream_loss[0], stream_acc[0]])
                sess.run([stream_loss[2], stream_acc[2]])

                for j in range(dev_batch_generator.batch_count):
                    tensor_dict = dev_batch_generator.get_batch()
                    sess.run([stream_acc[1], stream_loss[1]],
                             feed_dict={
                                 placeholders['rel_seq']: tensor_dict['rel_seq'],
                                 placeholders['seq_len']: tensor_dict['seq_len'],
                                 placeholders['target_rel']: tensor_dict['target_rel'],
                                 placeholders['path_partition']: tensor_dict['partition'],
                                 placeholders['label']: tensor_dict['label']
                             })
                dev_summ, mean_dev_loss, mean_dev_acc = sess.run([dev_summaries, stream_loss[0], stream_acc[0]])
                sess.run([stream_loss[2], stream_acc[2]])

                print('Step {} '
                      '\nTrain loss: {:0.2f} Train acc: {:0.2f} '
                      '\nDev   loss: {:0.2f} Dev   acc: {:0.2f}\n\n'.format(i,
                                                                            mean_train_loss,
                                                                            mean_train_acc,
                                                                            mean_dev_loss,
                                                                            mean_dev_acc))

                writer.add_summary(train_summ, i)
                writer.add_summary(dev_summ, i)
