import os
import time

import numpy as np
import tensorflow as tf

from tqdm import tqdm


def medhop_accuracy(dataset, probs):
    dataset['prob'] = probs
    grouped = dataset.groupby(['id']).agg({'label': list, 'prob': list})
    grouped['correct'] = grouped.apply(lambda row: int(np.argmax(row['label']) == np.argmax(row['prob'])), axis=1)
    accuracy = grouped['correct'].sum() / len(grouped)
    return accuracy


def train_model(model, train, train_batch_generator, train_eval_batch_generator, dev, dev_batch_generator,
                num_epochs, run_id_params, model_name, check_period=20, config=None):
    steps = train_batch_generator.batch_count * num_epochs

    medhop_acc = []
    max_dev_medhop_acc = 0.0

    start_time = time.strftime('%X_%d.%m.%y')
    run_id = 'run_{}_{}'.format(start_time, run_id_params)
    print('Run id: {}'.format(run_id))
    model_dir = './textual_chains_of_reasoning_models/{}/{}'.format(model_name, run_id)
    log_dir = './textual_chains_of_reasoning_logs/{}/{}'.format(model_name, run_id)
    acc_dir = './textual_chains_of_reasoning_logs/{}/acc_{}.txt'.format(model_name, run_id)

    # make save dir
    os.makedirs(model_dir)
    # make summary writer
    summ_writer = tf.summary.FileWriter(log_dir)
    summ_writer.add_graph(tf.get_default_graph())
    # make acc file
    acc_file = open(acc_dir, 'w+')

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for i in tqdm(range(steps)):
            batch = train_batch_generator.get_batch()
            model.train_step(batch, sess, summ_writer=summ_writer)

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
