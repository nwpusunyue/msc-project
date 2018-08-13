import numpy as np
import pandas as pd
import tensorflow as tf


def medhop_prediction(dataset, probs):
    dataset['prob'] = probs
    grouped = dataset.groupby(['id'], as_index=False).agg({'source': list, 'prob': list})
    grouped['prediction'] = grouped.apply(lambda row: row['source'][np.argmax(row['prob'])], axis=1)
    return grouped.loc[:, ['id', 'prediction']]


def generate_prediction(model, model_path, test,
                   test_batch_generator, eval_file_path=None):
    print('Generating predictions from model: {}'.format(model_path))

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list='0',
                                                      per_process_gpu_memory_fraction=1.0))

    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        model.load(sess, path=model_path)

        test_prob = np.array([])

        for j in range(test_batch_generator.batch_count):
            batch = test_batch_generator.get_batch()
            test_prob = np.concatenate((test_prob, model.predict_step(batch, sess)))

        pred = medhop_prediction(test, test_prob)
        pred.to_json(eval_file_path, orient='records')
