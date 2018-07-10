import numpy as np
import tensorflow as tf


def medhop_accuracy(dataset, probs):
    dataset['prob'] = probs
    grouped = dataset.groupby(['id']).agg({'label': list, 'prob': list})
    grouped['correct'] = grouped.apply(lambda row: int(np.argmax(row['label']) == np.argmax(row['prob'])), axis=1)
    accuracy = grouped['correct'].sum() / len(grouped)
    return accuracy


def evaluate_model(model, model_path, train, train_eval_batch_generator, dev, dev_batch_generator, test,
                   test_batch_generator):
    print('Evaluating model: {}'.format(model_path))
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list='0',
                                                      per_process_gpu_memory_fraction=0.5))

    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
        model.load(sess, path=model_path)

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
