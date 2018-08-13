import numpy as np
import tensorflow as tf


def medhop_accuracy(dataset, probs, pred_file_path=None):
    dataset['prob'] = probs
    grouped = dataset.groupby(['id']).agg({'label': list, 'prob': list})
    grouped['correct'] = grouped.apply(lambda row: int(np.argmax(row['label']) == np.argmax(row['prob'])), axis=1)
    accuracy = grouped['correct'].sum() / len(grouped)
    if pred_file_path is not None:
        grouped.to_json(pred_file_path)
    return accuracy


def evaluate_model(model, model_path, train, train_eval_batch_generator, dev, dev_batch_generator, test,
                   test_batch_generator, word_embd=None, eval_file_path=None, train_pred_file_path=None,
                   dev_pred_file_path=None, test_pred_file_path=None):
    print('Evaluating model: {}'.format(model_path))

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list='0',
                                                      per_process_gpu_memory_fraction=0.5))

    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        model.load(sess, path=model_path)

        if word_embd is not None:
            word_embd_vals = sess.run(word_embd)
            np.save('projected_word2vec', word_embd_vals)

        train_eval_prob = np.array([])

        for j in range(train_eval_batch_generator.batch_count):
            batch = train_eval_batch_generator.get_batch()
            model.eval_step(batch, sess)
            train_eval_prob = np.concatenate((train_eval_prob, model.predict_step(batch, sess)))

        train_eval_metrics = model.eval_step(batch=None, sess=sess, reset=True)
        train_eval_medhop_acc = medhop_accuracy(train, train_eval_prob, train_pred_file_path)

        print('Train loss: {} Train medhop acc: {}'.format(train_eval_metrics, train_eval_medhop_acc))

        dev_prob = np.array([])

        for j in range(dev_batch_generator.batch_count):
            batch = dev_batch_generator.get_batch()
            model.eval_step(batch, sess)
            dev_prob = np.concatenate((dev_prob, model.predict_step(batch, sess)))

        dev_metrics = model.eval_step(batch=None, sess=sess, reset=True)

        dev_medhop_acc = medhop_accuracy(dev, dev_prob, dev_pred_file_path)

        print('Dev loss: {} Dev medhop acc: {}'.format(dev_metrics, dev_medhop_acc))

        test_prob = np.array([])

        for j in range(test_batch_generator.batch_count):
            batch = test_batch_generator.get_batch()
            model.eval_step(batch, sess)
            test_prob = np.concatenate((test_prob, model.predict_step(batch, sess)))

        test_metrics = model.eval_step(batch=None, sess=sess, reset=True)

        test_medhop_acc = medhop_accuracy(test, test_prob, test_pred_file_path)

        print('Test loss: {} Test medhop acc: {}'.format(test_metrics, test_medhop_acc))

        if eval_file_path is not None:
            file = open(eval_file_path, 'a')
            file.write('Evaluating model: {}\n'.format(model_path))
            file.write('Train loss: {} Train medhop acc: {}\n'.format(train_eval_metrics, train_eval_medhop_acc))
            file.write('Dev loss: {} Dev medhop acc: {}\n'.format(dev_metrics, dev_medhop_acc))
            file.write('Test loss: {} Test medhop acc: {}\n\n'.format(test_metrics, test_medhop_acc))


def encode_word_embd(word_embd, encoded_word_embd_path, model, model_path, config):
    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        model.load(sess, path=model_path)

        if word_embd is not None:
            word_embd_vals = sess.run(word_embd)
            np.save(encoded_word_embd_path, word_embd_vals)
