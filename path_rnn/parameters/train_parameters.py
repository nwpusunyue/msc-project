import tensorflow as tf


class Optimizer:
    ADAM = 'adam'
    SGD = 'sgd'

    optimizers = {
        ADAM: tf.train.AdamOptimizer,
        SGD: tf.train.GradientDescentOptimizer
    }

    @staticmethod
    def get_optimizer(name):
        return Optimizer.optimizers[name]


class TrainParameters:

    def __init__(self,
                 optimizer=Optimizer.ADAM,
                 learning_rate=1e-3,
                 epochs=100,
                 batch_size=32,
                 test_prop=0.1,
                 train_eval_prop=1.0,
                 check_period=10,
                 debug_mode=False,
                 word_embd_path='medhop_word2vec_punkt',
                 train_eval_batch_size=None):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_eval_prop = train_eval_prop
        self.test_prop = test_prop
        self.check_period = check_period
        self.debug_mode = debug_mode
        self.word_embd_path = word_embd_path
        self.train_eval_batch_size = train_eval_batch_size

    def print(self):
        print('\n'.join("%s: %s" % item for item in vars(self).items()))