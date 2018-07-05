import tensorflow as tf

from abc import ABC, abstractmethod
from functools import reduce


class BaseModel(ABC):

    def __init__(self):
        # will be set in setup
        self._tensors = None
        self._placeholders = None

    @abstractmethod
    def _setup_model(self, model_params):
        pass

    @abstractmethod
    def _setup_evaluation(self):
        pass

    @property
    @abstractmethod
    def params_str(self):
        pass

    @abstractmethod
    def train_step(self, batch, sess):
        pass

    def _setup_training(self, loss, optimizer=tf.train.AdamOptimizer, l2=0.0, clip_op=None, clip=None):
        global_step = tf.train.get_global_step()
        if global_step is None:
            global_step = tf.train.create_global_step()

        if l2:
            loss += tf.add_n([tf.nn.l2_loss(v) for v in self.train_variables]) * l2

        gradients = optimizer.compute_gradients(loss=loss)
        if clip:
            if clip_op == tf.clip_by_value:
                gradients = [(tf.clip_by_value(grad, clip[0], clip[1]), var)
                             for grad, var in gradients if grad is not None]
            elif clip_op == tf.clip_by_norm:
                gradients = [(tf.clip_by_norm(grad, clip), var)
                             for grad, var in gradients if grad is not None]

        tf.summary.scalar('gradients_l2', tf.add_n([tf.nn.l2_loss(grad[0]) for grad in gradients]),
                          collections=['summary_train'])
        train_op = optimizer.apply_gradients(gradients, global_step)

        variable_size = lambda v: reduce(lambda x, y: x * y, v.get_shape().as_list()) if v.get_shape() else 1
        num_params = sum(variable_size(v) for v in self.train_variables)
        print("Number of parameters: {}".format(num_params))

        self._tensors['loss'] = loss
        self._tensors['train_op'] = train_op
        self._tensors['global_step'] = global_step
        return loss, train_op

    def _setup_summaries(self):
        self._tensors['summary_train'] = tf.summary.merge_all('summary_train')
        self._tensors['summary_test'] = tf.summary.merge_all('summary_test')
        self._tensors['summary_train_eval'] = tf.summary.merge_all('summary_train_eval')

    @property
    def placeholders(self):
        '''

        :return: [placeholder_name, placeholder]
        '''
        if hasattr(self, "_placeholders"):
            return self._placeholders
        else:
            print("Asking for placeholders without having setup this module. Returning None.")
            return None

    @property
    def tensors(self):
        '''

        :return: [tensor_name, tensor]
        '''
        if hasattr(self, "_tensors"):
            return self._tensors
        else:
            print("Asking for tensors without having setup this module. Returning None.")
            return None

    @property
    def train_variables(self):
        return self._training_variables

    @property
    def variables(self):
        return self._variables

    def store(self, sess, path):
        self._saver.save(sess, path)

    def load(self, sess, path):
        self._saver.restore(sess, path)

    def convert_to_feed_dict(self, mapping):
        result = {ph: mapping[ph_name] for ph_name, ph in self.placeholders.items() if ph_name in mapping}
        return result
