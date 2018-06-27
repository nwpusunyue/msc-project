import logging

import tensorflow as tf

from abc import ABC, abstractmethod
from functools import reduce

logger = logging.getLogger(__name__)


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

    @abstractmethod
    def print_params(self, train_params):
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
        if clip:
            gradients = optimizer.compute_gradients(loss=loss)
            if clip_op == tf.clip_by_value:
                gradients = [(tf.clip_by_value(grad, clip[0], clip[1]), var)
                             for grad, var in gradients if grad is not None]
            elif clip_op == tf.clip_by_norm:
                gradients = [(tf.clip_by_norm(grad, clip), var)
                             for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(gradients, global_step)
        else:
            train_op = optimizer.minimize(loss, global_step)

        variable_size = lambda v: reduce(lambda x, y: x * y, v.get_shape().as_list()) if v.get_shape() else 1
        num_params = sum(variable_size(v) for v in self.train_variables)
        logger.info("Number of parameters: {}".format(num_params))

        self._tensors['loss'] = loss
        self._tensors['train_op'] = train_op
        return loss, train_op

    @property
    def placeholders(self):
        '''

        :return: [placeholder_name, placeholder]
        '''
        if hasattr(self, "_placeholders"):
            return self._placeholders
        else:
            logger.warning("Asking for placeholders without having setup this module. Returning None.")
            return None

    @property
    def tensors(self):
        '''

        :return: [tensor_name, tensor]
        '''
        if hasattr(self, "_tensors"):
            return self._tensors
        else:
            logger.warning("Asking for tensors without having setup this module. Returning None.")
            return None

    @property
    def train_variables(self):
        return self._training_variables

    @property
    def variables(self):
        return self._variables

    def store(self, path):
        self._saver.save(self.tf_session, path)

    def load(self, path):
        self._saver.restore(self.tf_session, path)

    def convert_to_feed_dict(self, mapping):
        result = {ph: mapping[ph_name] for ph_name, ph in self.placeholders.items() if ph_name in mapping}
        return result
