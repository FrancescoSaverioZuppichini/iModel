from core.model import Model
import tensorflow as tf


class NeuralNetwork(Model.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(loss=tf.losses.mean_squared_error, *args, **kwargs)
        # self.loss = tf.losses.mean_squared_error

