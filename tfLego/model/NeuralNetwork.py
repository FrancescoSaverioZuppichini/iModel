import tensorflow as tf

from tfLego.model.Model import Model


class NeuralNetwork(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(loss=tf.losses.mean_squared_error, *args, **kwargs)


