from core.model import Model
import tensorflow as tf

class RecurrentNeuralNetwork(Model.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(loss=tf.nn.softmax_cross_entropy_with_logits, *args, **kwargs)
        # self.loss = tf.losses.mean_squared_error
