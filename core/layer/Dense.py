from core.layer import Layer

import tensorflow as tf

class Dense(Layer.Layer):

    def __init__(self, size, activation, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.shape = [size]
        self.activation = activation

    def build(self, x, n_input, cost="output"):

        self.shape = [n_input, self.shape[-1]]

        W = tf.Variable(tf.truncated_normal(self.shape, stddev=0.1), name='{}-W'.format(self.name))
        b = tf.Variable(tf.zeros([self.shape[-1]]), name='{}-b'.format(self.name))

        self.raw_output =  tf.matmul(x, W) + b
        self.output = self.activation(self.raw)

        outputs = {
            'raw': self.raw_output,
            'output': self.output,
            'next_size': self.shape[-1]
        }

        return outputs

