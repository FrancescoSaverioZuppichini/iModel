from core.layer import Layer

import tensorflow as tf

class Dense(Layer.Layer):

    def __init__(self, size, activation):
        self.shape = [size]
        self.activation = activation

    def build(self, x, n_input, cost="output"):

        self.shape = [n_input, self.shape[-1]]

        W = tf.Variable(tf.truncated_normal(self.shape, stddev=0.1), name='W')
        b = tf.Variable(tf.zeros([self.shape[-1]]), name='b')

        self.raw =  tf.matmul(x, W) + b
        self.output = self.activation(self.raw)


        n_next_input = self.shape[-1]

        outputs = {
            'raw': self.raw,
            'output': self.output
        }

        return outputs, n_next_input

