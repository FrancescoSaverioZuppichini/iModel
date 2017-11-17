from core.layer import Layer

import tensorflow as tf

class Dense(Layer.Layer):

    def __init__(self, size, activation):
        self.shape = [size]
        self.activation = activation

    def build(self, x, n_input):

        self.shape = [n_input, self.shape[-1]]

        W = tf.Variable(tf.truncated_normal(self.shape, stddev=0.1), name='W')
        b = tf.Variable(tf.zeros([self.shape[-1]]), name='b')

        self.raw =  tf.matmul(x, W) + b
        self.output = self.activation(self.raw)

        return self.output

