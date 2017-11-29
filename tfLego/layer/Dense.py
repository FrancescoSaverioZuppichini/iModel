import tensorflow as tf

from tfLego.layer.Layer import Layer


class Dense(Layer):

    def __init__(self, size, activation=None, dropout=None, *args, **kwargs):
        super().__init__( *args, **kwargs)

        self.shape = [size]
        self.activation = activation
        self.dropout = dropout

    def build(self, x, n_input, prev_layer, model, initializer=None, *args, **kwargs):

        self.shape = [n_input, self.shape[-1]]

        if(initializer != None):
            W, b  = initializer(x, n_input, prev_layer)
        else:
            W = tf.Variable(tf.truncated_normal(self.shape, stddev=0.1), name='{}-W'.format(self.name))
            b = tf.Variable(tf.zeros([self.shape[-1]]), name='{}-b'.format(self.name))

        self.raw_output =  tf.matmul(x, W) + b
        self.output = self.raw_output

        if(self.dropout != None):
            self.output = tf.layers.dropout(self.dropout)

        if(self.activation):
            self.output = self.activation(self.raw_output)

        return { 'activated': self.output, 'raw': self.raw_output }

