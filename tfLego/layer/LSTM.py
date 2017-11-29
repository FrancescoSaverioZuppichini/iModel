import tensorflow as tf
from tensorflow.contrib import rnn

from tfLego.layer.Layer import Layer


class LSTM(Layer):
    """
    LSTM layer implementation. Due to Tensorflow API, it represents a "block" of layers.
    """
    def __init__(self, size, cell=rnn.LSTMCell, dropout=None, name=None, *args, **kwargs):
        """
        :param size: An array composed: [n_units, n_units]
        :param name: A string that identifies the layer
        :param cell: The cell instance the layer should use
        :param dropout: The amount of dropout to apply
        """
        super().__init__(name)
        self.shape = size
        self.cell = cell
        self.dropout = dropout

    def get_states(self):
        return self.initial_state, self.state

    def build(self, x, n_input, prev_layer, model, *args, **kwargs):

        self.cells = [self.cell(size, **kwargs) for size in self.shape]

        if(self.dropout != None):
            self.cells = [rnn.DropoutWrapper(cell, input_keep_prob=self.dropout, output_keep_prob=self.dropout) for cell in self.cells]

        self.cells = rnn.MultiRNNCell(self.cells, state_is_tuple=True)

        self.initial_state = self.cells.zero_state(tf.shape(x)[0], dtype=tf.float32)

        self.output, self.state = tf.nn.dynamic_rnn(self.cells, x, initial_state=self.initial_state ,dtype=tf.float32)

        return { 'activated': self.output, 'raw': self.output }
