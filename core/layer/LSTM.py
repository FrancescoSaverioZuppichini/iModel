from core.layer.Layer import Layer
import tensorflow as tf
from tensorflow.contrib import rnn


class LSTM(Layer):
    """
    LSTM layer implementation. Due to Tensorflow API, it represent a "block" of layers.
    """
    def __init__(self, size, name="lstm", cell=rnn.LSTMCell, dropout=None):
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

    def build(self, x, n_input, prev_layer, *args, **kwargs):

        cells = [self.cell(size) for size in self.shape]

        if(self.dropout != None):
            cells = [rnn.DropoutWrapper(cell, **kwargs) for cell in cells]

        cells = rnn.MultiRNNCell(cells, state_is_tuple=True)

        self.initial_state = cells.zero_state(tf.shape(x)[0], dtype=tf.float32)

        self.raw_output, self.state = tf.nn.dynamic_rnn(cells, x, initial_state=self.initial_state ,dtype=tf.float32)

        # self.output = tf.reshape(self.raw_output, [-1, self.shape[-1]])

        outputs = {
            'raw': self.raw_output,
            'output': self.raw_output,
            'next_size': self.shape[-1]
        }

        return outputs
