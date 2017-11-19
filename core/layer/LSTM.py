from core.layer import Layer
import tensorflow as tf
from tensorflow.contrib import rnn


class LSTM(Layer.Layer):

    def __init__(self, size, name="lstm", cell=rnn.LSTMCell):
        super().__init__(name)
        self.size = size
        self.cell = cell

    def build(self, x, n_input, *args, **kwargs):

        cells = [self.cell(size) for size in self.size]
        cells = [rnn.DropoutWrapper(cell, **kwargs) for cell in cells]
        cells = rnn.MultiRNNCell(cells, state_is_tuple=True)

        self.initial_state = cells.zero_state(tf.shape(x)[0], dtype=tf.float32)

        raw_output, self.state = tf.nn.dynamic_rnn(cells, x, initial_state=self.initial_state ,dtype=tf.float32)

        output = tf.reshape(raw_output, [-1, self.size[-1]])

        outputs = {
            'raw': raw_output,
            'output': output
        }

        return outputs, self.size[-1]


def test2(dio):
    print(dio)


def test(**kwargs):
    test2(**kwargs)

test(dio='test')