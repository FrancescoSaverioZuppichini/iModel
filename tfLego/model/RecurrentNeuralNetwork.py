import tensorflow as tf
from tfLego.layer.LSTM import LSTM
from tfLego.layer.Dense import Dense
from tfLego.layer.Modifier import  Reshaper
from tfLego.model.Model import Model


class RecurrentNeuralNetwork(Model):

    def __init__(self, shape, n_classes, pass_state=True, *args, **kwargs):

        self.lstm_shape = shape
        self.last_state = None
        self.n_classes = n_classes
        self.pass_state = pass_state

        super().__init__(loss=self.softmax_cross_entropy, cost='raw', *args, **kwargs)

    def softmax_cross_entropy(self, logits, y):
        """
        Wrapper around tf.nn.softmax_cross_entropy_with_logits since logits
        and labels must be named
        """
        shape = y.get_shape().as_list()

        y_flat = tf.reshape(y, [-1, shape[-1]])

        return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_flat)

    def get_accuracy(self, outputs, targets=None):

        shape = self.x.get_shape().as_list()

        targets_flat= tf.reshape(self.y, [-1, shape[-1]])

        accuracy = super().get_accuracy(outputs, targets_flat)

        return accuracy

    def build(self, x, y, *args, **kwargs):

        self.add_layer(LSTM(size=self.lstm_shape, **kwargs))
        self.add_layer(Reshaper(shape=[-1, self.lstm_shape[-1]]))
        self.add_layer(Dense(size=self.n_classes))

        super().build(x, y)

    def state_feeder(self, sess, feed_dict):
        if(not self.pass_state):
            return

        initial_state, state = self.names['lstm'].get_states()

        if (self.last_state != None):
            feed_dict[initial_state] = self.last_state


    def reset_state(self):
        self.last_state = None

    def will_start_epoch(self):
        self.reset_state()

    def did_finish_batch(self, sess, loss, outputs, accuracy, feed_dict):
        self.state_feeder(sess, feed_dict)
