import tensorflow as tf
import numpy as np

from core.layer import Dense
from core.layer import LSTM

from core.model import NeuralNetwork
from core.model import Model
from core.reader import Reader

DATA_URL = "./shakespeare_little"

my_reader = Reader.Reader()
my_reader.load(DATA_URL)
iter = my_reader.create_iter(nb_epochs=30)

n_classes = 255

x = tf.placeholder(tf.int64, [None, None], name='x')
y = tf.placeholder(tf.int64, [None, None], name='y')
# each input/target must be a 1-hot vector
x = tf.one_hot(x, depth=n_classes, name='X_hot')
y = tf.one_hot(y, depth=n_classes, name='Y_hot')


def get_loss(logits, y):
    shape = y.get_shape().as_list()

    y_flat = tf.reshape(y, [-1, shape[-1]])

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_flat)

    return loss


rnn  = Model.Model(learning_rage=0.01, loss=get_loss, cost="raw")
rnn.add_layer(LSTM.LSTM(size=[128]))
rnn.add_layer(Dense.Dense(size=n_classes, activation=tf.nn.softmax))
rnn.build(x,y)
#
# print(net)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for x_batch, y_batch, epoch in iter:

        loss, _ = rnn.run(sess, feed_dict={'x:0': x_batch, 'y:0':y_batch})
        print(sess.run(tf.reduce_mean(loss)))

