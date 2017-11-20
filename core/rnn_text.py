import tensorflow as tf
import numpy as np

from core.layer.Dense import Dense
from core.layer.LSTM import LSTM
from core.layer.Modifier import Modifier


from core.model import NeuralNetwork
from core.model.Model import Model
from core.reader import Reader

DATA_URL = "./shakespeare_little"

batch_size = 64
sequence_len = 100

my_reader = Reader.Reader(batch_size=batch_size, sequence_len=sequence_len)
my_reader.load(DATA_URL)
iter = my_reader.create_iter(nb_epochs=5)

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


dropout = tf.placeholder(tf.float32, name='dropout')


def flat_output(x, n_input, prev_layer):

    outputs = { 'output': tf.reshape(x, [-1, prev_layer.shape[-1]]),
                'next_size': prev_layer.shape[-1] }

    return outputs

rnn  = Model(learning_rage=0.01, loss=get_loss, cost="raw")
rnn.add_layer(LSTM(size=[128], dropout=dropout))
# the output of the LSTM must be reshaped from 3D to 2D
rnn.add_layer(Modifier(handler=flat_output))
rnn.add_layer(Dense(size=n_classes, activation=tf.nn.softmax))
rnn.build(x,y)

# LSTM is automatically named 'lstm'
state = rnn.names['lstm'].state
initial_state = rnn.names['lstm'].initial_state
print(rnn)

def sample_prob_picker_from_best(distribution, n=2):
    p = np.squeeze(distribution)
    p[np.argsort(p)[:-n]] = 0
    p = p / np.sum(p)

    return np.array([np.random.choice(distribution.shape[-1], 1, p=p)[0]])

def generate_text(input_val, n_text=100):
    text = input_val

    x_batch = np.array([[ord(c) for c in input_val]])

    last_state = None

    for i in range(n_text):
        feed_dict = {'x:0': x_batch, dropout: 1.0}

        if (last_state != None):
            feed_dict[initial_state] = last_state

        preds, last_state = sess.run([rnn.outputs['output'], state], feed_dict=feed_dict)
        preds = np.array([preds[-1]])

        next = sample_prob_picker_from_best(preds)

        print(chr(next[0]), end='')
        # next is 1D vector like [14]
        text += "".join(chr(next[0]))

        x_batch = np.array([next])



last_state = None

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for x_batch, y_batch, epoch in iter:

        feed_dict = { 'x:0' : x_batch, 'y:0' : y_batch, dropout : 0.8 }

        loss, _ = rnn.run(sess, feed_dict=feed_dict)
        # loss_val = sess.run(rnn.loss, feed_dict={'x:0': my_reader.val_data['X'], 'y:0':  my_reader.val_data['Y']})

        last_state = sess.run(state, feed_dict=feed_dict)

        feed_dict[initial_state] = last_state

        print(sess.run(tf.reduce_mean(loss)),epoch)

    generate_text('T')