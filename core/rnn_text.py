import tensorflow as tf
import numpy as np

from core.layer import Dense
from core.layer import LSTM

from core.model import NeuralNetwork
from core.model import Model
from core.reader import Reader

DATA_URL = "./shakespeare_little"

batch_size = 64
sequence_len = 100

my_reader = Reader.Reader(batch_size=batch_size, sequence_len=sequence_len)
my_reader.load(DATA_URL)
iter = my_reader.create_iter(nb_epochs=1)

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


LSTM_block  = LSTM.LSTM(size=[128])

rnn  = Model.Model(learning_rage=0.01, loss=get_loss, cost="raw")
rnn.add_layer(LSTM_block)
rnn.add_layer(Dense.Dense(size=n_classes, activation=tf.nn.softmax))
rnn.build(x,y)
#
# print(net)

# lstm =  rnn.names['lstm']

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
        feed_dict = {'X:0': x_batch, 'pkeep:0': 1.0}

        if (last_state != None):
            feed_dict[LSTM_block.initial_state] = last_state

        preds, last_state = sess.run([rnn.outputs['output'], LSTM_block.state], feed_dict=feed_dict)
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

        feed_dict = {'x:0': x_batch, 'y:0':y_batch}

        loss, _ = rnn.run(sess, feed_dict=feed_dict)
        loss_val = sess.run(rnn.loss, feed_dict={'x:0': my_reader.val_data['X'], 'y:0':  my_reader.val_data['Y']})

        last_state = sess.run(LSTM_block.state, feed_dict=feed_dict)

        feed_dict[LSTM_block.initial_state] = last_state

        print(sess.run(tf.reduce_mean(loss)),sess.run(tf.reduce_mean(loss_val)),epoch)

    generate_text('T')