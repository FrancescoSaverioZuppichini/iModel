import numpy as np
import tensorflow as tf

from core.layer.Dense import Dense
from core.layer.LSTM import LSTM
from core.layer.Modifier import Modifier
from core.model.Model import Model

SYNTAX = 'abc'
FULL_SYNTAX = 'S' + SYNTAX + 'T'

ch_to_index = {ch: i for i, ch in enumerate(list(FULL_SYNTAX))}
index_to_ch = {i: ch for i, ch in enumerate(FULL_SYNTAX)}

def generate_context_lang(n, syntax):

    symbols = 'S'

    for s in syntax:
        symbols += s * n

    symbols += 'T'

    symbols = np.array([ ord(s) for s in list(symbols) ])

    X = symbols[:-1]
    Y = symbols[1:]

    return X.reshape(1,-1), Y.reshape(1,-1)

X = []
Y = []

for i in range(1,50):

    x, y = generate_context_lang(i, SYNTAX)
    X.append(x)
    Y.append(y)

X_val, Y_val = generate_context_lang(51, SYNTAX)

n_classes = len(ch_to_index.keys())

x = tf.placeholder(tf.int64, [None, None], name='x')
y = tf.placeholder(tf.int64, [None, None], name='y')
# each input/target must be a 1-hot vector
x = tf.one_hot(x, depth=n_classes, name='X_hot')
y = tf.one_hot(y, depth=n_classes, name='Y_hot')
# print(data)

def get_loss(logits, y):

    shape = y.get_shape().as_list()

    y_flat = tf.reshape(y, [-1, shape[-1]])

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_flat)

    return loss

def flat_output(x, n_input, prev_layer):

    outputs = { 'output': tf.reshape(x, [-1, prev_layer.shape[-1]]),
                'next_size': prev_layer.shape[-1] }

    return outputs

def generate(sess, input_val='S',n_text=100):
    text = input_val

    x_batch = np.array([[ch_to_index[c] for c in input_val]])

    last_state = None

    for i in range(n_text):
        feed_dict = {'x:0': x_batch}

        if (last_state != None):
            feed_dict[initial_state] = last_state

        preds, last_state = sess.run([rnn.outputs['output'], state], feed_dict=feed_dict)

        next = sess.run(tf.argmax(preds, -1))

        # print(index_to_ch[next[0]], end='')
        # next is 1D vector like [14]
        text += "".join(index_to_ch[next[0]])

        x_batch = np.array([next])

    return text


rnn  = Model(learning_rage=0.01, loss=get_loss, cost="raw")
rnn.add_layer(LSTM(size=[1,1,1,1]))
# the output of the LSTM must be reshaped from 3D to 2D
rnn.add_layer(Modifier(handler=flat_output))
rnn.add_layer(Dense(size=n_classes, activation=tf.nn.softmax))
rnn.build(x,y)

state = rnn.names['lstm'].state
initial_state = rnn.names['lstm'].initial_state

EPOCHS = 10

last_state = None

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(EPOCHS):

        for x_batch, y_batch in zip(X,Y):

            feed_dict = {'x:0': x_batch, 'y:0': y_batch }

            loss, _ = rnn.run(sess, feed_dict=feed_dict)

            last_state = sess.run(state, feed_dict=feed_dict)

            feed_dict[initial_state] = last_state

            print(sess.run(tf.reduce_mean(loss)))

        last_state = None

        feed_dict = {'x:0': X_val, 'y:0': Y_val}
        loss, _ = rnn.run(sess, feed_dict=feed_dict)

        print("Val loss: {}".format(sess.run(tf.reduce_mean(loss))))
        print(generate(sess, 'S'))
        # train