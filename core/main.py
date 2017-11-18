import tensorflow as tf
import numpy as np

from core.layer import Dense
from core.layer import LSTM

from core.model import NeuralNetwork

def twospirals(n_points=120, noise=1.6, twist=420):
    """
     Returns a two spirals dataset.
    """
    np.random.seed(0)
    n = np.sqrt(np.random.rand(n_points,1)) * twist * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    X, T =(np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),
            np.hstack((np.zeros(n_points),np.ones(n_points))))
    T = np.reshape(T, (T.shape[0],1))
    return X, T

X,T = twospirals()

X_val, T_val = X[200:], T[200:]
X, T = X[:200], T[:200]

x = tf.placeholder(tf.float32, [None,2], name='x')
y = tf.placeholder(tf.float32, [None,1], name='y')

net  = NeuralNetwork.NeuralNetwork(learning_rage=0.01)
net.add_layer(Dense.Dense(size=20, activation=tf.nn.relu))
net.add_layer(Dense.Dense(size=10, activation=tf.nn.relu))
net.add_layer(Dense.Dense(size=1, activation=tf.nn.tanh))
output = net.build(x,y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(300):
        loss, _ = net.run(sess, feed_dict={'x:0': X, 'y:0':T})
        loss_val = sess.run([net.loss], feed_dict={'x:0': X_val, 'y:0': T_val})
        print(loss, loss_val[0])

