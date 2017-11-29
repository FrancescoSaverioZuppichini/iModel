import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tfLego.layer.Dense import Dense
from tfLego.model.Model  import Model

def two_spirals(n_points=120, noise=1.6, twist=420):
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

def plot_data(X,T):
    """
    Plots the 2D data as a scatterplot
    """
    plt.scatter(X[:,0], X[:,1], s=40, c=T, cmap=plt.cm.Spectral)


def plot_boundary(X, targets, model, sess, threshold=0.0):
    """
    Plots the data and the boundary lane which seperates the input space into two classes.
    """
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    y = sess.run(model.output, feed_dict={'x:0': X_grid})
    plt.contourf(xx, yy, y.reshape(*xx.shape) < threshold, alpha=0.5)
    plot_data(X, targets)
    plt.ylim([y_min, y_max])
    plt.xlim([x_min, x_max])


X,T = two_spirals(200)

c = list(zip(X, T))
random.shuffle(c)

X, T = zip(*c)
X,T = np.array(X), np.array(T)
off_set = (len(X) // 100) * 80

X_val, T_val = X[off_set:], T[off_set:]
X, T = X[:off_set], T[:off_set]

x = tf.placeholder(tf.float32, [None,2], name='x')
y = tf.placeholder(tf.float32, [None,1], name='y')

drop_prob = tf.placeholder(tf.float32, name='drop')

net  = Model(optimizer=tf.train.AdamOptimizer(0.05), loss=tf.losses.mean_squared_error)
net.add_layer(Dense(size=20, activation=tf.nn.relu, dropout=drop_prob))
net.add_layer(Dense(size=20, activation=tf.nn.relu, dropout=drop_prob))
net.add_layer(Dense(size=1, activation=tf.nn.tanh, dropout=drop_prob))
output = net.build(x,y)

plt.ion()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):

        loss, _, output  = net.run(sess, feed_dict={'x:0': X, 'y:0':T, 'drop:0': 0.8})

        loss_val = sess.run(net.loss, feed_dict={'x:0': X_val, 'y:0': T_val, 'drop:0':1.0})

        print(loss, loss_val)

        if(i % 10 == 0):
            plot_boundary(X,T, net, sess, threshold=0.5)
            plt.title("iter {0}, loss {1:.4f}".format(i,loss))
            plt.pause(0.01)

