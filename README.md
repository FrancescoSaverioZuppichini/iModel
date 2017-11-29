# tfLego
### Low Level Expressive API for tensorflow
by Francesco Saverio Zuppichini

Similar to [lego](https://www.lego.com/en-us/), this framework aims to produce a easy and fast way to build models while giving the ability to the client to still deeply customise its implementation.
### Quick Start

TODO

docs here (TODO)

### Introduction

Nowadays lots of high level frameworks for TensorFlow exist, like Keras or tfLean. They provide a fast and scalable way to create machine learning model and train them, but, on the other hand, the client can not use most of the TensorFlow APIs and have to trust the custom made implementation from the framework provider that, in most of the cases, reinvende the wheels. With tfLego, only TensorFLows API are used.

Most of the time, the library explicit ask the client to provide an instance from TensorFlow. Let's see how to create a model:

```
net  = Model(optimizer=tf.train.AdamOptimizer(0.05), loss=tf.losses.mean_squared_error)

``` 
tfLego asks for an `optimizer` and a `loss` function that must be from TensorFlow, nothing more nothing less.

CONTINUE -> TODO
### Example

#### Feedforward Network

```[python]
# get some data
X,T = twospirals()
# create training and validation set
X_val, T_val = X[200:], T[200:]
X, T = X[:200], T[:200]
# create inputs plaholder, dimension MUST be specified
x = tf.placeholder(tf.float32, [None,2], name='x')
y = tf.placeholder(tf.float32, [None,1], name='y')
drop_prob = tf.placeholder(tf.float32, name='drop')

# create the network
net  = Model(optimizer=tf.train.AdamOptimizer(0.05), loss=tf.losses.mean_squared_error)
net.add_layer(Dense(size=20, activation=tf.nn.relu, dropout=drop_prob))
net.add_layer(Dense(size=20, activation=tf.nn.relu, dropout=drop_prob))
net.add_layer(Dense(size=1, activation=tf.nn.tanh, dropout=drop_prob))
output = net.build(x,y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(200):
    
        loss, _  = net.run(sess, feed_dict={'x:0': X, 'y:0':T, 'drop:0': 0.8})
        loss_val = sess.run(net.loss, feed_dict={'x:0': X_val, 'y:0': T_val, 'drop:0':1.0})

        print(loss, loss_val)
```
TODO:

- LSTM
- CNN
- RNN 
