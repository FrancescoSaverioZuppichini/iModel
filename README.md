# iModel
### Low Level expressive API for tensorflow
by Francesco Saverio Zuppichini

### Quick Start

TODO

docs here (TODO)

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
