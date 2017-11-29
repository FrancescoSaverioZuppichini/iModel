import tensorflow as tf
from tfLego.logger.BasicLogger import BasicLogger
from tfLego.protocols import Buildable, Runnable, Trainable


class Model(Buildable, Runnable, Trainable):
    """
    A Model is a high-level abstraction of a machine learning model. It takes inputs provided by
    the client and it return an output. Also, it stores information about each layer as well as provide
    a fast way to access them.
    """

    def __init__(self, loss, optimizer=tf.train.AdamOptimizer(), cost="activated"):
        """
        Constructor
        :param loss: A loss function, eg. tf.losses.mean_squared_error
        :param optimizer: An optimizer, default: tf.train.AdamOptimizer
        :param cost: A string that identifies witch output should be feed into the loss function.
        Available options are: 'raw' (=not activated output), 'output' (=activated output).
        :param learning_rage: The learning rate
        """
        self.layers = [] # raw arrays containing each layer
        self.names = {} # if a layer is named, then its name is used as a key
        self.loss = loss
        self.cost = cost
        self.optimizer = optimizer
        self.logger = BasicLogger()

    @property
    def output(self):
        return self.outputs['activated']

    def add_layer(self, layer):
        """
        Ad a layer to the model.
        :param layer: A istance that subclasses Layer
        :return:
        """
        has_name = layer.name != None
        # if name is not provided, its index is instead
        if( not has_name):
            layer.name = len(self.layers)

        self.names[layer.name] = layer

        self.layers.append(layer)

    def add_layers(self, layers):
        """
        Dynamic adds more than one layer
        :param layers: An array of instance that subclasses Layer
        :return:
        """
        for layer in layers:
            self.add_layer(layer)

    def create_train_step(self):

        return self.optimizer.minimize(self.loss)

    def build_layers(self, x):
        """
        Iteratively creates one layer after the other
        :param x: The inputs
        :return: The last layer outputs
        """
        outputs = { 'activated': x }
        i = 0

        prev_layer = None

        for layer in self.layers:
            # the next size must be specified layer-side
            # in order to give more control about the inner structure
            if(i > 0):
                prev_layer = self.layers[i - 1]

            next_size = outputs['activated'].get_shape().as_list()[-1]

            outputs = layer.build(outputs['activated'], next_size, model=self, prev_layer=prev_layer)

            i += 1

        return outputs

    def build(self, x, y, *args, **kwargs):
        """
        Create the whole computation graph from the model
        :param x: The inputs
        :param y: The targets
        :return: the train step and the predictions
        """
        # store ref to inputs and targets
        self.x, self.y = x, y

        self.outputs = self.build_layers(x)

        self.loss = self.loss(self.outputs[self.cost], y)

        self.train_step = self.create_train_step()

        # return them for convenience
        return self.train_step, self.outputs, self.loss

    def run(self, sess, feed_dict):

        outputs = sess.run([ tf.reduce_mean(self.loss),
                                  self.train_step,
                                  self.output ], feed_dict=feed_dict)

        return outputs

    @classmethod
    def from_args(cls, args):
        pass


    def get_accuracy(self, outputs, targets = None):
        """
        Compute the computation node to get accuracy. If
        targets is not specified self.y will be used instead
        :param outputs: The output of the network
        :param targets: The target
        :return: The accuracy computation node
        """
        if(targets == None):
            targets = self.y

        prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1))

        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

        return accuracy

    def will_start_epoch(self):
        pass

    def did_finish_epoch(self):
        pass

    def will_start_batch(self, sess, x_batch, y_batch):
        pass

    def did_finish_batch(self, sess, loss, outputs, accuracy, feed_dict):
        pass

    def train(self, sess, train_set, epochs, val_set=None, *args, **kwargs):

        X, Y = train_set['X'], train_set['Y']
        X_val, Y_val = [], []

        if (val_set):
            X_val, Y_val = val_set['X'], val_set['Y']

        for i in range(epochs):

            self.will_start_epoch()

            for x_batch, y_batch in zip(X,Y):

                self.will_start_batch(sess, x_batch, y_batch)

                feed_dict = { 'x:0' : x_batch, 'y:0' : y_batch }

                loss, _, outputs = self.run(sess, feed_dict=feed_dict)

                accuracy = sess.run(self.get_accuracy(outputs, y_batch), feed_dict=feed_dict)

                self.did_finish_batch(sess, loss, outputs, accuracy, feed_dict)

                self.logger.log_batch(loss, outputs, accuracy)

            self.did_finish_epoch()

            self.logger.log_epoch(i, X)

            if(val_set):

                for x_batch, y_batch in zip(X_val, Y_val):

                    feed_dict = { 'x:0' : x_batch, 'y:0': y_batch}

                    loss, outputs = sess.run([tf.reduce_mean(self.loss) , self.output], feed_dict=feed_dict)

                    accuracy = sess.run(self.get_accuracy(outputs, y_batch), feed_dict=feed_dict)

                    self.logger.log_batch(loss, outputs, accuracy)

                self.logger.log_epoch(i, X_val, is_val=True)



    def __str__(self):
        # TODO: improve it!
        model_str = "x -> "
        model_str += str(self.layers[0].shape[0])

        for layer in self.layers:
            model_str += " -> " +  str(layer.shape)

        model_str += " -> y"

        return model_str