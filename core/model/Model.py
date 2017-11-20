from core.protocols import  Buildable, Runnable

import tensorflow as tf

class Model(Buildable, Runnable):

    def __init__(self, loss, optimizer=tf.train.AdamOptimizer, cost="output",learning_rage=0.001):
        self.layers = []
        self.names = {}
        self.loss = loss
        self.cost = cost
        self.optimizer = optimizer
        self.learning_rage = learning_rage

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

        return self.optimizer(self.learning_rage).minimize(self.loss)

    def build_layers(self, x):
        """
        Iteratively creates one layer after the other
        :param x: The inputs
        :return: The last layer outputs
        """
        outputs = { "output" : x, "next_size": x.get_shape().as_list()[-1] }
        i = 0
        prev_layer = None
        for layer in self.layers:
            # the next size must be specified layer-side
            # in order to give more control about the inner structure
            if(i > 0):
                prev_layer = self.layers[i - 1]
            outputs = layer.build(outputs["output"], outputs["next_size"], prev_layer=prev_layer)
            i += 1

        return outputs

    def build(self, x, y):
        """
        Create the whole computation graph from the model
        :param x: The inputs
        :param y: The targets
        :return: the train step and the predictions
        """
        self.outputs = self.build_layers(x)

        self.loss = self.loss(self.outputs[self.cost], y)

        self.train_step = self.create_train_step()

        # return them for convenience
        return self.train_step, self.outputs['output'], self.loss

    def run(self, sess, feed_dict):

        current_loss = sess.run([ self.loss, self.train_step ], feed_dict=feed_dict)

        return current_loss

    @classmethod
    def from_args(cls):
        pass

    def __str__(self):
        model_str = "x -> "
        model_str += str(self.layers[0].shape[0])

        for layer in self.layers:
            model_str += " -> " +  str(layer.shape)

        model_str += " -> y"

        return model_str