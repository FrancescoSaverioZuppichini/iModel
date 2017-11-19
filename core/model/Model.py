from core.protocols import  Buildable, Runnable

import tensorflow as tf

class Model(Buildable, Runnable):

    def __init__(self, loss, optimizer=tf.train.AdamOptimizer, cost="output",learning_rage=0.01):
        self.layers = []
        self.names = {}
        self.loss = loss
        self.cost = cost
        self.optimizer = optimizer
        self.learning_rage = learning_rage

    def add_layer(self, layer, activation=tf.nn.relu):
        has_name = layer.name != None

        if(has_name):
            self.names[layer.name] = layer

        self.layers.append(layer)

    def add_layers(self, layers):

        for layer in layers:
            self.add_layer(layer)

    def create_train_step(self):

        return self.optimizer(self.learning_rage).minimize(self.loss)

    def build_layers(self, x):

        outputs = { 'output': x }
        n_input = x.get_shape().as_list()[-1]

        for layer in self.layers:
            outputs, n_input = layer.build(outputs["output"], n_input)

        return outputs

    def build(self, x, y):

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
            model_str += " -> " +  str(layer.shape[1])

        model_str += " -> y"

        return model_str