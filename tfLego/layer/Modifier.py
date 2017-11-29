import tensorflow as tf

from tfLego.layer.Layer import Layer


class Modifier(Layer):

    def __init__(self, handler, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.handler = handler

    def build(self, x, *args, **kwargs):

        return self.handler(x,*args, **kwargs)

class Reshaper(Layer):

    def build(self, x, *args, **kwargs):

        return { 'activated' : tf.reshape(x, self.shape) }

