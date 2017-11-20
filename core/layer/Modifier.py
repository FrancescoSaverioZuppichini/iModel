from core.layer.Layer import Layer
import tensorflow as tf
from tensorflow.contrib import rnn


class Modifier(Layer):
    def __init__(self, handler):
        super().__init__()

        self.handler = handler

    def build(self, x, *args, **kwargs):

        return self.handler(x,*args, **kwargs)
