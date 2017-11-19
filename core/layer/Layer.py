from abc import ABC, abstractmethod
from core.protocols import  Buildable

class Layer(Buildable):
    """
    A layer is a unique piece of a model.
    """
    def __init__(self, name=None):
        """
        :param name: A string that identifies the layer
        """
        self.name = name

