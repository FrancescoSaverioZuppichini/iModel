from abc import ABC, abstractmethod
from core.protocols import  Buildable

class Layer(Buildable):
    def __init__(self, name=None):
        self.name = name
