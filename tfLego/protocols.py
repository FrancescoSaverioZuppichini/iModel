from abc import ABC, abstractmethod

class Buildable(ABC):
    """
    Classes that is conform to Buildable must implement the run
    method in order to build the tensorflow graph of a model
    """

    @abstractmethod
    def build(self, *args, **kwargs):
        pass

class Runnable(ABC):
    """
    Classes that is conform to Runnable must implement the run
    method in order to run the model
    """

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

class Trainable(ABC):
    """
    Classes that is conform to Trainable must implement the train
    method in order to provide the client a way to train the model.
    """

    @abstractmethod
    def train(self, sess, train_set, epochs,val_set=None, *args, **kwargs):
        pass
