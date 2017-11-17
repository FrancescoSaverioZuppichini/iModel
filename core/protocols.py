from abc import ABC, abstractmethod


class Buildable(ABC):

    @abstractmethod
    def build(self, *args, **kwargs):
        pass

class Runnable(ABC):

    @abstractmethod
    def run(self, *args, **kwargs):
        pass
