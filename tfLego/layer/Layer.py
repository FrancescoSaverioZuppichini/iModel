from tfLego.protocols import  Buildable

class Layer(Buildable):
    """
    A layer is a unique piece of a model.
    """
    def __init__(self, name=None, shape=None):
        """
        :param name: A string that identifies the layer
        """
        self.name = name

        if(shape == None):
            self.shape = []
        else:
            self.shape = shape

    def build(self, x, n_input, prev_layer, model, *args, **kwargs):
        pass
