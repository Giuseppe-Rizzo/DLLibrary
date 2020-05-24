import tensorflow as tf
from enum import Enum
class ActivationFunctionsName(Enum):
    relu =1 #'adam'
    softmax = 2#'nadam'
    sigmoid = 3
    softplus=4
    @staticmethod
    def fromValue(n):
        return n.name

class ActivationFunction:
    @staticmethod
    def lookup(name):
        return tf.keras.activations.get(name)
