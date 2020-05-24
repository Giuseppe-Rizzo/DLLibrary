import tensorflow as tf
from enum import Enum
class OptimizersName(Enum):
    adam =1 #'adam'
    nadam = 2#'nadam'
    adadelta =3 #'adadelta'
    adamax= 5 #'adamax'
    @staticmethod
    def fromValue(n):
        return n.name

class Optimizers:
    @staticmethod
    def lookup(name):
        return tf.keras.optimizers.get(name)
