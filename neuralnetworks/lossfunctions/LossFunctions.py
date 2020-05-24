import tensorflow as tf
from enum import Enum
class LossFunctionsName(Enum):
    sparse_categorical_crossentropy =1 #'adam'
    MAE = 2#'nadam'
    MSE =3 #'adadelta'
    categorical_hinge= 5 #'adamax'
    @staticmethod
    def fromValue(n):
        return n.name

class LossFunction:
    @staticmethod
    def lookup(name):
        return tf.keras.losses.get(name)
