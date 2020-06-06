import tensorflow as tf

from neuralnetworks.activations.ActivationFunctions import ActivationFunction
"""
A Factory class for create layera

"""
class LayerFactory:
    @staticmethod
    def getFlattenLayer():
            return tf.keras.layers.Flatten()

    @staticmethod
    def getDenseLayer(n_inputs,activation):
        return tf.keras.layers.Dense(n_inputs,activation=activation)

    @staticmethod
    def getInputLayer(n_input):
         return tf.keras.layers.Input(shape=(n_input,))

    @staticmethod
    def getDropout(dropoutrate):
        return tf.keras.layers.Dropout(dropoutrate)