import tensorflow as tf

class LayerFactory:
    @staticmethod
    def getLayer():
        return tf.keras.layers.Flatten()
    @staticmethod
    def getDenseLayer(n_inputs,activation):
        return tf.keras.layers.Dense(n_inputs, activation=activation)
    @staticmethod
    def getDropout(dropoutrate):
        return tf.keras.layers.Dropout(dropoutrate)