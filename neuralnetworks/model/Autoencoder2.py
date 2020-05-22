class AutoEncoder: FeedForwardNN
    __encoding__
    def __init__(self, encoding):
        self.__model__ = tf.keras.Sequential()
        self.__encoding__= encoding
    
    def addLayer(self,layer_type):
        if layer_type != 'encoding':
        self.__model__.add(l)
        self.__model__compile()
        
    