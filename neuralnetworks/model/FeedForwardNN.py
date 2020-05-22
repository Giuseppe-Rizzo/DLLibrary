import tensorflow as tf 

class  MLN: 
     def __init__(self):
         self.__model__ = tf.keras.models.Sequential()
         
     def addLayer(self,l):
         self.__model__.add(l)
         self.__model__.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
         
           
     def fit(self,x,y,e):
         self.__model__.fit(x, y,epochs=e)
         
     def evaluate(self,x,y):
         self.__model__.evaluate(x,y)
         
 