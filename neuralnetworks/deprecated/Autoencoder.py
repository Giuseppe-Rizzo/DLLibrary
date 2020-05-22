# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 12:40:22 2018

@author: Giuseppe
"""

import tensorflow as tf 
import numpy as np

class  Autoencoder: 
     def __init__(self, inputdim):
         encoding= 217
         self.model= tf.keras.models.Sequential()
         #self.model.add(tf.keras.layers.Flatten())
         self.model.add(tf.keras.layers.Dense(encoding, activation='relu'))
         self.model.add(tf.keras.layers.Dense(inputdim, activation='sigmoid')) 
         self.model.compile(optimizer='adam', loss='mean_error_squared')
         
         
     def fit(self,x,e):
         t=tf.convert_to_tensor(x,dtype=tf.float32)
         print(t) 
         #normalized = tf.reshape(t, [57, 217,1])
         normalized= tf.nn.batch_normalization(t, mean=0, variance=1, scale=None,offset=None,variance_epsilon=0.001)
         #print(normalized)
         self.model.fit(normalized,normalized,steps_per_epoch=5,epochs=e)
         
     def evaluate (self,x,y):
         self.model.evaluate(x,y)
