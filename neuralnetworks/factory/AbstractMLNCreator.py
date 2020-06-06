from abc import abstractmethod
from neuralnetworks.Builder import AbstractBuilder
import tensorflow as tf
"""
Abstract Factory for network
"""
class AbstractNetworkCreator:
    def __init__(self):
        pass
    
    @staticmethod
    def createNetworkCreator():
        return NetworkCreator()
    
    @abstractmethod
    def createNetwork(self,builder:AbstractBuilder, layers,input,output, optimizer, loss, metrics):
        pass
    
  
"""
 Concrete factory Creator
 
"""
class NetworkCreator(AbstractNetworkCreator):
     def __init__(self):
         pass
     
     def createNetwork(self,builder, layers:list,input, output, optimizer, loss, metrics): #SequentialMLN
         #network= MLN()

         l = None
         #functional approach to feedforward implementation of a network
         for layer in layers:
             if l is None:
                l = layer
             else:
                 l = layer(l)
         network= builder.build(input,l)
         network = builder.setOptimizer(network, optimizer)
         network =builder.setLossFunction(network,loss)
         network= builder.setMetrics(network, metrics)
         network =builder.compile(network)
         return network
