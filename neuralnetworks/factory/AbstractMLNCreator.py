from abc import abstractmethod

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
    def createNetwork(self,builder, layers):
        pass
    
  
"""
 Concrete factory Creator
 
"""
class NetworkCreator(AbstractNetworkCreator):
     def __init__(self):
         pass
     
     def createNetwork(self,builder, layers:list): #SequentialMLN
         #network= MLN()
         network= builder.init()
         for layer in layers:
             network= builder.build(network,layer)
         return network
