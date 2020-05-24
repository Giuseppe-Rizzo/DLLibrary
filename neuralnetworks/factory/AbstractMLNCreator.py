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
    def createNetwork(self,builder, layers, optimizer, loss, metrics):
        pass
    
  
"""
 Concrete factory Creator
 
"""
class NetworkCreator(AbstractNetworkCreator):
     def __init__(self):
         pass
     
     def createNetwork(self,builder, layers:list, optimizer, loss, metrics): #SequentialMLN
         #network= MLN()
         network= builder.init()
         for layer in layers:
             network= builder.addLayer(network,layer)
         network = builder.setOptimizer(network, optimizer)
         network =builder.setLossFunction(network,loss)
         network= builder.setMetrics(network, metrics)
         network =builder.compile(network)
         return network
