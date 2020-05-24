# -*- coding: utf-8 -*-
"""
Implement a builder to build a generic neural networl

"""
from neuralnetworks.model.FeedForwardNN import MLN
class Builder:

    def __init__(self):
        pass
    
    def addLayer(self, model:MLN, layer):
        model.addLayer(layer)
        return model

    def setOptimizer(self,model, optimizer):
        model.setOptimizer(optimizer)
        return model
    def setLossFunction(self, model, function):
        model.setLossFunction(function)
        return model
    def setMetrics(self, model, metrics):
        model.setMetrics(metrics)
        return model


    def compile(self, model):
        model.compile()
        return model
    
    def init(self):
        return MLN()
