# -*- coding: utf-8 -*-
"""
Implement a builder to build a generic neural networl

"""
from neuralnetworks.model.FeedForwardNN import MLN
class Builder:

    def __init__(self):
        pass
    
    def build(self, model:MLN, layer):
        model.addLayer(layer)
        return model
    
    def init(self):
        return MLN()
