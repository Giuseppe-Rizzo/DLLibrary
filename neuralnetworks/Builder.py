# -*- coding: utf-8 -*-
"""
Implement a builder to build a generic neural networl

"""
from abc import ABCMeta, abstractmethod

from neuralnetworks.model.FeedForwardNN import MLN


class AbstractBuilder(metaclass=ABCMeta):

    @abstractmethod
    def setOptimizer(self, model, optimizer):
        pass

    @abstractmethod
    def setLossFunction(self, model, function):
        pass

    @abstractmethod
    def setMetrics(self, model, metrics):
        pass

    @abstractmethod
    def compile(self, model):
        pass

    @abstractmethod
    def init(self):
        pass
    @abstractmethod
    def build(self, input, output):
        pass


class Builder(AbstractBuilder):
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
        pass

    def build(self, input, output):
        return MLN(input, output)
