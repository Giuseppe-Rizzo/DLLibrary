from abc import ABCMeta, abstractmethod
import tensorflow as tf
from pandas import DataFrame


class AbstractNetwork(metaclass=ABCMeta):
    def __init__(self):
        self.__model__ = tf.keras.models.Sequential()

    @abstractmethod
    def addLayer(self, l):
        pass

    @abstractmethod
    def setOptimizer(self, opt):
        pass

    @abstractmethod
    def setLossFunction(self, lossFunction):
        pass

    @abstractmethod
    def setMetrics(self, metrics_list):
        pass

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def fit(self, x, y, e):
        pass

    @abstractmethod
    def evaluate(self, x, y):
        pass


class MLN(AbstractNetwork):

    def addLayer(self,l):
         self.__model__.add(l)

    def setOptimizer(self,opt):
        self.__optimizer__ = opt # e.g. adam

    def setLossFunction(self, lossFunction):
        self.__loss__ = lossFunction #sparse_categorical_entropy

    def setMetrics(self,metrics_list):
        self.__metrics__ = metrics_list

    def compile(self):
         self.__model__.compile(optimizer=self.__optimizer__,
              loss= self.__loss__,
              metrics= self.__metrics__)

    def fit(self,x,y,e):
         examples = None;
         labels = None
         if (type(x) is DataFrame):
             examples =x.to_numpy()
         else:
             examples =x
         if (type(y) is DataFrame):
             labels  = y.to_numpy()
         else:
             labels = y

         self.__model__.fit(examples, labels,epochs=e)
         
    def evaluate(self,x,y):
         if (type(x) is DataFrame):
             examples =x.to_numpy()
         else:
             examples =x
         if (type(y) is DataFrame):
             labels  = y.to_numpy()
         else:
             labels = y
         return self.__model__.evaluate(examples,labels)
         
 