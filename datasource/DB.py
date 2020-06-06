# -*- coding: utf-8 -*-
from datasource.FileLoader import FileLoader
from sklearn.model_selection import train_test_split, KFold,LeaveOneOut
class Dataset:
  def __init__(self,file):
    a = FileLoader('csv',file)
    originaldf = a.load()
    self.__array__ = originaldf.to_numpy()
    self.__instances__ = self.__array__[:,:-1]
    self.__labels__ = self.__array__[:,-1]

  def getInstances(self):
    return self.__instances__
  def getLabels(self):
    return self.__labels__
  def setInstances(self, i):
    self.__instances__=i

  def setLabels(self,i):
    self.__labels__=i

  def split_training_test(self, test_size_rate):
    return train_test_split(self.__instances__,self.__labels__,test_size=test_size_rate)

  def cross_validation(self, nOfFold):
     splitter =KFold(n_splits=nOfFold)
     return splitter.split(self.__instances__,self.__labels__)

  def llov(self):
    splitter = LeaveOneOut()
    return splitter.split(self.__instances__,self.__labels__)