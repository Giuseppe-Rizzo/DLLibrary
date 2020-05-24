# -*- coding: utf-8 -*-
from datasource.FileLoader import FileLoader

class Dataset:
  def __init__(self,file):
    a = FileLoader('csv',file)
    originaldf = a.load()
    array= originaldf.to_numpy()
    self.__instances__ = array[:,:-1]
    self.__labels__ = array[:,-1]

  def getInstances(self):
    return self.__instances__
  def getLabels(self):
    return self.__labels__
  def setInstances(self, i):
    self.__instances__=i

  def setLabels(self,i):
    self.__labels__=i

