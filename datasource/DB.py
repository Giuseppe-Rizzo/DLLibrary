# -*- coding: utf-8 -*-
from datasource.FileLoader import FileLoader
from sklearn.model_selection import train_test_split, KFold,LeaveOneOut, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold
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

  def cross_validation(self, folds):
      splitter = KFold(n_splits=folds)
      return self.split(self.__array__,splitter)


  def split(self, dataset, splitter):
    splits = []
    for tr, te, in splitter.split(dataset):
      print("XXXX", dataset[tr])
      X_train = dataset[tr, :-1]
      Y_train = dataset[tr, -1]
      X_test = dataset[te, :-1]
      Y_test = dataset[te, -1]
      splits.append({"x_training": X_train, "y_training": Y_train, "x_test": X_test, "y_training": Y_test})
    return splits

  def llov(self):
    splitter = LeaveOneOut()
    return self.split(self.__array__,splitter)

  def repeatedCV(self, folds, replicates):
    splitter= RepeatedKFold(n_splits=folds,n_repeats=replicates)
    return self.split(self.__array__,splitter)

  def stratifiedCV(self, folds):
    splitter = StratifiedKFold(folds)
    return self.split(self.__array__,splitter)

  def repeatedStratifiedCV(self, folds, replicates):
    splitter = RepeatedStratifiedKFold(n_splits=folds,n_repeats=replicates)
    return self.split(self.__array__,splitter)