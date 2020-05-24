# -*- coding: utf-8
#from scipy.io import arff
from neuralnetworks.LayerFactory import LayerFactory
from datasource.FileLoader import FileLoader
from preprocessing.Preprocessor import Preprocessor
#from neuralnetworks.deprecated.MLNWithoutKeras import MLNWithoutKeras
from neuralnetworks.factory.AbstractMLNCreator import *
from neuralnetworks.LayerFactory import LayerFactory
from neuralnetworks.Builder import Builder
from neuralnetworks.optimizer.optimizers import *
from neuralnetworks.lossfunctions.LossFunctions import *
from datasource.DB import Dataset
from pandas import DataFrame

path= 'C:/Users/Giuseppe/Downloads/sonar_csv.csv'
scaline = 'standard' #standard
dataset = Dataset(path)
pp = Preprocessor()

#label conversions
label = pp.replace(DataFrame(dataset.getLabels()),'R',0.0)
label = pp.replace(label,'M',1.0)
dataset.setLabels(label)

 # access to n-2 colunm
n_inputs = len(dataset.getInstances()[0])
#print("Input", n_inputs, dataset.getInstances()[60])
n_outputs =  2 #len(set([row[-1] for row in dataset.getLabels()]))
builder = Builder()
# create a multilayer perceptron for classification
network_creator = AbstractNetworkCreator().createNetworkCreator()
layers = [LayerFactory.getLayer(), LayerFactory.getDenseLayer(n_inputs, 'relu'), LayerFactory.getDenseLayer(n_outputs,'softmax')]
optimizer = Optimizers.lookup(OptimizersName.fromValue(OptimizersName.adadelta))
mln = network_creator.createNetwork(builder, layers, optimizer, LossFunction.lookup(LossFunctionsName.fromValue(LossFunctionsName.sparse_categorical_crossentropy)),['accuracy'])
mln.fit(dataset.getInstances(),label,200)
print(label, dataset.getLabels())
evaluation = mln.evaluate(dataset.getInstances(),dataset.getLabels())
print (evaluation)


#create an autencode for


print("End of program")





