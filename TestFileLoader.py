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
from neuralnetworks.activations.ActivationFunctions import *

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
layers = [LayerFactory.getInputLayer(n_inputs), LayerFactory.getDenseLayer(n_inputs, ActivationFunction.lookup(ActivationFunctionsName.fromValue(ActivationFunctionsName.relu))), LayerFactory.getDenseLayer(n_outputs,ActivationFunction.lookup(ActivationFunctionsName.fromValue(ActivationFunctionsName.softmax)))]
optimizer = Optimizers.lookup(OptimizersName.fromValue(OptimizersName.adadelta))
loss_function_lookup = LossFunction.lookup(LossFunctionsName.fromValue(LossFunctionsName.sparse_categorical_crossentropy))
mln = network_creator.createNetwork(builder, layers, layers[0], layers[-1],optimizer, loss_function_lookup,['accuracy'])
mln.fit(dataset.getInstances(),label,200)
print(label, dataset.getLabels())
evaluation = mln.evaluate(dataset.getInstances(),dataset.getLabels())
print ("Loss: ",evaluation[0])
print ("Accuracy: ",evaluation[1])

#create a logistic regression
layers = [LayerFactory.getInputLayer(n_inputs),LayerFactory.getDenseLayer(1, ActivationFunction.lookup(ActivationFunctionsName.fromValue(ActivationFunctionsName.softmax)))]
builder2 = Builder()
optimizer2 = Optimizers.lookup(OptimizersName.fromValue(OptimizersName.adam))
loss = LossFunction.lookup(LossFunctionsName.fromValue(LossFunctionsName.MAE))
logistic_regressor = AbstractNetworkCreator.createNetworkCreator().createNetwork(builder2,layers,layers[0],layers[-1],optimizer2,loss,['accuracy'])
logistic_regressor.fit(dataset.getInstances(),dataset.getLabels(),200)



print("End of program")





