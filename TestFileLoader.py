# -*- coding: utf-8 
#from scipy.io import arff
from neuralnetworks.LayerFactory import LayerFactory
from datasource.FileLoader import FileLoader
from preprocessing.Preprocessor import Preprocessor 
#from neuralnetworks.deprecated.MLNWithoutKeras import MLNWithoutKeras
from neuralnetworks.factory.AbstractMLNCreator import *
from neuralnetworks.LayerFactory import LayerFactory
from neuralnetworks.Builder import Builder


path= 'C:/Users/Giuseppe/Downloads/sonar_csv.csv'
scaline = 'standard' #standard
#arff.loadarff(path)
#print(arff)
a = FileLoader('csv',path)
originaldf= a.load()
print(originaldf)
pp = Preprocessor()
originaldf= pp.replace(originaldf,'M',1.0)
originaldf= pp.replace(originaldf,'R',0.0)
print(originaldf)
array= originaldf.to_numpy()
y= array[:,-1] # access to n-2 colunm
n_inputs = len(array[0]) - 1
print("Input", n_inputs)
n_outputs = len(set([row[-1] for row in array]))    
builder=Builder()
# create a multilayer perceptron for classification
network_cretor= AbstractNetworkCreator().createNetworkCreator()
layers= [LayerFactory.getLayer(), LayerFactory.getDenseLayer(n_inputs, 'relu'), LayerFactory.getDenseLayer(n_outputs,'softmax')]
mln= network_cretor.createNetwork(builder,layers)
mln.fit(array[:,:-1],array[:,-1],200)

# create an autencode for


print("End of program")


        
        
        
    