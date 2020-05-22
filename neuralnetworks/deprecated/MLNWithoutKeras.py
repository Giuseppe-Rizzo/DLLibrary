from random import random
import math
from  neuralnetworks.optimizer.BackPropagationUtils import BackPropagationUtils
class  MLNWithoutKeras:
    
    def __init__(self,n_input,n_hidden,n_output):
        self.__network__= list()
        hidden_layer=[{'weights':[random() for i in range(n_input+1)]} for i in range(n_hidden)]
        self.__network__.append(hidden_layer)
        output_layer = [{'weights':[random() for i in range(n_hidden+1)]} for i in range(n_output)]
        self.__network__.append(output_layer)
        
    def getNetwork(self):
        return self.__network__
    
    def train(self,network,examples,epoch, l_rate,n_outputs):
     for e in range(epoch):
         sum_error=0.0
         for example in examples:
             outputs = BackPropagationUtils.forwardpropagate(network, example)
             expected = [0 for i in range(n_outputs)]
             expected[-1] = 1
             sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
             BackPropagationUtils.backwardpropagate(network,expected)
             BackPropagationUtils.update_weights(network, example, l_rate)
         #for layer in network:
          #   print(layer)    
         print('>epoch=%d, lrate=%.3f, error=%.3f' % (e, l_rate, sum_error))

    def predict(self,network, example):
        print('Example', example)
        prediction = BackPropagationUtils.forwardpropagate(network, example)
        print ('Prediction', prediction)
        return prediction.index(max(prediction))
#a = MLNWithoutKeras(3,4,5)
#print(a.getNetwork())