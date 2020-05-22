import math
class BackPropagationUtils:
    @staticmethod            
    def activate(weights,inputs):
        activation = weights[-1] # the bias
        for i in range(len(weights)-1):
            activation+= (weights[i]* inputs[i])
        return activation
    
    @staticmethod
    def transfer(activation):
        return 1.0/(1.0 + math.exp(-activation))  
    
    @staticmethod
    def transferderivative(output):
        return output * (1.0-output)

    @staticmethod
    def forwardpropagate(network, example):
        inputs= example
        for layer in network:
            new_inputs=[]
            for neuron in layer:
                activation = BackPropagationUtils.activate(neuron['weights'],inputs)
                neuron['output'] = BackPropagationUtils.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs= new_inputs
        return inputs
   
    @staticmethod
    def backwardpropagate(network, expected):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if i != len(network)-1:
                for j in range(len(layer)): # for each node of the current layer
                #update the weights
                    error =  0.0
                    for neuron in network[i+1]:
                        error+=  (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron =layer[j]
                    #print(expected, neuron['output'])
                    errors.append(expected[j]-neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta']= errors[j]- BackPropagationUtils.transferderivative(neuron['output'])
        
    @staticmethod
    def update_weights(network, row, l_rate):
        for i in range(len(network)):
            inputs= row[:-1]
            if i!=0:
                inputs=[neuron['output'] for neuron in network[i-1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j]+= l_rate *neuron['delta']*inputs[j]
                neuron['weights'][-1]+=l_rate *neuron['delta']
   