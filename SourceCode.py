from math import exp
from random import seed
from random import random

def transfer_derivative(output):
    return(output*(1-output))

def init_Network(NumInput,NumHidden,NumOutput):
    network=list()
    hiddenLayer=[{'weight':[random() for j in range(NumInput+1)]} for i in range(NumHidden)]
    network.append(hiddenLayer)
    outputLayer=[{'weight':[random() for j in range(NumHidden+1)] } for i in range(NumOutput)]
    network.append(outputLayer)
    return(network)

def activate(weight,input):
    layerWeight=weight[:-1]  #removing bias from the weight vector
    output=weight[-1]        # adding  bias to the output
    for x, w in zip(layerWeight,input):
        output += x * w
    return(output)
        

def sigmoid(z):
    return(1/(1+exp(-z)))
    
    
    

def forwardPropagation(network,input):
    LayerOutput=input
    for layer in network:
        Output=list()
        for NeuronWeights in layer:
            z=activate(NeuronWeights['weight'],LayerOutput)
            
            NeuronWeights['output']=sigmoid(z)
            
            Output.append(NeuronWeights['output'])
        LayerOutput=Output
    return(LayerOutput)
    
def backPropagate(network,expected):    
    for i in reversed(range(len(network))):
        error=list()
        layer=network[i]
        if(i==len(network)-1):                                                  # output layer
            for j in range(len(layer)):
                neuron=layer[j]
                error.append(expected[j]-neuron['output'])
                
        else:  
            for j in range(len(layer)):
                temp=0
                for neuron in network[i+1]:
                   temp += (neuron['weight'][j]  * neuron['delta']) 
                error.append(temp)
        for j in range(len(layer)):
            neuron=layer[j]
            neuron['delta']=error[j] * transfer_derivative(neuron['output'])
                                                            
               
            
def WeightUpdate(network,alpha,inp):

    for i in range(len(network)):
                
        layer=network[i]
        
        if i!=0:
            tempLayer=network[i-1]   
            inp=[x['output'] for x in tempLayer] # input for output layer is output from hidden layer
        
        for neuron in layer:
            TempWeights=neuron['weight']
            
            for j in range(len(TempWeights[:-1])):                              #removing bias from the equation
                
                neuron['weight'][j] +=  neuron['delta'] * alpha * inp[j]        #alpha is the learn rate
            neuron['weight'][j+1] +=  neuron['delta'] * alpha                     #updating bias tem
            
            
        
    
    
    
def TrainTheNetwork(network,LearnRate,NumEpochs,TrainData,NumInput,NumOutput):
    for i in range(NumEpochs):
          MseSem=0
          for OnlineData in TrainData:
              ExpectedOutput=[0 for i in range(NumOutput)]
              ExpectedOutput[OnlineData[-1]]=1
              Inp=OnlineData[:-1]                      # last data comes from y
              output=forwardPropagation(network,Inp)
              backPropagate(network,ExpectedOutput)
              WeightUpdate(network,LearnRate,Inp)
              MseSem +=sum([x[1]-x[0] for x in zip(ExpectedOutput,output)])**2
          print('epoch>> %d LearnRate>> %.5f Error>> %.5f' % (i, LearnRate, MseSem)) 
              
#####              
seed(1)
dataSet = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]        
            
NumInput= len(dataSet[0])-1
NumHidden=5
NumOutput=len(set([row[-1] for row in dataSet]))   
LearnRate=0.05
NumEpochs=10**3

network=init_Network(NumInput,NumHidden,NumOutput)
TrainTheNetwork(network,LearnRate,NumEpochs,dataSet,NumInput,NumOutput)
    