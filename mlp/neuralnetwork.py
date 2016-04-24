import numpy as np
from numpy.matlib import arange,tanh

class NeuralNet:
    Weights = []
    def __init__(self, nInputs, nOutputs, weight_file=None):
        self.inputs = nInputs
        self.hiddenLayers = 1
        self.outputs = nOutputs
        self.hiddenNodes = [1]

        if weight_file:
            self.Weights = np.load(weight_file)
            self.hiddenLayers = len(self.Weights) -1
            self.hiddenNodes = [16,32]
        else:
            pass

    def setHiddenNodes(self, hiddenNodes):
        self.hiddenNodes = hiddenNodes
        self.setHiddenLayers(len(hiddenNodes))

    def setHiddenLayers(self, hiddenLayers):
        self.hiddenLayers = hiddenLayers

    def initialization(self):
        self.Weights = []
        self.Weights.append(np.random.randn(self.inputs,self.hiddenNodes[0]))
        for i in arange(self.hiddenLayers-1)+1:
            self.Weights.append(np.random.randn(self.hiddenNodes[i-1],self.hiddenNodes[i]))
        self.Weights.append(np.random.randn(self.hiddenNodes[-1],self.outputs))

    def sim(self,Xi):
        H = []
        N = []
        import ipdb; ipdb.set_trace()
        N.append(np.dot(Xi,self.Weights[0]))
        H.append(tanh(N[0]))
        for i in arange(self.hiddenLayers-1)+1:
            N.append(np.dot(H[i-1],self.Weights[i]))
            H.append(tanh(N[i]))
        N.append(np.dot(H[-1],self.Weights[-1]))
        Out = np.dot(H[-1],self.Weights[-1])
        return Out,H,N

    def getInfo(self):
        print("Tipo de red: ", self.type)
        print("Dimension de entrada: ", str(self.inputs))
        print("Numero de capas ocultas: ", str(self.hiddenLayers))
        print("Numero de nodos ocultos: ", str(self.hiddenNodes))
        print("Dimension de salida: ", str(self.outputs))
