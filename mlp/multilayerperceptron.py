import numpy as np
from numpy.matlib import arange,tanh
from sklearn.metrics.classification import log_loss,accuracy_score
from sklearn.metrics.regression import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.cross_validation import KFold
from numpy.linalg.linalg import pinv

from .neuralnetwork import NeuralNet

class MLPerceptron(NeuralNet):
    def __init__(self, nInputs, nOutputs, weight_file=None):
        self.type = "Multi-Layer Perceptron"
        NeuralNet.__init__(self, nInputs, nOutputs, weight_file)

    def sim(self, Xi):
        Out,H,N = NeuralNet.sim(self, Xi)
        Out[Out <= 0.5] = 0
        Out[Out > 0.5] = 1
        return Out,H,N

    def train(self, data, target, deep):
        'En esta funcion se realiza 10-Fold CV para entrenar la red con una expansion de entre 20-75%.'
        'El algoritmo de entrenamiento es Descenso por Gradiente Estocastico.'
        # 10-Fold Cross Validation
        folds = 10; iters = 10;
        kf = KFold(data.shape[0], n_folds=folds)
        if deep:
            hiddenNodes = np.arange(data.shape[1],2*data.shape[1])+1
        else:
            hiddenNodes = np.arange(data.shape[1],10*data.shape[1])+1
        hiddenNodes = hiddenNodes[hiddenNodes>0]
        Error_HNodes = []
        Nets_HNodes = []
        for j in hiddenNodes:
            self.setHiddenNodes([j])
            Mean_error_iter = []
            Mean_nets_iter = []
            for train_index, val_index in kf:
                X, Xval = data[train_index], data[val_index]
                T, Tval = target[train_index], target[val_index]
                Error_iter = []
                Nets_iter = []
                for i in np.arange(iters):
                    self.initialization() # Inicializaciones comunes
                    Out,H,N = self.sim(X)
                    H = H[-1]
                    self.Weights[-1] = np.dot(pinv(H),T)
                    # Validation
                    Out_val,H_val,N_val = self.sim(Xval)
                    # Se guarda el error y la red
                    # MSE = [mean_squared_error(Tval,Out_val)]
                    # Error de clasificacion
                    Error = [accuracy_score(Tval, Out_val)]
                    #Error = [f1_score(Tval, Out_val)]
                    Networks = [self.Weights]
                    Error_iter.append(np.min(Error))
                    Nets_iter.append(Networks[np.argmin(Error)])
                Mean_error_iter.append(np.mean(Error_iter))
                Mean_nets_iter.append(Nets_iter[np.argmin(Error_iter)])
            Error_HNodes.append(np.mean(Mean_error_iter))
            Nets_HNodes.append(Mean_nets_iter[np.argmin(Mean_error_iter)])
        self.Weights = Nets_HNodes[np.argmin(Error_HNodes)]
        Final_Error = np.min(Error_HNodes)
        selected_Nodes = hiddenNodes[np.argmin(Error_HNodes)]
        self.setHiddenNodes([selected_Nodes])
        return Final_Error

    def fineTuning(self,data,target):
        # Una vez establecidos todos los pesos, se procede al ajuste fino
        epoch = 0
        Error = []
        Networks = []
        while epoch <= 10:
            Out,H,N = self.sim(data)
            H = H[-1]
            pseudoinverse = pinv(H)
            beta = np.dot(pseudoinverse,target)
            self.Weights[-1] = beta
            # Validation
            Out,H,N = self.sim(data)
            # Error de regresion. MSE
            #Error.append(mean_squared_error(data,Out))
            Networks.append(self.Weights)
            # Error de clasificacion
            Error.append(accuracy_score(target, Out))
            #Error.append(f1_score(target, Out))
            epoch += 1
        Final_Error = np.min(Error)
        self.Weights = Networks[np.argmin(Error)]
        return Final_Error


