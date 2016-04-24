import numpy as np
from numpy.matlib import arange,tanh
from sklearn.metrics.regression import mean_squared_error
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
from numpy.linalg.linalg import pinv

from .neuralnetwork import NeuralNet


class Autoencoder(NeuralNet):

    def __init__(self, nInputs):
        self.type = "Autoencoder"
        NeuralNet.__init__(self, nInputs, nInputs)

    def train(self, data, training):
        'En esta funcion se realiza 10-Fold CV para entrenar la red con una expansion de entre 20-75%.'
        'El algoritmo de entrenamiento es Descenso por Gradiente Estocastico o Extreme Learning Machine.'
        # 10-Fold Cross Validation
        folds = 10; iters = 10;
        kf = KFold(data.shape[0], n_folds=folds)
        hiddenNodes = arange(2*data.shape[1])+1
        Error_HNodes = []
        Nets_HNodes = []
        for j in hiddenNodes:
            self.setHiddenNodes([j])
            Mean_error_iter = []
            Mean_nets_iter = []
            for train_index, val_index in kf:
                X, Xval = data[train_index], data[val_index]
                Error_iter = []
                Nets_iter = []
                for i in np.arange(iters):
                    self.initialization() # Inicializaciones comunes
                    if training == 'elm':
                        Out,H,N = self.sim(X)
                        H = H[-1]
                        pseudoinverse = pinv(H)
                        beta = np.dot(pseudoinverse,X)
                        self.Weights[-1] = beta
                        # Validation
                        Out_val,H_val,N_val = self.sim(Xval)
                        # Se guarda el error y la red
                        MSE = [mean_squared_error(Xval,Out_val)]
                        Networks = [self.Weights]
                    Error_iter.append(np.min(MSE))
                    Nets_iter.append(Networks[np.argmin(MSE)])
                Mean_error_iter.append(np.mean(Error_iter))
                Mean_nets_iter.append(Nets_iter[np.argmin(Error_iter)])
            Error_HNodes.append(np.mean(Mean_error_iter))
            Nets_HNodes.append(Mean_nets_iter[np.argmin(Mean_error_iter)])
        self.Weights = Nets_HNodes[np.argmin(Error_HNodes)]
        Final_Error = np.min(Error_HNodes)
        selected_Nodes = hiddenNodes[np.argmin(Error_HNodes)]
        self.setHiddenNodes([selected_Nodes])
        return Final_Error
