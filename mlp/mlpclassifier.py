import copy

from sklearn.metrics.regression import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import confusion_matrix

from .autoencoder import Autoencoder
from .dataSet import dataset
from .multiLayerPerceptron import MLPerceptron

def threshold(matrix):
    matrix[matrix <= 0.5] = 0
    matrix[matrix > 0.5] = 1
    return matrix

if __name__ == "__main__":

    data = dataset()
    # Se carga el cjto de datos
    data.load()
    # Preprocesado media 0 var 1
    data.preprocess(training = 'gauss')
    # division entrenamiento y test
    data.split(random=True)

    # Se crea un Deep Autoencoder
    MLP = MLPerceptron(nInputs=data.X.shape[1],nOutputs=1)
    # Entrenamiento del DAE
    MLP.train(data=data.X,target=data.T,deep=True)
    MLP.getInfo()
    # Guardamos la Red
    Networks = [copy.deepcopy(MLP)]

    # Errores
    Out,H_pred,N_pred = MLP.sim(data.X)
    Accuracy = [accuracy_score(data.T, Out)]
    print("Accuracy: "+str(Accuracy))

    # # Figura
    # plt.figure(); plt.title('Imputacion con '+str(DeepAutoencoder.hiddenLayers)+' capa oculta')
    # plt.plot(data.X[data.M==0], 'b*-')
    # plt.plot(Xmi[data.M==0], 'r*-')
    # plt.show(block=False)
    # print("Error de Missing Values: " + str(Error_missing_values))

    for i in np.arange(5)+1:
        print(i)
        # Se obtienen las salidas de la capa oculta
        Out,H,N = MLP.sim(data.X)
        H = H[-1]
        # Creacion y entrenamiento de un AE
        AE = Autoencoder(H.shape[1])
        AE.train(H, training='elm')

        '''#Prueba del AE
        Prueba,H_prueba,N_prueba = AE.sim(H)
        for j in np.arange(data.Dataset.shape[1]):
            plt.figure()
            plt.plot(data.Dataset[:,j])
            Prueba = np.asarray(Prueba)
            plt.plot(Prueba[:,j],'r')
            plt.show()'''

        # Adicion de los pesos de salida del AE al MLP
        MLP.hiddenNodes.append(AE.hiddenNodes[0])
        MLP.setHiddenNodes(MLP.hiddenNodes)
        MLP.Weights[-1] = np.transpose(AE.Weights[-1])
        MLP.Weights.append(np.random.randn(MLP.hiddenNodes[-1],MLP.outputs))
        # Fine Tuning
        MLP.fineTuning(data.X, data.T)
        MLP.getInfo()
        # Guardamos la red
        Networks.append(copy.deepcopy(MLP))
        # Guardamos los pesos
        np.save('PesosDeepMLP_'+str(len(MLP.Weights))+'layers',MLP.Weights)

        # Errores
        Out,H_pred,N_pred = MLP.sim(data.X)
        print("Accuracy: "+str(Accuracy))
        Accuracy.append(accuracy_score(data.T, Out))

        # Criterio de Parada
        if i>=1 and Accuracy[i] < np.max(Accuracy):
            break

    # Se selecciona la mejor red
    MLP = Networks[np.argmax(Accuracy)]

    # Se pasa el TEST por la red y comprobamos el error final
    Out,H_pred,N_pred = MLP.sim(data.Xtest)
    Accuracy_test = accuracy_score(data.Ttest, Out)
    print("Accuracy_test: "+str(Accuracy_test))
    Conf_matrix = confusion_matrix(data.Ttest,Out)
    print("Confusion Matrix: "+ str(Conf_matrix))
