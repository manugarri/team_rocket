import numpy as np
import pandas as pd
from sklearn import datasets, preprocessing
from copy import deepcopy

class dataset():   
    def __init__(self):
        self.Dataset = None
        self.Targets = None
        self.X = None
        self.T = None
        self.Xtest = None
        self.Ttest = None

    def load(self):
        data = pd.read_csv('Training.csv',sep=';',decimal=',',low_memory=False)
        #data = pd.read_csv('Adrian_balanceado.csv',sep=';',decimal=',',low_memory=False)
        self.Targets = data['NEO_flag'].values
        del data['NEO_flag']
        self.Dataset = data.values
        for i in range(5):
            self.Dataset = np.concatenate((self.Dataset,self.Dataset[self.Targets==1]),axis=0)
            self.Targets = np.concatenate((self.Targets,self.Targets[self.Targets==1]),axis=0)
        '''final_data = pd.DataFrame(self.Dataset, columns=data.columns.values)
        final_targets = pd.DataFrame(self.Targets, columns=['Targets'])
        final_data.to_csv('Balanced_data.csv',sep=';')
        final_targets.to_csv('Balanced_targets.csv',sep=';')'''

    def randomize(self):
        indices = np.random.choice(range(self.Dataset.shape[0]), self.Dataset.shape[0], replace=False)
        self.Dataset = self.Dataset[indices]
        self.Targets = self.Targets[indices]
    
    def preprocess(self, training = None, test = None):
        if training == 'gauss':
            #Preprocesado: media 0 varianza 1
            self.Dataset = preprocessing.scale(self.Dataset)
        elif training == 'scale':
            min_max_scaler = preprocessing.MinMaxScaler()
            self.X = min_max_scaler.fit_transform(self.X)
            self.Xtest = min_max_scaler.fit_transform(self.Xtest)
        if test == '1ofK':
            # Targets a 1 of K
            lb = preprocessing.LabelBinarizer()
            if self.Targets:
                lb.fit(np.arange(len(np.unique(self.Targets))))
                self.Targets = lb.transform(self.Targets)
            else:
                lb.fit(np.arange(len(np.unique(self.T))))
                self.T = lb.transform(self.T)
                lb.fit(np.arange(len(np.unique(self.Ttest))))
                self.Ttest = lb.transform(self.Ttest)

    def artMissing(self, features, percentage, cjto):
        if cjto == 'training':
            self.X_missing = deepcopy(self.X) # Con esto se crea otro objeto lista igual a la de in1
            samples = np.around((self.X_missing.shape[0]-1)*np.random.rand(np.around((percentage/100)*self.X_missing.shape[0]),len(features)))
            for j in np.arange(len(features)):
                for i in np.arange(samples.shape[0]):
                    self.X_missing[samples[i,j],features[j]] = np.NaN
            indexNaN = np.isnan(self.X_missing)
            self.M = 1*(~indexNaN)
        if cjto == 'test':
            self.Xtest_missing = deepcopy(self.Xtest) # Con esto se crea otro objeto lista igual a la de in1
            samples = np.around((self.Xtest_missing.shape[0]-1)*np.random.rand(np.around((percentage/100)*self.X_missing.shape[0]),len(features)))
            for j in np.arange(len(features)):
                for i in np.arange(samples.shape[0]):
                    self.Xtest_missing[samples[i,j],features[j]] = np.NaN
            indexNaN = np.isnan(self.Xtest_missing)
            self.Mtest = 1*(~indexNaN)
         
    def split(self,random=False):
        if random:
            indices = np.random.choice(range(self.Dataset.shape[0]), self.Dataset.shape[0], replace=False)
        else:
            indices = np.arange(self.Dataset.shape[0])
        # 1/3 Test y 2/3 Entrenamiento
        indices_training = indices[:np.around((2.0/3.0)*len(indices))]
        indices_test = indices[np.around((2.0/3.0)*len(indices)):]
        self.X = self.Dataset[indices_training,:]
        self.T = self.Targets[indices_training]
        self.Xtest = self.Dataset[indices_test,:]
        self.Ttest = self.Targets[indices_test]

    def preImp(self, strategy, cjto):
        if cjto == 'training':
            Imp = preprocessing.Imputer(missing_values='NaN',strategy=strategy,axis=0)
            Imp.fit(self.X_missing)
            return Imp.transform(self.X_missing)
        if cjto == 'test':
            Imp = preprocessing.Imputer(missing_values='NaN',strategy=strategy,axis=0)
            Imp.fit(self.Xtest_missing)
            return Imp.transform(self.Xtest_missing)
