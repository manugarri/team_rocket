import os
import json
import urllib2
import gzip
import StringIO

import numpy as np
import pandas as pd

from mlp import MLPerceptron
from publish_tweet import status_update

SOURCE_URL = 'http://minorplanetcenter.net/Extended_Files/mpcorb_extended.json.gz'
DATA_PATH = './data/mpcorb_extended.json'

 mlp = MLPerceptron(15, 1, 'mlp/PesosDeepMLP_3layers.npy'

def download_data():
    response = urllib2.urlopen(SOURCE_URL)
    compressedFile = StringIO.StringIO()
    compressedFile.write(response.read())
    compressedFile.seek(0)
    decompressedFile = gzip.GzipFile(fileobj=compressedFile, mode='rb')
    with open(DATA_PATH, 'w') as outfile:
        outfile.write(decompressedFile.read())
        outfile.write(decompressed_file.read())

def parse_data():
    print('Processing data')
    data = pd.read_json(DATA_PATH)
    columns = [
        'H', 'G','Epoch','M','Peri','Node', 'i', 'e', 'n', 'a','Num_obs', 'Num_opps', 'Arc_length', 'rms',
            ]
    data = data[columns]
    data.reset_index(inplace=True)
    return data.as_matrix()

def predict(data):

def main():
    """main logic, downloads latest MPCORB daily orbits, computes the risk and
    sends a tweet with the current status"""
    download_data()
    print('Loading data')
    data = parse_data(data)
    print('Predicting PHAs')
    Out,H_pred,N_pred = mlp.sim(data)
    predicted_pha = data[Out==1.0]
    print('Updating tweet')
    status_update(predicted_pha.shape[0])

if __name__ == '__main__':
    main()

