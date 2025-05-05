import numpy as np
import pandas as pd

# normalize data sets
def normalize_data(X):
    for i in range(X.shape[1]):
        temp = X[:,i]
        temp = temp - np.min(temp)
        temp = temp/np.max(temp)
        temp = (temp - 0.5)*2
        X[:,i] = temp
    return X

def get_data(data_name = 'cancer'):
    # import data set
    # Breast Cancer Wisconsin data set
    if data_name == 'cancer':
        df = pd.read_csv('data/cancer.csv')
        Y = df['diagnosis']
        X = df.drop(['id','diagnosis'], axis = 1)

        # Change the labels to 0 & 1
        new_Y = [Y[i] == "B" for i in range(len(Y))]
        Y = np.array(new_Y)

    # Banknote authentication data set
    elif data_name == 'banknote':
        df = pd.read_csv('data/banknote.csv')
        Y = df['class']
        X = df.drop('class', axis = 1)
        Y = Y.to_numpy(dtype='int')

    else:
        raise NotImplementedError(f"Data set {data_name} not implemented")
    
    # Normalizing data
    X = X.to_numpy()
    X = normalize_data(X)

    return X, Y