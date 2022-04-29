import numpy as np
from os.path import abspath

class DataHandler():
    URL = ""

    def __init__(self, URL):
        self.URL = abspath(URL)

    def zscore_data(self, X):
        means = np.zeros((1, X.shape[1]))
        stds = np.zeros((1, X.shape[1]))
        for i in range(X.shape[1]):
            data_ = X[:, i]
            mean = np.mean(data_)
            means[:, i] = mean
            std = np.std(data_, ddof=1)
            stds[:, i] = std
        return means, stds

    def apply_zscore(self, means, stds, X):
        return (X - means)/stds
    
    def parse_data_no_header(self):
        return np.genfromtxt(
            self.URL, delimiter=","
        )
    
    def parse_data(self):
        return np.loadtxt(
            self.URL, delimiter=",", skiprows=2
        )

    def shuffle_data(self, data, seed=0):
        np.random.seed(seed)
        # Shuffle Data
        np.random.shuffle(data)
        return data
    
    def dynamic_split(self, X, Y, mean, std):
        m = X.shape[0]
        # X = self.apply_zscore(mean, std, X)
        split_X = np.array([X[Y[:,0] == y] for y in np.unique(Y)], dtype=object)
        split_means = np.array([X[Y[:,0] == y].mean(axis=0) for y in np.unique(Y)], dtype=object)
        split_vars = np.array([X[Y[:,0] == y].var(axis=0) for y in np.unique(Y)], dtype=object)
        priors = []
        for j in range(split_X.shape[0]):
            priors.append(len(split_X[j])/m)
        return split_X, np.array(priors), split_means, split_vars

    def split_data(self, data):
        # Create Arrays for Training vs Validation
        training_index = round(len(data)*2/3)
        train = data[0:training_index]
        validation_index = training_index+1
        validation = data[validation_index:]
        return train, validation

    def getXY(self, data, xInd, yInd):
        return data[:, :xInd], data[:, yInd:]