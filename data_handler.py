import numpy as np
from os.path import abspath

class DataHandler():
    URL = ""

    def __init__(self, URL):
        self.URL = abspath(URL)

    def zscore_data(self, tX, X):
        means = np.zeros((1, tX.shape[1]))
        stds = np.zeros((1, tX.shape[1]))
        for i in range(tX.shape[1]):
            data_ = tX[:, i]
            mean = np.mean(data_)
            means[:, i] = mean
            std = np.std(data_, ddof=1)
            stds[:, i] = std
            # X[:,1] = (X[:,1]-mean)/std
        return means, stds

    def apply_zscore(self, means, stds, X):
        for i in range(X.shape[1]):
            X[:,i] = np.subtract(X[:,i], means[:, i])/stds[:, i]
        return X

    def parse_data_no_header(self):
        return np.genfromtxt(
            self.URL, delimiter=","
        )
    
    def parse_data_multi(self):
        return np.genfromtxt(
            self.URL, delimiter=",", dtype="|U19"
        )

    def parse_data(self):
        return np.loadtxt(
            self.URL, delimiter=",", skiprows=1, usecols=[1, 2, 3]
        )

    def shuffle_data(self, data, seed=0):
        np.random.seed(seed)
        # Shuffle Data
        np.random.shuffle(data)
        return data
    
    def dynamic_split(self, data):
        X, Y = self.getXY(data, -1, -1)
        m = X.shape[0]
        mean, std = self.zscore_data(X, X)
        X = self.apply_zscore(mean, std, X)
        split_X = np.array([X[Y[:,0] == y] for y in np.unique(Y)], dtype=object)
        priors = []
        for j in range(split_X.shape[0]):
            priors.append(len(split_X[j])/m)
        return split_X, np.array(Y), np.array(priors)

    def split_data(self, data):
        # Create Arrays for Training vs Validation
        training_index = round(len(data)*2/3)
        train = data[0:training_index]
        validation_index = training_index+1
        validation = data[validation_index:]
        return train, validation

    def getXYFolded(self, data):
        result = list()
        data = np.array(data)
        for X in data:
            result.append(self.getXY(X))
        return result

    def getXY(self, data, xInd, yInd):
        return data[:, :xInd], data[:, yInd:]
            
    def filter(self, tX, tY, classifier):
        tY = np.char.decode(np.array(tY).astype(np.bytes_), 'UTF-8')
        Y = np.array_like(tY)
        print(Y)
        # print(tY)
        m, n = tX.shape
        # for i in range(m):
        #     if tY[i, 0] == classifier:
        #         Y[i, 0] == 