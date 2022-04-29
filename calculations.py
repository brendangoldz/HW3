import numpy as np


class Calculations():
    def __init__(self, Y, Y_):
        self.Y = Y
        self.Y_h = Y_
    
    def setup(self):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(self.Y.shape[0]):
            if self.Y_h[0, i] == 1 and self.Y[i, 0] == 1:
                TP += 1;
            elif self.Y_h[0, i] == 1 and self.Y[i, 0] == 0:
                FP += 1;
            elif self.Y_h[0:, i] == 0 and self.Y[i, 0] == 0:
                TN += 1;
            elif self.Y_h[0, i] == 0 and self.Y[i, 0] == 1:
                FN += 1;        
        return TP, TN, FP, FN
    
    def setup_multi(self):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        P = 0
        classifiers = np.unique(self.Y)
        for i in range(self.Y.shape[0]):
            for j, classi in enumerate(classifiers):
                if self.Y_h[:, i] == self.Y[i, :]:
                    P += 1
                if self.Y_h[:, i] == classi and self.Y[i, :] == classi:
                    TP += 1;
                if self.Y_h[:, i] == classi and self.Y[i, :] != classi:
                    FP += 1;
                if self.Y_h[:, i] != classi and self.Y[i, :] != classi:
                    TN += 1;
                if self.Y_h[:, i] != classi and self.Y[i, :] == classi:
                    FN += 1;        
        return P, TP, TN, FP, FN
    
    def evaluate(self):
        TP, TN, FP, FN = self.setup()        
        precision = self.precision(TP, FP)
        recall = self.recall(TP, FN)
        return self.accuracy(TP, TN, FP, FN), precision, recall, self.fmeasure(precision, recall) 
 
    def recall(self, TP, FN):
        return (TP)/(TP+FN)
    
    def precision(self, TP, FP):
        return (TP)/(TP+FP)

    def fmeasure(self, precision, recall):
        return 2*((precision*recall)/(precision+recall))

    def accuracy(self, TP, TN, FP, FN):
        return (TP+TN)/(TP+TN+FP+FN)