import numpy as np


class Calculations():
    # true_values = Y
    # predictions = Y_

    # N = true_values.shape[1]
    # accuracy = (true_values == predictions).sum() / N
    # TP = ((predictions == 1) & (true_values == 1)).sum()
    # FP = ((predictions == 1) & (true_values == 0)).sum()
    # precision = TP / (TP+FP)
    def __init__(self, Y, Y_):
        self.Y = Y
        self.Y_h = Y_
    
    def setup(self):
        P = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(self.Y.shape[0]):
            if self.Y_h[0,i] == self.Y[i, 0]:
                P += 1;
            if self.Y_h[0,i] == 1 and self.Y[i, 0] == 1:
                TP += 1;
            if self.Y_h[0,i] == 1 and self.Y[i, 0] == 0:
                FP += 1;
            if self.Y_h[0,i] == 0 and self.Y[i, 0] == 0:
                TN += 1;
            if self.Y_h[0,i] == 0 and self.Y[i, 0] == 1:
                FN += 1;        
        return P, TP, TN, FP, FN
    
    def evaluate(self):
        P, TP, TN, FP, FN = self.setup()        
        precision = self.precision(TP, FN)
        recall = self.recall(TP, FP)
        return self.accuracy(P), precision, recall, self.fmeasure(precision, recall) 
 
    def recall(self, TP, FN):
        return (TP)/(TP+FN)
    
    def precision(self, TP, FP):
        return (TP)/(TP+FP)

    def fmeasure(self, precision, recall):
        return (2*precision*recall)/(precision+recall)

    def accuracy(self, P):
        return P/self.Y.shape[0]