import numpy as np
from data_handler import DataHandler


class LogisticalRegression():
    def __init__(self, TERMINATION_VALUE, ITERATIONS, LEARNING_RATE):
        # self.TERMINATION_VALUE = 2^-32
        # self.ITERATIONS = 10000
        # self.LEARNING_RATE = 0.001
        self.TERMINATION_VALUE = TERMINATION_VALUE
        self.ITERATIONS = ITERATIONS
        self.LEARNING_RATE = LEARNING_RATE

    def linear_mod(self, w, X, b):
        # print(w.shape)
        # print(X.T.shape)
        return np.dot(X, w) + b
    
    def sigmoid(self, wx_b):
        # Probability F(X) for where Y = 1
        return 1/(1 + np.exp(-(wx_b)))

    def weights(self, X, diff):
        m = X.shape[0]
        return (1/m) * (X.T @ (diff))

    def compute_weights_bias(self, w, b, tX, tY, vX, vY):
        twxb = self.linear_mod(w, tX, b)
        vwxb = self.linear_mod(w, vX, b)
        
        P = self.sigmoid(twxb)
        vP = self.sigmoid(vwxb)
        diffs = np.subtract(P, tY)
        
        weight = self.weights(tX, diffs)
        
        bias = np.mean(diffs)
        return weight, bias, P, vP
    
    def cost(self, Y, P):
        logP = np.log(P, where=(P>0))
        p1 = np.multiply(Y, logP)
        p2 = 1 - Y
        p3 = 1 - P
        p4 = np.log(p3, where=((p3)>0))
        cost = -(p1+np.multiply(p2, p4))
        return cost

    def calculate(self, w, b, tX, tY, vX, vY):
        # t_costs = np.empty((tX.shape[0],1))
        # v_costs = np.empty((vX.shape[0],1))
        t_costs = list()
        v_costs = list()
        b = b
        for i in range(self.ITERATIONS):
            weight, beta, P, vP = self.compute_weights_bias(w, b, tX, tY, vX, vY)
            
            # Loss Calc
            t_cost = np.mean(self.cost(tY, P))
            v_cost = np.mean(self.cost(vY, vP))
            
            # Record the costs
            t_costs.append(t_cost)
            v_costs.append(v_cost)
            
            # Gradient Recalc
            w = w - self.LEARNING_RATE * weight
            b = b - self.LEARNING_RATE * beta
           
            # if i % 100 == 0:
            #     print("TMean Cost after iteration %i: %f" % (i, np.mean(t_costs)))
            #     print("VMean Cost after iteration %i: %f" % (i, np.mean(v_cost)))
            
        losses = {
            "TR": t_costs,
            "VAL": v_costs
        }
        return w, b, losses
    
    def prediction(self, w, b, X, thresh=0.5):
        m = X.shape[0]
        P = self.sigmoid(self.linear_mod(w,X,b))
        Y_preds = np.zeros((1, m))
        for i in range(m):
            if P[i] > thresh:
                Y_preds[:, i] = 1
            else:
                Y_preds[:, i] = 0
        return Y_preds