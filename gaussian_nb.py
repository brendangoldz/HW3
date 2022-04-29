import numpy as np

class GaussianNB():
    eps = 1e-6
    
    def fit(self, X, Y):
        m = X.shape[0]
        split_X = np.array([X[Y[:,0] == y] for y in np.unique(Y)], dtype=object)
        priors = []
        means = []
        xvars = []
        for j in range(split_X.shape[0]):
            priors.append(len(split_X[j])/m)
            mean = np.mean(split_X[j], axis=0)
            var_ = np.var(split_X[j], axis=0)
            means.append(mean)
            xvars.append(var_)
        return means, xvars, np.array(priors)
    
    def prob(self, x, mean, var, num_y):
        equation_1 = (-num_y/2) * np.log(2*np.pi) - 0.5*np.sum(np.log(var+self.eps))
        equation_2 = 0.5*(np.sum(np.square(x-mean)/(var+self.eps), 1))
        return equation_1 - equation_2

    def getPFC(self, x, Y, means, xvars, prior):
        probs_per_class = []
        m = x.shape[0]
        classes = np.unique(Y)
        class_len = len(classes)
        prob_ =  np.zeros((m, class_len))
        for j in range(len(np.unique(Y))):
            prob = self.prob(x, means[j],  xvars[j], len(np.unique(Y))) + np.log(prior[j])
            prob_[:,j] = prob
        return prob_
