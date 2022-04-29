import numpy as np

class GaussianNB():
    
    def __init__(self, means, x_vars):
        self.means = means
        self.x_vars = x_vars
    
    def prob(self, x, mean, var):
        equation_1 = 0
        equation_2 = 0
        if var < 0.0001:
            var = 0.0001
        equation_1 = 1/(np.sqrt(2 * np.pi * var))    

        e2denom = np.square(2 * var)
        e2num = np.square(x - mean)

        equation_2 = np.exp(-(e2num/e2denom))

        prob = equation_1 * equation_2
        return prob

    def getPFC(self, priors, x, Y):
        probs_per_class = []
        total_probability = 0
        for i in range(len(np.unique(Y))):
            prior = priors[i]
            prob_ = 1
            for j in range(x.shape[0]):
                prob_ = prob_ * self.prob(x[j], self.means[i][j], self.x_vars[i][j])
            total_probability += (np.multiply(prior, prob_))
            probs_per_class.append(np.multiply(prior, prob_))
        return probs_per_class, total_probability
    
    def trainModel(self, X, Y, classes, priors):
        preds = []
        for i in range(X.shape[0]):
            probs, total_probability = self.getPFC(priors, X[i,:], Y)
            probs = probs/(total_probability+0.0001)
            max_prob_ind = probs.argmax()
            preds.append(classes[max_prob_ind])
        return preds;