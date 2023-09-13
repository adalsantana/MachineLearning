import numpy as np

class AdalineGD(object):
    """
        ADAptive LInear NEuron classifier 

        Parameters
        -----------
        eta : float 
            Learning rate (between0.0 and 1.0)
        n_iter : int 
            Numbe of passes over the training dataset. (i.e. also referred to as number of epochs)
        random_state : int 
            Random number generator seed for random weight initialization 

        Attributes
        -----------

        w_ : 1d-array 
            Weights after fitting 
        cost_ : list 
            Sum of sqaures cost function value in each epoch
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state


    def fit(self, x, y):
        """ Fit training data 
            This method loops over all individual samples in the training set and updates the weights according to the perceptron learning rule.
            Parameters: 
            -----------
            x : {array-like}, shape = [n_samples, m_features]
                Training vectors, where n_samples is the number of samples and n_features is the number of features 
            y : {array-like}, shape = [n_samples]
                Target values 
            
            Returns:
            ----------
            self : object 
        """

        rgen = np.random.RandomState(self.random_state)
        # rgen is a numpy random number generator 
        # The weights aren't initalized to zero bc the learning rate η (eta) only has an effect on the classification outcome if the weights are initalized to non-zero values
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + x.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(x)
            output = self.activate(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * x.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0 #why divide by 2?
            self.cost_.append(cost)
        return self

    def net_input(self, x): 
        """ Calculate net input """
        return np.dot(x, self.w_[1:] + self.w_[0])
    
    def activation(self, x):
        """ Computer linear activation"""
        return x
    
    def predict(self, x): 
        """ Return class label after unit step"""
        return np.where(self.activation(self.net_input(x)) >= 0.0, 1, -1)