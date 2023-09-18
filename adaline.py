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
    
class AdalineSGD(object):
    """
        ADAptive LInear NEuron classifier 

        Parameters
        -----------
        eta : float 
            Learning rate (between0.0 and 1.0)
        n_iter : int 
            Number of passes over the training dataset. (i.e. also referred to as number of epochs)
        shuffle : bool (default: True)
            Shuffles training data every epoch if True to prevent cycles 
        random_state : int 
            Random number generator seed for random weight initialization 

        Attributes
        -----------

        w_ : 1d-array 
            Weights after fitting 
        cost_ : list 
            Sum of sqaures cost function value in each epoch
    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state
    
    def _shuffle(self, X, y): 
         """ Shuffle training data
            Works via the permutation function in np.random. We generate a random sequence of unique numbers in the range 0 to 100. 
            Those numbers can then be used as indices to shuffle our feature matrix and class label vector. 
            We can then use the fit method to train AdalineSGD classifier and use our plot_decision_regions function to plot the training plots
        """
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """ Initilaize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_[1:] += self.rgen.normal(loca = 0.0, scale = 0.01, size = 1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target): 
        """ Apply Adaline learning to rule to update the weights"""
        output = self.activation(self.net_input)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    def _partial_fit(self, X, y): 
        """ Fit training data without reinitializing the weights. 
            To use this method for online learning with streaming data, call the partial_fit function but feed in individual samples
            e.g. ada.partial_fit(X_std[0, :], y)
        """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1: 
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else: 
            self._update_weights(X, y)
        return self

    def fit(self, X, y):
        """ Fit training data 
            This method loops over all individual samples in the training set and updates the weights according to the perceptron learning rule.
            Parameters: 
            -----------
            X : {array-like}, shape = [n_samples, m_features]
                Training vectors, where n_samples is the number of samples and n_features is the number of features 
            y : {array-like}, shape = [n_samples]
                Target values 
            
            Returns:
            ----------
            self : object 
        """
        self._initialize_weights(X.shape[1])
        self.cost = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y): 
                cost.append(self._update_weights(xi, target))
                avg_cost = sum(cost) / len(y)
                self.cost.append(avg_cost)
        return self

    def net_input(self, x): 
        """ Calculate net input """
        return np.dot(x, self.w_[1:] + self.w_[0])
    
    def activation(self, x):
        """ Computer linear activation. 
            Serves no purpose in current state. Essentially an identity function for this exercise. 
            This function is meant to illustrate how information flows through a single layer neural network: 
                features from the input data, net input, activation, and output
        """
        return x
    
    def predict(self, x): 
        """ Return class label after unit step"""
        return np.where(self.activation(self.net_input(x)) >= 0.0, 1, -1)
