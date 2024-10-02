import numpy as np
import pandas as pd


def pearson_correlation( x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values. 

    Returns:
    - The Pearson correlation coefficient between the two columns.    
    """
    # Calculate the mean
    mx = np.mean(x)
    my = np.mean(y)

    # Calculate the cov 0f x and y
    cov = np.sum((x-mx)*(y-my))

    # Calculate the denominator
    sx = np.sqrt(np.sum((x-mx)**2))
    sy = np.sqrt(np.sum((y-my)**2))

    # Calculate the Pearson correlation 
    r = cov/(sx*sy)

    return r

def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).  
    """
    # remove non-numeric columns 
    nonumeric = X.select_dtypes(include=[np.number])

    # Dictionary to hold feature and their correlation to y
    cor = {}
    for feat in nonumeric.columns:
        cor[feat] = pearson_correlation(nonumeric[feat], y)

    # Sort the dictionary in descending
    sortcor = sorted(cor.items(), key=lambda x: x[1], reverse=True)

    # give the top n_features 
    best_features = [feat for feat, cora in sortcor[:n_features]]

    return best_features

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        np.random.seed(self.random_state)
        m, n = X.shape
        X_b = np.insert(X, 0, 1, axis=1)  # Bias trick
        self.theta = np.random.random(n + 1)  # Random initialization 

        for _ in range(self.n_iter):
            z = np.dot(X_b, self.theta)
            h = 1 / (1 + np.exp(-z))  # Sigmoid 
            gradient = np.dot(X_b.T, (h - y)) / m
            self.theta -= self.eta * gradient

            # Cost function 
            epsilon = 1e-5  # avoid log(0)
            J = (-1 / m) * (np.dot(y, np.log(h + epsilon)) + np.dot((1 - y), np.log(1 - h + epsilon)))
            self.Js.append(J)
            self.thetas.append(self.theta.copy())

            if len(self.Js) > 1 and abs(self.Js[-2] - self.Js[-1]) < self.eps:
                break
        #########################################
    
    def predict(self, X):
        """
    Return the predicted class labels for a given instance.
    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
    """
        s = np.insert(X, 0, 1, axis=1)  # Bias trick
        z = np.dot(s, self.theta)
        h = 1 / (1 + np.exp(-z))  # Sigmoid 
        preds = (h >= 0.5).astype(int)
        return preds

    

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    

    
    np.random.seed(random_state)
    # Shuffle the dataset 
    permutation = np.random.permutation(X.shape[0])
    X_permuted = X[permutation]
    y_permuted = y[permutation]

    # Compute the size of each fold.
    size_per_fold = len(X) // folds
    accuracies = []

    for idx in range(folds):
        # Determine indices 
        test_idx_start = idx * size_per_fold
        test_idx_end = test_idx_start + size_per_fold if idx != folds - 1 else len(X)
        X_test = X_permuted[test_idx_start:test_idx_end]
        y_test = y_permuted[test_idx_start:test_idx_end]

        # Determine indices 
        train_idx = np.concatenate((np.arange(test_idx_start), np.arange(test_idx_end, len(X))))
        X_train = X_permuted[train_idx]
        y_train = y_permuted[train_idx]

        # Train the model and evaluate 
        algo.fit(X_train, y_train)
        predictions = algo.predict(X_test)
        fold_accuracy = np.mean(predictions == y_test)
        accuracies.append(fold_accuracy)

    # Calculate the mean of the computed accuracies.
    accuracy = np.mean(accuracies)
    return accuracy
    
def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    res = 1 / (sigma * np.sqrt(2 * np.pi))
    res1 = -0.5 * ((data - mu) / sigma) ** 2
    return res * np.exp(res1)

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = []

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        self.weights = np.full(self.k, 1 / self.k)
        self.mus = np.random.choice(data.flatten(), self.k, replace=False)
        self.sigmas = np.random.rand(self.k) + 1  # to avoid zero
        self.responsibilities = np.zeros((len(data), self.k))
        
    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        pdfs = norm_pdf(data[:, np.newaxis], self.mus, self.sigmas)
        weighted_pdfs = pdfs * self.weights
        self.responsibilities = weighted_pdfs / weighted_pdfs.sum(axis=1, keepdims=True)

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        total_responsibilities = self.responsibilities.sum(axis=0)
        self.weights = total_responsibilities / len(data)
        self.mus = (data[:, np.newaxis] * self.responsibilities).sum(axis=0) / total_responsibilities
        self.sigmas = np.sqrt(((data[:, np.newaxis] - self.mus) ** 2 * self.responsibilities).sum(axis=0) / total_responsibilities)


    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)
        previous_cost = float('inf')

        for i in range(self.n_iter):
            self.expectation(data)
            self.maximization(data)

            # Calculate the log likelihood cost directly in the fit function
            pdfs = norm_pdf(data[:, np.newaxis], self.mus, self.sigmas)
            weighted_log_likelihood = np.log(pdfs * self.weights)
            total_log_likelihood = weighted_log_likelihood.sum(axis=1)
            current_cost = -total_log_likelihood.sum()

            self.costs.append(current_cost)

            if abs(previous_cost - current_cost) < self.eps:
                break
            previous_cost = current_cost
            
    def get_dist_params(self):
            return self.weights, self.mus, self.sigmas      

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    pdf = np.sum(weights*norm_pdf(data.reshape(-1,1) , mus , sigmas), axis=1)

    return pdf
class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.prior = {}
        self.feature_models = {}



    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
       # Initialize priors and models for each class and feature
        classes = np.unique(y)
        self.prior = {cls: np.mean(y == cls) for cls in classes}
        self.feature_models = {cls: {i: EM(self.k, random_state=self.random_state) for i in range(X.shape[1])} for cls in classes}

        for cls in classes:
            for i in range(X.shape[1]):
                feature_data = X[y == cls, i]
                self.feature_models[cls][i].fit(feature_data.reshape(-1, 1))

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []
        for instance in X:
            posteriors = []
            for cls in self.prior:
                likelihood = 1
                for i in range(X.shape[1]):
                    weights, mus, sigmas = self.feature_models[cls][i].get_dist_params()
                    likelihood *= gmm_pdf(instance[i], weights, mus, sigmas)
                posterior = likelihood * self.prior[cls]
                posteriors.append((posterior, cls))
            preds.append(max(posteriors, key=lambda x: x[0])[1])
        return np.array(preds)
      
      
def calculate_accuracy(y_pred, y_test):
    
    accuracy = np.mean(y_pred == y_test)
    return accuracy    

      
      

      

        
def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    # Logistic Regression Model
    logistic_regression = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    logistic_regression.fit(x_train, y_train)
    lor_train_acc = np.mean(logistic_regression.predict(x_train) == y_train)
    lor_test_acc = np.mean(logistic_regression.predict(x_test) == y_test)

    # Naive Bayes Model
    naive_bayes = NaiveBayesGaussian(k=k)
    naive_bayes.fit(x_train, y_train)
    bayes_train_acc = np.mean(naive_bayes.predict(x_train) == y_train)
    bayes_test_acc = np.mean(naive_bayes.predict(x_test) == y_test)

    return {
        'lor_train_acc': lor_train_acc,
        'lor_test_acc': lor_test_acc,
        'bayes_train_acc': bayes_train_acc,
        'bayes_test_acc': bayes_test_acc
    }
def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    
    # Helper function to generate data from multivariate normal distribution
    def generate_data(means, covariances, class_labels, num_samples=500):
        features = []
        labels = []
        for mean, cov, label in zip(means, covariances, class_labels):
            # Generate samples for each class
            data = multivariate_normal(mean, cov).rvs(num_samples)
            features.append(data)
            labels.append(np.full(num_samples, label))
        # Concatenate all class samples into a single array
        features = np.vstack(features)
        labels = np.concatenate(labels)
        return features, labels

    # Parameters for dataset A (better for Naive Bayes)
    means_a = [np.array([0, 0, 0]), np.array([3, 3, 3])]
    covariances_a = [np.eye(3) * 0.5, np.eye(3) * 0.5]
    class_labels_a = [0, 1]
    dataset_a_features, dataset_a_labels = generate_data(means_a, covariances_a, class_labels_a)

    # Parameters for dataset B (better for Logistic Regression)
    means_b = [np.array([1, 1, 1]), np.array([2, 2, 2])]
    covariances_b = [np.eye(3), np.eye(3)]
    class_labels_b = [0, 1]
    dataset_b_features, dataset_b_labels = generate_data(means_b, covariances_b, class_labels_b)

    return {
        'dataset_a_features': dataset_a_features,
        'dataset_a_labels': dataset_a_labels,
        'dataset_b_features': dataset_b_features,
        'dataset_b_labels': dataset_b_labels
    }