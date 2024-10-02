import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.15,
            (1, 1): 0.45
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.35,
            (1, 1): 0.35
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.25,
            (0, 1): 0.25,
            (1, 0): 0.25,
            (1, 1): 0.25
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.15,
            (0, 0, 1): 0.15,
            (0, 1, 0): 0.15,
            (0, 1, 1): 0.15,
            (1, 0, 0): 0.35,
            (1, 0, 1): 0.35,
            (1, 1, 0): 0.25,
            (1, 1, 1): 0.25,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        dependent = any(not np.isclose(X[key[0]] * Y[key[1]], val) for key, val in X_Y.items())

        return dependent

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        independent = all(np.isclose(val/C[key[2]], (X_C[(key[0], key[2])]/C[key[2]]) * Y_C[(key[1], key[2])]/C[key[2]]) for key, val in X_Y_C.items())
        
        return independent

def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    # (λ^(k) * e^(−λ)) / k! = k * log(λ) − λ − log(k!)
    log_p = k * np.log(rate) - rate - np.log(np.math.factorial(k))
    
    return log_p

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = np.zeros(len(rates))

    for i, rate in enumerate(rates): 
        log_likekihood = 0
        for k in samples:
            log_likekihood += poisson_log_pmf(k, rate) #likelihood is product of individuals observations and beacause log -> log(product) = sum(log)
        likelihoods[i] = log_likekihood
    
    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    likelihoods = get_poisson_log_likelihoods(samples, rates) # might help
    rate = rates[np.argmax(likelihoods)]
    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    # MLE turns out to be equivalent to the sample mean of the n observations in the sample.
    mean = np.mean(samples)
    return mean

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    if std == 0:
        return 0
    std_pow = np.power(std,2)
    exp = -np.power(x - mean, 2) / (2 * std_pow)
    p = (1 / np.sqrt(2 * np.pi * std_pow)) * np.power(np.e, exp)
    
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        self.data=np.copy(dataset)
        self.class_value=class_value

        class_data = dataset[dataset[:, -1] == class_value][:, :-1]
        self.class_data=class_data
        self.mean=np.mean(class_data, axis=0)
        self.std=np.std(class_data, axis=0)
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = len(self.class_data) / len(self.data)
        
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = 1
        for i in range(len(self.class_data.T)):
            likelihood *= normal_pdf(x[i], self.mean[i], self.std[i])
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        return posterior

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        self.ccd0=ccd0
        self.ccd1=ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = 0 if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x) else 1
        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = 0
    cnt_correct_classified=0

    for instance in test_set:
        instance_class = instance[-1]
        label = map_classifier.predict(instance[:-1])
        if label == instance_class:
            cnt_correct_classified += 1
    
    acc = cnt_correct_classified / len(test_set)
    return acc

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    d = x.shape[0]
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    diff = (x - mean).reshape((d,1))
    exponent = -0.5 * np.matmul(np.matmul(diff.T, inv), (x - mean))
    pdf = (1 / np.power(np.pi * 2, d/2) * np.power(det, 0.5)) * np.exp(exponent) 
    
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        self.data=dataset
        self.class_value=class_value

        class_data = dataset[dataset[:, -1] == class_value, :-1]

        self.class_data = class_data
        self.mean=np.mean(class_data, axis=0)
        self.cov_matrix=np.cov(class_data, rowvar=False)
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = len(self.class_data) / len(self.data)
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = multi_normal_pdf(x, self.mean, self.cov_matrix)
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_prior() * self.get_instance_likelihood(x)
        return posterior

class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0=ccd0
        self.ccd1=ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = 0 if self.ccd0.get_prior() > self.ccd1.get_prior() else 1
        return pred

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0=ccd0
        self.ccd1=ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        ## i dont understand why i need to do product on all xi. if im doing it im considering this as inconditional which is may not.
        
        pred = 0 if np.prod(self.ccd0.get_instance_likelihood(x)) > np.prod(self.ccd1.get_instance_likelihood(x)) else 1
        return pred

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        class_data = dataset[dataset[:,-1] == class_value][:, :-1]

        self.data=dataset
        self.class_data=class_data
        self.class_value=class_value
    
    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = len(self.class_data) / len(self.class_data)
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = 1

        n_i = len(self.class_data)
        for i, feature in enumerate(self.class_data.T[:-1]):
            V_j = len(set(feature))
            n_i_j = len(feature[x[i] == feature[:]])

        likelihood *= (n_i_j + 1) / (n_i + V_j) if n_i + V_j != 0 else (n_i_j + 1) / EPSILLON
        
        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_prior() * self.get_instance_likelihood(x)
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0=ccd0
        self.ccd1=ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = 0 if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x) else 1
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = 0
        cnt_correct_classified=0

        for instance in test_set:
            instance_class = instance[-1]
            label = self.predict(instance[:-1])
            if label == instance_class:
                cnt_correct_classified += 1
        
        acc = cnt_correct_classified / len(test_set)
        return acc


