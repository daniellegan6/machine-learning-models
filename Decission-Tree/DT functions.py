from queue import Queue
import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    total_instances = len(data)
    y = data[:, -1]
    _, labels_count = np.unique(y, return_counts=True)

    gini = 1 - np.sum((labels_count / total_instances) ** 2)
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    total_instances = len(data)
    y= data[:, -1]
    _, labels_count = np.unique(y, return_counts=True)

    entropy = (-1) * (np.sum((labels_count / total_instances) * np.log2(labels_count / total_instances)))
    return entropy

def chi_square(data, groups):
    chi_square = 0 
    size_total_data = len(data)
    count_total_label = sum([len(group) for group in groups.values()])

    for group in groups.values():
        size_val_data = len(group)
        expected = size_val_data / size_total_data * count_total_label
        if expected != 0:
            chi_square += ((size_val_data - expected) ** 2) / expected

    return chi_square

class DecisionNode:

    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        label, counts = np.unique(self.data[:, -1], return_counts=True)
        pred = label[np.argmax(counts)]
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
        
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        #leaf:
        if self.feature is None: 
            self.feature_importance = None
            return
    
        node_impurity = self.impurity_func(self.data)
        children_impurity_sum = 0

        for child in self.children:
            children_impurity_sum += len(child.data) / len(self.data) * self.impurity_func(child.data)
        
        delta_impurity = node_impurity - children_impurity_sum
        node_probability = len(self.data) / n_total_sample
        self.feature_importance = node_probability * delta_impurity
    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {} # groups[feature_value] = data_subset

        if self.gain_ratio: self.impurity_func = calc_entropy

        data_impurity = self.impurity_func(self.data)
        total_data = len(self.data)
        unique_feature_values = np.unique(self.data[:, feature])
        split_information = 0
        feature_impurity = 0

        for val in unique_feature_values:
            groups[val] = self.data[self.data[:, feature] == val]
            val_impurity = self.impurity_func(groups[val])
            prob = len(groups[val]) / total_data
            split_information += (prob * np.log2(prob))
            feature_impurity += (prob * val_impurity)

        goodness = data_impurity - feature_impurity

        if self.gain_ratio: 
            if split_information == 0: 
                goodness = 0
            else:
                split_information = (-1) * split_information
                goodness = goodness / split_information 

        return goodness, groups
    
    
    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """

        if self.terminal: 
            return

        # check whether node is leaf
        if self.depth >= self.max_depth or len(np.unique(self.data[:, -1])) == 1:
            self.terminal = True
            return

        max_goodness = -float('inf')
        best_feature = None
        best_groups = None

        for feature in range(self.data.shape[1] - 1):
            feature_goodness, groups = self.goodness_of_split(feature)
            if feature_goodness > max_goodness:
                max_goodness = feature_goodness
                best_feature = feature
                best_groups = groups

        # make a leaf
        if self.feature is None:
            self.terminal = True
            return
        
        self.feature = best_feature

        # chi-squared test - pruning
        if self.chi != 1:
            df = len(best_groups) - 1  # degree of freedom
            chi_val_from_table = chi_table[df][self.chi]
            chi_square_val = chi_square(self.data, best_groups)
            if chi_square_val <= chi_val_from_table:
                self.terminal = True
                return
            
        for feature_val, group in best_groups.items():
            group = np.array(group)
            child_node = DecisionNode(group,
                                    self.impurity_func,
                                    depth=self.depth + 1,
                                    chi=self.chi,
                                    max_depth=self.max_depth,
                                    gain_ratio=self.gain_ratio)
            self.add_child(child_node, feature_val)

class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the tree
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio #
        self.root = None # the root node of the tree
        
    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = DecisionNode(
            data=self.data,
            impurity_func=self.impurity_func,
            chi=self.chi,
            max_depth=self.max_depth,
            gain_ratio=self.gain_ratio
        )

        q = Queue()
        q.put(self.root)

        while not q.empty():
            node = q.get()

            if len(np.unique(node.data)) == 1:
                node.terminal = True
                continue
            
            node.split()

            if node.terminal == True: 
                continue

            for child in node.children:
                q.put(child)

        
    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        node = self.root
    
        while not node.terminal: 
            found_child = False

            for child in node.children:
                if (child.data[:, node.feature] == instance[node.feature]).all():
                    node = child
                    found_child = True
                    break
            
            if not found_child:
                break  

        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        count_correct = 0
        total_instances = len(dataset)
        for instance in dataset:
            pred = self.predict(instance)
            if pred == instance[-1]:
                count_correct += 1
        
        accuracy = (count_correct / total_instances) * 100
        return accuracy
        

    def depth(self):
        return self._calculate_depth(self.root)
    

    def _calculate_depth(self, node):
        "helper func to depth"

        if node is None:
            return 0

        depths = []

        for child in node.children:
            depths.append(1 + self._calculate_depth(child))

        return max(depths, default=0)


def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = DecisionTree(X_train, impurity_func=calc_entropy, max_depth=max_depth, gain_ratio=True)
        tree.build_tree()
        training_accuracy = tree.calc_accuracy(X_train)
        training.append(training_accuracy)
        validating_accuracy = tree.calc_accuracy(X_validation)
        validation.append(validating_accuracy)
    return training, validation


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc  = []
    depth = []
    chi_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]
    for chi in chi_values:
        tree = DecisionTree(X_train, impurity_func=calc_entropy, gain_ratio=True, chi=chi)
        tree.build_tree()
        training_accuracy = tree.calc_accuracy(X_train)
        chi_training_acc.append(training_accuracy)
        validating_accuracy = tree.calc_accuracy(X_test)
        chi_validation_acc.append(validating_accuracy)
        depth.append(tree.depth())
        
    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    if node is None:
        return 0
    
    n_nodes = 1

    for child_node in node.children:
        n_nodes += count_nodes(child_node)

    return n_nodes






