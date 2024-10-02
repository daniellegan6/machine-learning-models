# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    
    X = (X - np.mean(X, axis=0))/(np.max(X, axis=0) - np.min(X, axis=0))
    y = (y - np.mean(y, axis=0))/(np.max(y, axis=0) - np.min(y, axis=0))
    
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    ones_col = np.ones((len(X)))
    X = np.c_[ones_col, X]
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    J = 0  # We use J for the cost.
    m = len(X)  

    h = np.dot(X, theta)  
    squared_error = (h - y) ** 2  # Squared differences
    J = np.sum(squared_error) / (2 * m)  
    
    
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    m = len(X)

    for i in range(num_iters):
        h = np.dot(X, theta)
        error = h - y
        theta = theta - (alpha / m) * np.dot(X.T, error) # notice - theta and gradient have the same length array
        J = compute_cost(X,y, theta)
        J_history.append(J)

    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    Xt = X.T
    Xt_mult_X = np.matmul(Xt, X)
    inv_Xt_mult_X = np.linalg.inv(Xt_mult_X) 
    inv_mult_Xt = np.matmul(inv_Xt_mult_X, Xt)
    pinv_theta = np.dot(inv_mult_Xt, y)

    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    m = len(X)

    for i in range(num_iters):
        h = np.dot(X, theta)
        error = h - y
        theta = theta - (alpha / m) * np.dot(X.T, error) # notice - theta and gradient have the same length array
        J = compute_cost(X,y, theta)
        J_history.append(J)

        if i > 0 and J_history[i - 1] - J < 1e-8:
            break

    return theta, J_history


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}

    for alpha in alphas:
        theta = np.ones(X_train.shape[1])
        theta = efficient_gradient_descent(X_train, y_train, theta, alpha, iterations)[0]
        J_val = compute_cost(X_val, y_val, theta)
        alpha_dict[alpha] = J_val
        
    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []

    while len(selected_features) < 5:
        best_J_val = float('inf')
        best_feature = None
        
        for feature in range(X_train.shape[1]):
            if feature not in selected_features:
                selected_features.append(feature)

                X_train_new = apply_bias_trick(X_train[:, selected_features])
                X_val_new = apply_bias_trick(X_val[:, selected_features])
                
                np.random.seed(42)
                theta_init = np.random.random(size=len(selected_features) + 1)
                theta = efficient_gradient_descent(X_train_new, y_train, theta_init, best_alpha, iterations)[0]
                J_val = compute_cost(X_val_new, y_val, theta)

                if J_val < best_J_val:
                    best_J_val = J_val
                    best_feature = feature

                selected_features.pop()

        selected_features.append(best_feature)

    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    new_cols = []
    for i, col in enumerate(df_poly.columns):
        for new_col in df_poly.columns[i:]:
            if col != new_col:
                feature_name = col + '*' + new_col
            else:
                feature_name = col + '^2'

            new_col_pd = df_poly[col] * df_poly[new_col]
            new_col_pd.name = feature_name
            new_cols.append(new_col_pd)
            
    df_poly = pd.concat([df_poly] + new_cols, axis=1)
   
    return df_poly