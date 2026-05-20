import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Write code here
    x_transpose = np.transpose(X)
    internal_x = x_transpose @ X
    internal_x_inverse = np.linalg.inv(internal_x)
    weights = internal_x_inverse @ x_transpose @ y
    return weights