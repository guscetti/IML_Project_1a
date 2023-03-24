import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

def fit(X, y, lam):
    """
    This function receives training data points, then fits the ridge regression on this data
    with regularization hyperparameter lambda. The weights w of the fitted ridge regression
    are returned.

    Parameters
    ----------
    X: matrix of floats, dim = (135,13), inputs with 13 features
    y: array of floats, dim = (135,), input labels)
    lam: float. lambda parameter, used in regularization term

    Returns
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression
    """
    #w = np.zeros((13,))
    # TODO: Enter your code here
    #assert w.shape == (13,)
    clf = Ridge(alpha=lam)
    clf.fit(X, y)
    w = clf.coef_
    return w

def closed_form(X,y,lamda):
    X_t = X.transpose()
    y_t = y.transpose()
    XX_t = np.dot(X_t, X)
    id = np.identity(len(XX_t))
    w = np.dot(np.linalg.inv(XX_t + lamda*id), np.dot(X_t, y))
    return w

data = pd.read_csv("train.csv")
y = data["y"].to_numpy()
data = data.drop(columns="y")
X = data.to_numpy()


lambdas = [0.1, 1, 10, 100, 200]
n_folds = 10



print(fit(X,y,0.1))
print('break')
print(closed_form(X,y,0.1))
