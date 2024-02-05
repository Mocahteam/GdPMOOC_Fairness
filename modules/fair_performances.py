import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

def separate_pred(X, Y, pred, sf):
    """Separates the predicted probabilities according to the two groups of a specified binary sensitive feature.

    Parameters
    ----------
    X : pd.DataFrame
        The test set
    Y : np.ndarray of shape (n, 1)
        The test labels
    pred : np.ndarray of shape (n, 1)
        The predicted probabilities of positive predictions
    sf : str
        The name of the binary sensitive feature

    Returns
    -------
    couple of np.ndarray
        The couple of predicted probabilities separated (pred_proba_sf0, pred_proba_sf1)
    """
    X_sf0 = X[X[sf] == 0]
    X_sf1 = X[X[sf] == 1]
    Y_sf0 = Y[X_sf0.index]
    Y_sf1 = Y[X_sf1.index]
    pred_sf0 = pred[X_sf0.index]
    pred_sf1 = pred[X_sf1.index]
    return (Y_sf0, pred_sf0, Y_sf1, pred_sf1)


def group_performances(X_test=None, Y_test=None, pred=None, sf=None):
    """Computes the MADD.
    
    Parameters
    ----------
    h : float or str
        The bandwidth (previously called the probability sampling parameter)
    X_test : pd.DataFrame
        The test set
    Y_test : np.ndarray of shape (n, 1)
        The test labels
    pred : np.ndarray of shape (n, 1)
        The model's prediction
    sf: str
        The name of the binary sensitive feature
    
    Returns
    -------
    float
        The F1 value
    """
    if (Y_test is not None) and (pred is not None):
        if sf is None:
            raise Exception("sf should be given (it sould be the column name of the sensitive feature).")
        else:
            y_sf0, pred_sf0, y_sf1, pred_sf1 = separate_pred(X_test, Y_test, pred, sf)
    
    if (X_test is None) and (pred is None):
        if (pred_sf0 is None) or (pred_sf1 is None):
            raise Exception("Both pred_sf0 and pred_sf1 should be given.")
    

    #f1_sf0 = f1_score(y_sf0, pred_sf0, average='weighted')
    #f1_sf1 = f1_score(y_sf1, pred_sf1, average='weighted')


    f1_sf0 = roc_auc_score(y_sf0, pred_sf0, average='weighted')
    f1_sf1 = roc_auc_score(y_sf1, pred_sf1, average='weighted')

    return [f1_sf0, f1_sf1]

