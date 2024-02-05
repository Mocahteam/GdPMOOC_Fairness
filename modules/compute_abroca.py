import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt
from sklearn import preprocessing

def compute_roc(y_scores, y_true):
    """
    Function to compute the Receiver Operating Characteristic (ROC) curve for a set of predicted probabilities and the true class labels.
    y_scores - vector of predicted probability of being in the positive class P(X == 1) (numeric)
    y_true - vector of true labels (numeric)
    Returns FPR and TPR values
    """
    fpr, tpr, _ = metrics.roc_curve(y_true, y_scores)
    return fpr, tpr


def compute_auc(y_scores, y_true):
    """
    Function to Area Under the Receiver Operating Characteristic Curve (AUC)
    y_scores - vector of predicted probability of being in the positive class P(X == 1) (numeric)
    y_true - vector of true labels (numeric)
    Returns AUC value
    """
    auc = metrics.roc_auc_score(y_true, y_scores)
    return auc


def interpolate_roc_fun(fpr, tpr, n_grid):
    """
    Function to Use interpolation to make approximate the Receiver Operating Characteristic (ROC) curve along n_grid equally-spaced values.
    fpr - vector of false positive rates computed from compute_roc
    tpr - vector of true positive rates computed from compute_roc
    n_grid - number of approximation points to use (default value of 10000 more than adequate for most applications) (numeric)

    Returns  a list with components x and y, containing n coordinates which  interpolate the given data points according to the method (and rule) desired
    """
    roc_approx = interpolate.interp1d(x=fpr, y=tpr)
    x_new = np.linspace(0, 1, num=n_grid)
    y_new = roc_approx(x_new)
    return x_new, y_new


def slice_plot(
    majority_roc_fpr,
    minority_roc_fpr,
    majority_roc_tpr,
    minority_roc_tpr,
    majority_group_name="baseline",
    minority_group_name="comparison",
    fout="./slice_plot.png",
):
    """
    Function to create a 'slice plot' of two roc curves with area between them (the ABROCA region) shaded.

    majority_roc_fpr, minority_roc_fpr - FPR of majority and minority groups
    majority_roc_tpr, minority_roc_tpr - TPR of majority and minority groups
    majority_group_name - (optional) - majority group display name on the slice plot
    minority_group_name - (optional) - minority group display name on the slice plot
    fout - (optional) -  File name (including directory) to save the slice plot generated

    No return value; displays slice plot & file is saved to disk
    """
    plt.figure(1, figsize=(6, 5))
    plt.title("ABROCA - Slice Plot")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(
        majority_roc_fpr,
        majority_roc_tpr,
        label="{o} - Baseline".format(o=majority_group_name),
        linestyle="-",
        color="r",
    )
    plt.plot(
        minority_roc_fpr,
        minority_roc_tpr,
        label="{o} - Comparison".format(o=minority_group_name),
        linestyle="-",
        color="b",
    )
    plt.fill(
        majority_roc_fpr.tolist() + np.flipud(minority_roc_fpr).tolist(),
        majority_roc_tpr.tolist() + np.flipud(minority_roc_tpr).tolist(),
        "y",
    )
    plt.legend()
    plt.savefig(fout)
    plt.show()


def compute_abroca(
    df,
    pred_col,
    label_col,
    protected_attr_col,
    majority_protected_attr_val,
    n_grid=10000,
    plot_slices=False,
    lb=0,
    ub=1,
    limit=1000,
    file_name="slice_image.png",
):
    # Compute the value of the abroca statistic.
    """
    df - dataframe containing colnames matching pred_col, label_col and protected_attr_col
    pred_col - name of column containing predicted probabilities (string)
    label_col - name of column containing true labels (should be 0,1 only) (string)
    protected_attr_col - name of column containing protected attribute (should be binary) (string)
    majority_protected_attr_val name of 'majority' group with respect to protected attribute (string)
    n_grid (optional) - number of grid points to use in approximation (numeric) (default of 10000 is more than adequate for most cases)
    plot_slices (optional) - if TRUE, ROC slice plots are generated and saved to file_name (boolean)
    lb (optional) - Lower limit of integration (use -numpy.inf for -infinity) Default is 0
    ub (optional) - Upper limit of integration (use -numpy.inf for -infinity) Default is 1
    limit (optional) - An upper bound on the number of subintervals used in the adaptive algorithm.Default is 1000
    file_name (optional) - File name (including directory) to save the slice plot generated

    Returns Abroca value
    """
    if df[pred_col].between(0, 1, inclusive="both").any():
        pass
    else:
        print("predictions must be in range [0,1]")
    if len(df[label_col].value_counts()) == 2:
        pass
    else:
        print("The label column should be binary")
        #print(df[label_col])
    if len(df[protected_attr_col].value_counts()) == 2:
        pass
    else:
        print("The protected attribute column should be binary")
    # initialize data structures
    # slice_score = 0
    prot_attr_values = df[protected_attr_col].value_counts().index.values
    fpr_tpr_dict = {}

    # compute roc within each group of pa_values
    for pa_value in prot_attr_values:
        if pa_value != majority_protected_attr_val:
            minority_protected_attr_val = pa_value
        pa_df = df[df[protected_attr_col] == pa_value]
        fpr_tpr_dict[pa_value] = compute_roc(pa_df[pred_col], pa_df[label_col])

    # compare minority to majority class; accumulate absolute difference btw ROC curves to slicing statistic
    majority_roc_x, majority_roc_y = interpolate_roc_fun(
        fpr_tpr_dict[majority_protected_attr_val][0],
        fpr_tpr_dict[majority_protected_attr_val][1],
        n_grid,
    )
    minority_roc_x, minority_roc_y = interpolate_roc_fun(
        fpr_tpr_dict[minority_protected_attr_val][0],
        fpr_tpr_dict[minority_protected_attr_val][1],
        n_grid,
    )

    # use function approximation to compute slice statistic via piecewise linear function
    if list(majority_roc_x) == list(minority_roc_x):
        f1 = interpolate.interp1d(x=majority_roc_x, y=(majority_roc_y - minority_roc_y))
        f2 = lambda x, acc: abs(f1(x))
        slice, _ = integrate.quad(f2, lb, ub, limit)
    else:
        print("Majority and minority FPR are different")
        exit(1)

    if plot_slices == True:
        slice_plot(
            majority_roc_x,
            minority_roc_x,
            majority_roc_y,
            minority_roc_y,
            majority_group_name="baseline",
            minority_group_name="comparison",
            fout=file_name,
        )

    return slice
