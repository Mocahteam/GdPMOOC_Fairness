"""
AI fairness analysis code
@date 01/02/2024
@licence BSD-3-Clause 
"""

import pandas as pd
import numpy as np

import os
import sys
import glob
import warnings
from collections import OrderedDict

from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GroupShuffleSplit
# from modules.ordinal import OrdinalClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE 
from copy import deepcopy

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score

from sklearn.exceptions import UndefinedMetricWarning

# adding src to the system path to find scripts in there
sys.path.insert(0, "./src")
sys.path.insert(0, "./modules")

# local imports
import edx_logs_processing
import edx_events
import edx_resources
import toolbox
from ML_models import *

from threshold_optimizer import ThresholdOptimizer
import maddlib
import fair_performances
import compute_abroca as abroca

############################################################################################################################################################
#CONSTANT
############################################################################################################################################################
RAND_STATE_SEED = 2204 #for reproducibility
EXPORT_SUFFIX = "_run8AUC" #suffix to add to all exported file (for version control)
NBFOLDS = 5 # number of cross validation folds
NBFOLDS_GRID = 5 # number of inner folds for the grid search hyperparameter tuning
GRIDSEARCH = False #True do to grid-search hyperparameter tuning
LABEL_KEY = "success" # the key of the labels in the feature dataframe (successfully passed the course)
LABEL_VALUES = [0, 1] #the values of the label (pass: 0 for no, 1 for yes)
STUDENT_ID_KEY = "username_anon" # the grouping key used to do cross-validation over users (effectively using the StratifiedGroupKFold function of Sklearn)
WEEK_ID_KEY = "week"
CORREL_THRESHOLD = 0.9 # thresholds for discarding correlated features (1 means no feature removal) 
SMOTE_MINORITY = False # True/False: to turn on/off Smote over-sampling of the minority classes
FEATURES_CUMUL = True # True/False: use cumulative features over week
VERBOSE = False # True/False
SENSITIVE_ATTRIBUTES = ["Age","Gender","HDICountry","Lang","JobStatut","HasKids","Education","ParentsEducation","FirstTimeMOOC","InCohort"]


###################################################################################################################################
#FUNCTIONS
###################################################################################################################################
def clean_up_features(df):
    """
    Cleanup features based on feature inspection: remove unused events, duplicates events, and combine similar, less used events
    """
    #remove unused actions
    dfout = df.drop(list(df.filter(regex='demandhint|feedback|check_fail|rescore|reset')), axis=1)
    
    #remove duplicate features
    dfout = dfout.drop("showanswer_count", axis=1) #same as "problem_show_count"

    #combine transcript
    trcolumns = ["show_transcript_count", "hide_transcript_count"]
    dfout['showhide_transcript_count'] = dfout[trcolumns].sum(axis=1)
    dfout = dfout.drop(trcolumns, axis=1)
    
    #combine cc menu
    cccolumns = ["video_hide_cc_menu_count", "video_show_cc_menu_count"]
    dfout['showhide_cc_menu_count'] = dfout[cccolumns].sum(axis=1)
    dfout = dfout.drop(cccolumns, axis=1)
    
    #combine creation on forum
    createcolumns = ["edx.forum.comment.created_count", "edx.forum.response.created_count", "edx.forum.thread.created_count"]
    dfout['forum_created_count'] = dfout[createcolumns].sum(axis=1)
    dfout = dfout.drop(createcolumns, axis=1)
    
    #combine votes on forum
    votecolumns = ["edx.forum.response.voted_count", "edx.forum.thread.voted_count"]
    dfout['forum_vote_count'] = dfout[votecolumns].sum(axis=1)
    dfout = dfout.drop(votecolumns, axis=1)
    
    return dfout


def merge_features_label(df_features, df_labels):
    """
    Merge the features dataframe with the label to predict, by joining on the STUDENT_ID_KEY
    """
    df_labels2 = df_labels[[STUDENT_ID_KEY, LABEL_KEY]]
    return pd.merge(df_features, df_labels2, on = STUDENT_ID_KEY, how ='inner')


def merge_features_demo(df_features, df_demo):
    """
    Merge the features dataframe with the demographics' one to predict, by joining on the STUDENT_ID_KEY
    """
    return pd.merge(df_features, df_demo, on = STUDENT_ID_KEY, how ='inner')


def group_features_by_week(df):
    """
    Take as input a dataframe with the features for all weeks, and group the features by week, using the groupby method of pandas.
    """
    return df.groupby(df[WEEK_ID_KEY])


def create_k_folds(df):
    """
    Create the train and test sets for k folds cross-validation, and return a dict with the weeks as the key and the index of the cv folds as the value.
    User the StratifiedKFold method of sklearn to create the folds and get the index of the sample within each folds
    """
    #levels_train_sets = {}
    #levels_test_sets = {}
    #level_train_set_index = []
    #level_test_set_index = []
    cv_split_indices = []

    #for week, week_features in dfweeks:
    cv = StratifiedKFold(n_splits = NBFOLDS, shuffle=True, random_state = RAND_STATE_SEED)
    
    for train_set_index, test_set_index in cv.split(df, y = df[LABEL_KEY]):
        cv_split_indices.append( (train_set_index, test_set_index) )
    return cv_split_indices


def feature_label_split(dataset):
    """ Split the features and label for the  MOOC data. Also remove irrelevant columns."""
    dataset_features = dataset.drop(columns=[LABEL_KEY,STUDENT_ID_KEY,WEEK_ID_KEY])
    dataset_labels = dataset[LABEL_KEY].copy()
    return (dataset_features, dataset_labels)  


def th_scoring(th, y, prob):
    pred = (prob > th).astype(int)
    return 0 if not pred.any() else -fbeta_score(y, pred, beta=0.1) 


def init_classifiers():
    #Create the final list of all classifiers
    kwargs = {"nb_inner_folds": NBFOLDS_GRID, "labels": LABEL_VALUES, "verbose": VERBOSE, "seed": RAND_STATE_SEED}
    classifiers = {
        "Base1": StratifiedBaselineClassifier(**kwargs),
        "Base2": MajorityClassBaselineClassifier(**kwargs),
        "NB": NaiveBayesClassifier(CORREL_THRESHOLD, GRIDSEARCH, **kwargs),
        "KNN": KNeirestNeighboorsClassifier(CORREL_THRESHOLD, GRIDSEARCH, **kwargs),
        "LR": LogisticClassifier(CORREL_THRESHOLD, GRIDSEARCH, **kwargs),
        "SVM": SVMClassifier(CORREL_THRESHOLD, GRIDSEARCH, **kwargs),
        "SGB": StochasticGradientBoostingAlgorithmClassifier(CORREL_THRESHOLD, GRIDSEARCH, **kwargs),
        "RF": RandomForestEnsembleClassifier(CORREL_THRESHOLD, GRIDSEARCH, **kwargs),
        "MLP": MultiLayerPerceptronClassifier(CORREL_THRESHOLD, GRIDSEARCH, **kwargs)
    }
    return classifiers


def results_classifiers(week, k, clf, hyperparam, test_set_labels, predictions, predictions2, nb_features_in, res_madd, res_abroca, res_groupf1):
    d = OrderedDict( {
        "week": week,
        "K": k,
        "classifier": clf,
        "grid_params": hyperparam,
        "precision": precision_score(test_set_labels, predictions, average='weighted'),
        "recall": recall_score(test_set_labels, predictions, average='weighted'),
        "f1_score": f1_score(test_set_labels, predictions, average='weighted'),
        "f1_score_th": f1_score(test_set_labels, predictions2, average='weighted'),
        "features_in": nb_features_in
    })
    #add madd values
    for i in range(len(SENSITIVE_ATTRIBUTES)):
        if i < len(res_madd):
            d["MADD_"+SENSITIVE_ATTRIBUTES[i]] = res_madd[i]
    #add abroca values
    for i in range(len(SENSITIVE_ATTRIBUTES)):
        if i < len(res_abroca):
            d["ABROCA_"+SENSITIVE_ATTRIBUTES[i]] = res_abroca[i]
    #add groupf1 values
    for i in range(len(SENSITIVE_ATTRIBUTES)):
        if i < len(res_groupf1):
            d["GROUPF1_"+SENSITIVE_ATTRIBUTES[i]+"_G0"] = res_groupf1[i][0]
            d["GROUPF1_"+SENSITIVE_ATTRIBUTES[i]+"_G1"] = res_groupf1[i][1]
    return d


def train_models(dfweeks, classifiers, sensitive_features = []):
    """
    Train a set of classifiers on the provided MOOC  data.
    dfweeks: a Panda dataframe built with the group_features_by_week() function
    classifiers: a list of classifiers initialized with the init_classifiers() function
    returns: a dataframe with the prediction performance of the trained classifiers, whose rows are formatted with the results_classifiers() function
    """
    levels_results_grid = {}
            
    fold_results = []
    pred_results = {}
    pred_proba = {}
    pred_labels = {}
    grid_classifier = {}

    # Iterate over the game level, feature sets and fold to train the models
    for week, df_week_features in dfweeks:
        print(f"------------------ Week {week} ------------------")
        
        pred_results[week] = {}
        pred_proba[week] = {}
        pred_labels[week] = {}
        grid_classifier[week] = {}
        cv_split_indices = create_k_folds(df_week_features) #create the k folds (outer loop)

        for k in range(NBFOLDS):
            print(f"------------------ K {k} ------------------")
            
            (train_set_index, test_set_index) = cv_split_indices[k]
            train_set = df_week_features.iloc[train_set_index]
            test_set = df_week_features.iloc[test_set_index]
            
            if VERBOSE:
                print ("Label distribution:")
                print("-Train", train_set[LABEL_KEY].value_counts())
                print("-Test", test_set[LABEL_KEY].value_counts())
                
                
            train_set_features, train_set_labels  = feature_label_split(train_set)
            test_set_features, test_set_labels  = feature_label_split(test_set)

            if SMOTE_MINORITY: # Oversample minority classes, in the train sets ONLY
                smoter = SMOTE(random_state = RAND_STATE_SEED)
                (train_set_features, train_set_labels) = smoter.fit_resample(train_set_features, train_set_labels)
                
            for classifier_name, classifier in classifiers.items():
                if classifier_name not in pred_results[week]: pred_results[week][classifier_name] = [] 
                if classifier_name not in pred_proba[week]: pred_proba[week][classifier_name] = [] 
                if classifier_name not in pred_labels[week]: pred_labels[week][classifier_name] = []
                if classifier_name not in grid_classifier[week]: grid_classifier[week][classifier_name] = []
                
                if VERBOSE:
                    print(train_set_features.info())
                    print(test_set_features.info())
                    print(train_set_labels.shape)
                    print(test_set_labels.shape)
                

                if "Base" in classifier_name: # No grid search for the baselines
                    grid_predictions, grid_proba = classifier.fit_predict(train_set_features, train_set_labels, test_set_features)
                    res = results_classifiers(week, k, classifier_name, "", test_set_labels, grid_predictions, grid_predictions, 0, [], [], [])
                    
                    if VERBOSE:
                        print("Baseline res", res)
                    
                    fold_results.append(res)
                    pred_results[week][classifier_name] += list(grid_predictions)
                    pred_proba[week][classifier_name] += list(grid_proba)
                    pred_labels[week][classifier_name] += list(test_set_labels)  
                else:
                    #grid_predictions, grid_proba = classifier.grid_fit_predict(train_set_features.loc[:, ~train_set_features.columns.isin(sensitive_features)], train_set_labels, test_set_features.loc[:, ~train_set_features.columns.isin(sensitive_features)])
                    grid_predictions, grid_proba = classifier.grid_fit_predict(train_set_features, train_set_labels, test_set_features)
                    nb_used_features = classifier.grid.best_estimator_['data_scaler'].n_features_in_
                    
                    #opti threshold
                    #train_proba = classifier.grid.best_estimator_.predict_proba(train_set_features.loc[:, ~train_set_features.columns.isin(sensitive_features)])
                    train_proba = classifier.grid.best_estimator_.predict_proba(train_set_features)
                    grid_predictions2 = classifier.classification_threshold_optimizer(train_proba, train_set_labels, grid_proba)

                    #fairness eval
                    res_madd = []
                    res_abroca = []
                    res_groupf1 = []
                    df_madd = test_set_features.copy(deep=True).reset_index() #need to reset index for the MADD to work
                    df_madd["pred_proba"] = grid_proba[:, 1] #for abroca
                    df_madd["final_result"] = test_set_labels.to_numpy() #for abroca

                    for sattr in sensitive_features:
                        #MADD
                        res_madd.append(maddlib.MADD(h='auto', X_test=df_madd, pred_proba=grid_proba[:, 1], sf=sattr, model=week+"_"+classifier_name))

                        #ABROCA
                        g1 = 1
                        g0 = 0
                        if df_madd[sattr].value_counts().loc[[g1]].values[0] > df_madd[sattr].value_counts().loc[[g0]].values[0]:  # select by index value (class 0 or class 1)
                            majority_group = 1
                        else:
                            majority_group = 0
                        res_abroca.append(abroca.compute_abroca(
                                                       df=df_madd,
                                                       pred_col='pred_proba',
                                                       label_col='final_result',
                                                       protected_attr_col=sattr,
                                                       majority_protected_attr_val=majority_group,
                                                       n_grid=10000,
                                                       plot_slices=False
                                                       ))
                    
                        #group F1
                        res_groupf1.append(fair_performances.group_performances(X_test=df_madd, Y_test=df_madd["final_result"], pred=grid_predictions, sf=sattr))
                    del df_madd

                    """thresholds = np.linspace(0,1, 100)[1:-1]
                    scores = [th_scoring(th, train_set_labels, classifier.grid.predict_proba(train_set_features)[:,1]) for th in thresholds]
                    newth = thresholds[np.argmin( np.min(scores) )]
                    grid_predictions2 = [1 if x >= newth else 0 for x in grid_proba[:, 1]]
                    """
                    res = results_classifiers(week, k, classifier_name, str(classifier.grid.best_params_), test_set_labels, grid_predictions, grid_predictions2, nb_used_features, res_madd, res_abroca, res_groupf1)
                    
                    fold_results.append(res)
                    grid_classifier[week][classifier_name].append(deepcopy(classifier.grid.best_estimator_))
                    pred_results[week][classifier_name] += list(grid_predictions)
                    pred_proba[week][classifier_name] += list(grid_proba)
                    pred_labels[week][classifier_name] += list(test_set_labels)
                    
                    if VERBOSE: #prints for debug:
                        print(classifier_name, "res", res)
                        print(classifier.grid.best_estimator_)
                        print(classifier.grid.best_params_)
                        print(classifier.grid.best_estimator_["features_correlated"].to_keep)
                        print(classifier.grid.best_estimator_["features_correlated"].to_drop)           

    print("Model training done.")
    return pd.DataFrame.from_records(fold_results) 




#############################################################################################################################
#Main program
#############################################################################################################################

#open the data
if FEATURES_CUMUL: #cumulative features
    df_features = pd.read_csv("./features/df_features_cumulative2.csv")
else: #weekly features
    df_features = pd.read_csv("./features/df_features2.csv")
df_labels = pd.read_csv("./features/df_labels.csv")
df_demo = pd.read_csv("./features/df_demographics.csv")

#clean-up data
df_features_clean = clean_up_features(df_features)
df_ml = merge_features_demo(df_features_clean, df_demo)
df_ml = merge_features_label(df_ml, df_labels)
grouped_features = group_features_by_week(df_ml)


if VERBOSE:
    print(df_labels.info())
    print(df_labels[LABEL_KEY].value_counts())
    print(grouped_features)
    for week, df_week_features in grouped_features:
        print(f"------------------ Week {week} ------------------")
        print(df_week_features.info())
        print(df_week_features.describe())
        print(df_week_features.head())
        print(df_week_features[LABEL_KEY].value_counts())

#train classifiers
classifiers = init_classifiers()
df_results = train_models(grouped_features, classifiers, SENSITIVE_ATTRIBUTES)

if VERBOSE:
    print(f"------------------ Results ------------------")
    print(df_results.info())
    print(df_results.describe())
    print(df_results.head())

#output results
output_name = "dfres_" + ("cumul_" if FEATURES_CUMUL else "sliding_") + str(NBFOLDS)+"folds_" + ("grid_" if GRIDSEARCH else "nogrid_") + ("smote_" if SMOTE_MINORITY else "nosmote_") + str(CORREL_THRESHOLD)+"correl_" + str(RAND_STATE_SEED)+"seed"+EXPORT_SUFFIX
df_results.to_pickle(f"./output/{output_name}.pkl")
df_results.to_csv(f"./output/{output_name}.csv", index=False)
print("Done.")