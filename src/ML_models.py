"""
AI fairness analysis code
@licence BSD-3-Clause 
"""

import pandas as pd
import numpy as np

import os
import sys
import warnings

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
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE 
from copy import deepcopy

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# adding src to the system path to find scripts in there
sys.path.insert(0, "./modules")
from threshold_optimizer import ThresholdOptimizer

DEFAULT_SEED = 42

############################################################################################################################################################
#PREPROCESSING CLASSES 
############################################################################################################################################################
class DataFrameMinMaxScaler(BaseEstimator, TransformerMixin):
    """ Custom transformer for scicit-learn, adapts MinMaxScaler() for panda dataframe"""
    def __init__(self): # no *args or ** kargs
         self.scaler = MinMaxScaler()
         self.n_features_in_ = 0
            
    def fit(self, X, y = None):
        return self
            
    def transform(self, X, y = None):
        modified_X = self.scaler.fit_transform(X)
        modified_X= pd.DataFrame(modified_X)
        self.n_features_in_ = len(modified_X.columns)
        return modified_X


class DeleteCorrelatedFeatures(BaseEstimator, TransformerMixin):
    """ Custom transformer for scicit-learn, meant to delete highly correlated features in a dataframe"""
    default_threshold = 0.8
    
    def __init__(self, threshold = default_threshold): # no *args or ** kargs
        self.threshold = threshold
        self.to_drop = []
        self.to_keep = []
        
    def fit(self, X, y = None):
        if isinstance(X, pd.DataFrame):
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            self.to_drop = [column for column in upper.columns if any(upper[column] >= self.threshold)]
        else:
            corr_matrix = np.absolute(np.corrcoef(X, rowvar=False))
            upper = corr_matrix*np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            self.to_drop = [column for column in range(upper.shape[1]) if any(upper[:,column] >= self.threshold)]
        return self
    
    def transform(self, X, y = None):
        if isinstance(X, pd.DataFrame):
            X_reduced = X.drop(columns = self.to_drop)
            self.to_keep = X_reduced.columns
        else:
            X_reduced = np.delete(X, self.to_drop, axis=1)
        return X_reduced
        

class DeleteIrrelevantFeatures(BaseEstimator, TransformerMixin):
    """ Custom transformer for scicit-learn, to delete irrelevant features (all values are the same, no variance)"""
    def __init__(self, threshold_variance = 1): # no *args or ** kargs
         self.threshold_variance = threshold_variance
            
    def fit(self, X, y = None):
        self.unused_features = []
        for key in X.columns:
            nb_values = X[key].value_counts()
            if len(nb_values) == self.threshold_variance: # If all the values are the same
                self.unused_features.append(key)
        return self
        
    def transform(self, X, y = None):
        # print("Number of deleted features",len(self.unused_features))
        modified_X = X.drop(columns=self.unused_features)    
        return modified_X


############################################################################################################################################################
#ML MODEL SUPER CLASS 
############################################################################################################################################################

class ML_Model:
    """
    Genetric superclass for ML classifiers in scitcit-learn
    """
    def __init__(self, pipeline, hyperparameter_grid, correl_threshold=0.8, grid_search=False, nb_inner_folds=5, labels = [0, 1], verbose=False, seed=DEFAULT_SEED):
        """
        Init 3 attributes: a sklearn pipeline (self.pipeline), a sklearn GridSearchCV process (self.grid), and a grid for the hyper-parameter tuning in sklearn format (self.hyperparameter_grid)
        """
        self.pipeline = pipeline
        self.hyperparameter_grid = hyperparameter_grid
        self.correl_threshold = correl_threshold
        self.grid_search = grid_search
        self.labels_values = labels
        self.nb_inner_folds = nb_inner_folds
        self.seed = seed
        self.verbose = verbose
        self.init_grid_search()
        
    def init_grid_search(self, scoring ='f1_weighted'):
        """
        Initialize the GridSearchCV object. see GridSearchCV() documentation in sklearn for the scoring.
        """
        stratified_inner_cross_val = StratifiedKFold(n_splits = self.nb_inner_folds, shuffle = True, random_state = self.seed)
        self.grid = GridSearchCV( 
            estimator = self.pipeline,
            param_grid = self.hyperparameter_grid,
            cv = stratified_inner_cross_val,
            verbose = 1,
            scoring ='f1_weighted',
            n_jobs=-1)
            
    def fit_predict(self, train_set_features, train_set_labels, test_set_features):
        """
        Fit the classifier on the train set and predict on the test set. Return two values: the predictions and the probabilities outputed by the classifiers.
        """
        with warnings.catch_warnings(record=True) as w:
            self.pipeline.fit(train_set_features, train_set_labels)
            predictions = self.pipeline.predict(test_set_features)
            proba = self.pipeline.predict_proba(test_set_features)
            
            if w is not None and len(w) > 0:
                print(print(w[-1].category))
                if self.verbose:
                    print(w[-1].message)
                    
        return predictions, proba
                    
    def grid_fit_predict(self, train_set_features, train_set_labels, test_set_features):
        """
        Fit the classifier on the train set using the grid hyperparameter tuning method, and predict on the test set. Return two values: the predictions and the probabilities outputed by the classifiers.
        """
        with warnings.catch_warnings(record=True) as w:
            self.grid.fit(train_set_features, y = train_set_labels)
            grid_predictions = self.grid.predict(test_set_features)
            grid_proba = self.grid.predict_proba(test_set_features)
            
            if w is not None and len(w) > 0:
                print(print(w[-1].category))
                if self.verbose:
                    print(w[-1].message)

        return grid_predictions, grid_proba
        
    def classification_threshold_optimizer(self, pred_proba_train, labels_train, pred_proba_test):
        """
        Optimize the classification threshold using the threshold_optimizer package on the train set, and return the threshold along with the new predictions on the test set.
        """
        # init optimization
        thresh_opt = ThresholdOptimizer(y_score = pred_proba_train, y_true = labels_train)

        # optimize for f1 score
        thresh_opt.optimize_metrics(metrics=['f1'], verbose=False)
        f1_threshold = thresh_opt.optimized_metrics['f1_score']['best_threshold']
        
        # use best accuracy threshold for test set to convert probabilities to classes
        return np.where(pred_proba_test[:,1] > f1_threshold, self.labels_values[1], self.labels_values[0])



############################################################################################################################################################
#ML MODELS SUBCLASSES
############################################################################################################################################################

# Subclasses for specific baselines and classifiers
class StratifiedBaselineClassifier(ML_Model):
    def __init__(self, **kwargs):
        #Dummy baseline from sklearn
        dummy_random_clf_pipe = Pipeline([
            ("dummy_random_clf", DummyClassifier(strategy="stratified"))
        ])
        super().__init__(dummy_random_clf_pipe, None, **kwargs)

class MajorityClassBaselineClassifier(ML_Model):
    def __init__(self, **kwargs):
        #Dummy baseline from sklearn
        dummy_random_clf_pipe = Pipeline([
            ("dummy_random_clf", DummyClassifier(strategy="most_frequent"))
        ])
        super().__init__(dummy_random_clf_pipe, None, **kwargs)

class NaiveBayesClassifier(ML_Model):
    def __init__(self, correl_threshold=0.8, grid_search=False, **kwargs):
        # Multinomial Naive Bayes classifier
        naive_bayes_clf_pipe = Pipeline([
            #("time_spend_discretizer",TimeSpendDiscretizer()),
            ("feature_deleter",DeleteIrrelevantFeatures()),
            ("features_correlated",DeleteCorrelatedFeatures(correl_threshold)),
            ("data_scaler", DataFrameMinMaxScaler()),
            ("naive_bayes_clf", GaussianNB())
        ])
        
        param_grid_nb = [{
                'naive_bayes_clf__var_smoothing': [1e-9] #default
            }]
        if grid_search:
            param_grid_nb = [{
                # Default value in sklearn: 1.0
                'naive_bayes_clf__var_smoothing': [1e-7, 1e-8, 1e-9, 1e-10, 1e-11] #large grid
                #'naive_bayes_clf__alpha': [10.0, 1.0, 0.1] #small grid
            }]
            
        super().__init__(naive_bayes_clf_pipe, param_grid_nb, correl_threshold, grid_search, **kwargs)

class KNeirestNeighboorsClassifier(ML_Model):
    def __init__(self, correl_threshold=0.8, grid_search=False, **kwargs):
        # Multinomial Naive Bayes classifier
        knn_clf_pipe = Pipeline([
            #("time_spend_discretizer",TimeSpendDiscretizer()),
            ("feature_deleter",DeleteIrrelevantFeatures()),
            ("features_correlated",DeleteCorrelatedFeatures(correl_threshold)),
            ("data_scaler", DataFrameMinMaxScaler()),
            ("k_neirest_neighboors_clf", KNeighborsClassifier())
        ])
        
        param_grid_nb = [{
                'k_neirest_neighboors_clf__n_neighbors': [5] #default
            }]
        if grid_search:
            param_grid_nb = [{
                # Default value in sklearn: 1.0
                'k_neirest_neighboors_clf__n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10, 15, 20] #large grid
            }]
            
        super().__init__(knn_clf_pipe, param_grid_nb, correl_threshold, grid_search, **kwargs)
        
class LogisticClassifier(ML_Model):
    def __init__(self, correl_threshold=0.8, grid_search=False, **kwargs):
        # Logistic regression
        logistic_regression_clf_pipe = Pipeline([
            ("feature_deleter",DeleteIrrelevantFeatures()),
            ("features_correlated",DeleteCorrelatedFeatures(correl_threshold)),
            ("data_scaler", DataFrameMinMaxScaler()),
            ("logistic_regression_clf", LogisticRegression(max_iter=250, random_state=kwargs.get("seed", DEFAULT_SEED)))
        ])
        
        param_grid_lr = [{
            'logistic_regression_clf__C': [1.0], #default
            'logistic_regression_clf__max_iter': [250] #to converge
        }]
        if grid_search:
            param_grid_lr = [{
                # Default value in sklearn: 1.0
                'logistic_regression_clf__C': [100.0, 50.0, 25.0, 10.0, 5.0, 1.0, 0.5, 0.1], #large grid
                #'logistic_regression_clf__C': [10.0, 1.0, 0.1], #small grid
                'logistic_regression_clf__max_iter': [250] #to converge
            }]
        
        super().__init__(logistic_regression_clf_pipe, param_grid_lr, correl_threshold, grid_search, **kwargs)
        
class SVMClassifier(ML_Model):
    def __init__(self, correl_threshold=0.8, grid_search=False, **kwargs):    
        # SVM
        svm_clf_pipe = Pipeline([
            ("feature_deleter",DeleteIrrelevantFeatures()),
            ("features_correlated",DeleteCorrelatedFeatures(correl_threshold)),
            ("data_scaler", DataFrameMinMaxScaler()),
            ("svm_gaussian_clf", SVC(probability=True))
        ])
        
        param_grid_svm = [{
            'svm_gaussian_clf__C': [1.0], #default
            'svm_gaussian_clf__gamma': [1.0], #default
            'svm_gaussian_clf__kernel': ['rbf'] #default
        }]
        
        if grid_search:
            param_grid_svm = [{
                # Default value in sklearn: 1.0
                'svm_gaussian_clf__C': [100.0, 50.0, 25.0, 10.0, 5.0, 1.0, 0.5, 0.1], #large grid
                #'svm_gaussian_clf__C': [10.0, 1.0, 0.1], #small grid
                # Default value in sklearn: 1.0
                'svm_gaussian_clf__gamma': [10.0, 1.0, 0.5, 0.1, 0.01, 0.001, 0.0001], #large grid
                #'svm_gaussian_clf__gamma': [10.0, 1.0, 0.1], #small grid
                # Default value in sklearn: rbf
                'svm_gaussian_clf__kernel': ['rbf', 'poly', 'sigmoid'] #large+small grid
            }]

        super().__init__(svm_clf_pipe, param_grid_svm, correl_threshold, grid_search, **kwargs)



class RandomForestEnsembleClassifier(ML_Model):
    def __init__(self, correl_threshold=0.8, grid_search=False, **kwargs):   
        #Random Forests
        random_forest_clf_pipe = Pipeline([
            ("feature_deleter",DeleteIrrelevantFeatures()),#VarianceThreshold()),
            ("features_correlated",DeleteCorrelatedFeatures(correl_threshold)),
            ("data_scaler", DataFrameMinMaxScaler()),
            ("random_forest_clf", RandomForestClassifier(random_state = kwargs.get("seed", DEFAULT_SEED)))
        ])
        
        param_grid_rf = [{
            "random_forest_clf__n_estimators": [100], #default
            "random_forest_clf__max_depth" : [None],    #default
        }]
        if grid_search:
            param_grid_rf = [{
                # Default value in sklearn: 100
                "random_forest_clf__n_estimators": [20, 50, 100, 200, 300, 500], #large grid
                #"random_forest_clf__n_estimators": [50, 100, 200], #small grid
                # Default value in sklearn: None
                "random_forest_clf__max_depth" : [4,6,8,10,12,14,16,None],    #large grid
                #"random_forest_clf__max_depth" : [6,12,None],    #small grid
            }]
        
        super().__init__(random_forest_clf_pipe, param_grid_rf, correl_threshold, grid_search, **kwargs)


class StochasticGradientBoostingAlgorithmClassifier(ML_Model):
    def __init__(self, correl_threshold=0.8, grid_search=False, **kwargs):   
        #Random Forests
        gradient_boosting_clf_pipe = Pipeline([
            ("feature_deleter",DeleteIrrelevantFeatures()),#VarianceThreshold()),
            ("features_correlated",DeleteCorrelatedFeatures(correl_threshold)),
            ("data_scaler", DataFrameMinMaxScaler()),
            ("gradient_boosting_clf", GradientBoostingClassifier(n_estimators = 250, random_state = kwargs.get("seed", DEFAULT_SEED)))
        ])
        
        param_grid_sgb = [{
            "gradient_boosting_clf__learning_rate": [0.1], #default
            "gradient_boosting_clf__subsample" : [1.0],    #default
            "gradient_boosting_clf__max_depth" : [3],    #default
            "gradient_boosting_clf__min_samples_split" : [2],    #default
        }]
        if grid_search:
            param_grid_rf = [{
                "gradient_boosting_clf__learning_rate": [10.0, 5.0, 1.0, 0.1, 0.01, 0.001, 0.0], #large grid
                "gradient_boosting_clf__subsample" : [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],    #large grid
                "gradient_boosting_clf__max_depth" : [2, 3, 4, 5, 8, 10, 12, 14, 16, None],    #large grid
                "gradient_boosting_clf__min_samples_split" : [2, 4, 6, 8, 10],    #large grid
            }]
        
        super().__init__(gradient_boosting_clf_pipe, param_grid_sgb, correl_threshold, grid_search, **kwargs)


class MultiLayerPerceptronClassifier(ML_Model):
    def __init__(self, correl_threshold=0.8, grid_search=False, **kwargs): 
    #Multilayer Perceptron
        mlp_clf_pipe = Pipeline([
            ("feature_deleter",DeleteIrrelevantFeatures()),#VarianceThreshold()),
            ("features_correlated",DeleteCorrelatedFeatures(correl_threshold)),
            ("data_scaler", DataFrameMinMaxScaler()),
            ("mlp_clf", MLPClassifier(max_iter=100, random_state = kwargs.get("seed", DEFAULT_SEED)))
        ])   
    
        param_grid_mlp = [{
            "mlp_clf__hidden_layer_sizes": [(10,)], #default/2 small dataset
            "mlp_clf__activation" : ['logistic'],    #logistic sigmoid function
            "mlp_clf__solver" : ['lbfgs'],    #for small dataset
            "mlp_clf__alpha" : [0.0001],    #default
            "mlp_clf__learning_rate" : ['constant']    #default
        }]
        if grid_search:
            param_grid_mlp = [{
                "mlp_clf__hidden_layer_sizes": [(5,), (10,), (15,), (20,), (20,10,), (15, 8,), (10, 5,)], #large grid
                "mlp_clf__activation" : ['logistic', 'relu'],    #large grid
                "mlp_clf__solver" : ['lbfgs', 'adam'],    #large grid
                "mlp_clf__alpha" : [0.5, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.0],    #large grid
                "mlp_clf__learning_rate" : ['constant'],    #large grid
                "mlp_clf__max_iter" : [30, 50, 100, 250, 500],    #large grid
            }]
        
        super().__init__(mlp_clf_pipe, param_grid_mlp, correl_threshold, grid_search, **kwargs)
       