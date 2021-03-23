# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:44:10 2021

@author: serge
"""


from constants import RANDOM_STATE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier


"""
Classification models with feature selection
"""

# Linear SVC classifier
svc_selector_pipeline = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("imputer", SimpleImputer(add_indicator=True)),
        ("selector", SelectPercentile(mutual_info_classif, percentile=100)),
        (
            "svc",
            SVC(kernel="linear", class_weight="balanced", probability=True, tol=1e-4, random_state=RANDOM_STATE),
        ),
    ]
)
svc_selector_grid = {
    "svc__C": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4],
    "imputer__strategy": ["mean", "median", "most_frequent"],
    "selector__percentile": [1, 10, 100]
}

# logistic regression classifier
lr_selector_pipeline = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("imputer", SimpleImputer(add_indicator=True)),
        ("selector", SelectPercentile(mutual_info_classif, percentile=100)),
        (
            "logistic",
            LogisticRegression(
                solver="saga", max_iter=10000, penalty="elasticnet", class_weight='balanced', random_state=RANDOM_STATE
            ),  # without elasticnet penalty, LR can get awful performance
        ),
    ]
)
lr_selector_grid = {
    "logistic__C": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4],
    "imputer__strategy": ["mean", "median", "most_frequent"],
    "logistic__l1_ratio": [0.1, 0.25, 0.5, 0.75, 0.9],
    "selector__percentile": [1, 10, 100]
}

# random forest
rf_selector_pipeline = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("imputer", SimpleImputer(add_indicator=True)),
        ("selector", SelectPercentile(mutual_info_classif, percentile=100)),
        ("rf", RandomForestClassifier(random_state=RANDOM_STATE))
    ]
)


rf_selector_grid = {
    "rf__max_depth": [1, 2, 4, 8, 16, None],
    "imputer__strategy": ["mean", "median", "most_frequent"],
    "rf__max_features": [0.1, 0.25, 0.5, 0.75, 0.9],
    "selector__percentile": [1, 10, 50, 100]
}

# QDA
qda_selector_pipeline = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("imputer", SimpleImputer(add_indicator=True)),
        ("selector", SelectPercentile(mutual_info_classif, percentile=100)),
        ("qda", QuadraticDiscriminantAnalysis())
    ]
)


qda_selector_grid = {
    "qda__reg_param": [1e-4, 1e-3, 1e-2, 1e-1, 0.5],
    "imputer__strategy": ["mean", "median", "most_frequent"],
    "selector__percentile": [1, 10, 50, 100]
}

# GPR
gpr_selector_pipeline = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("imputer", SimpleImputer(add_indicator=True)),
        ("selector", SelectPercentile(mutual_info_classif, percentile=100)),
        ("gpr", GaussianProcessClassifier())
    ]
)


gpr_selector_grid = {
    "imputer__strategy": ["mean", "median", "most_frequent"],
    "selector__percentile": [1, 10, 50, 100]
}

"""
Classification models without feature selection for the stacker phase
note: currently not being used. only LR is used for simplicity.
"""
svc = SVC(kernel="linear", class_weight="balanced", probability=True, tol=1e-4, random_state=RANDOM_STATE)

lr = LogisticRegression(max_iter=10000, solver='liblinear', random_state=RANDOM_STATE)

# Create a pipeline
meta_pipeline = Pipeline([('classifier', lr)])

# Create space of candidate learning algorithms and their hyperparameters
meta_grid = [{'classifier': [svc, lr]}]
