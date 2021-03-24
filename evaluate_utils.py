# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:31:42 2021

@author: sergey feldman
"""
from time import time
from copy import deepcopy
import numpy as np
from constants import N_JOBS, RANDOM_STATE, N_SPLITS, N_BAYESSEARCH_ITER
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, ShuffleSplit
from mlxtend.feature_selection import ColumnSelector
from sklearn.model_selection import GridSearchCV
from ml_models import meta_pipeline, meta_grid
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.linear_model import LogisticRegression
from skopt import BayesSearchCV
from skopt.callbacks import DeltaYStopper, DeltaXStopper


def get_cv(
    learning_task,
    groups=False,
    n_splits=N_SPLITS,
    shuffle_split=False,
    shuffle_split_train_size=0.75,
    shuffle_split_test_size=0.25,
    random_state=RANDOM_STATE
):
    # this function creates the appropriate cross-validation splitter.
    # if a group is provided, it is used to stratify the splits
    # otherwise if it's a classification problem, the classes as used as the strata.
    # for regression there is no stratification
    # there is some wonkiness when you have exactly one split - that's a scikit learn issue
    assert n_splits >= 1
    if shuffle_split or n_splits == 1:
        if groups:
            cv = GroupShuffleSplit(n_splits=n_splits, train_size=shuffle_split_train_size,
                                   test_size=shuffle_split_train_size, random_state=random_state)
        elif learning_task in {"binary", "multiclass"}:
            cv = StratifiedShuffleSplit(n_splits=n_splits, train_size=shuffle_split_train_size,
                                        test_size=shuffle_split_test_size, random_state=random_state)
        else:
            cv = ShuffleSplit(n_splits=n_splits, train_size=shuffle_split_train_size,
                              test_size=shuffle_split_test_size, random_state=random_state)
    else:
        if groups:
            cv = GroupKFold(n_splits)
        elif learning_task in {"binary", "multiclass"}:
            cv = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
        else:
            cv = KFold(n_splits, shuffle=True, random_state=random_state)
    return cv


def prepend_column_subset(pipeline, covariate_names):
    # note: this will fall apart if the input is a naked classifier/regressor and not a pipeline
    # but that shouldn't happen anywhere in this repo
    steps = [('column_subset', ColumnSelector(cols=covariate_names))] + deepcopy(pipeline.steps)
    return Pipeline(steps)


def get_default_scoring(learning_task):
    # this function defines the "default" scoring
    # that will be used to evaluate ML models
    if learning_task == "binary":
        return "roc_auc"
    elif learning_task == "multiclass":
        return "roc_auc_ovr_weighted"
    else:
        return "neg_mean_absolute_error"


def stacked_model(df, y, pipeline, grid, input_covariates_list, names_of_covariate_groups, inner_cv, scoring, n_jobs=N_JOBS, verbose=False):
    # individual models to be stacked
    # have to fit them all first and figure out best hyper-params
    clfs = []
    for feat_name, feat_list in zip(names_of_covariate_groups, input_covariates_list):
        # if there's too few features, we can't take only 1% of them
        if len(feat_list) < 100 and "selector__percentile" in grid:
            grid["selector__percentile"] = [5, 10, 100]
        else:
            grid["selector__percentile"] = [1, 10, 100]

        # too much to search via gridsearch so using bayesian search
        if verbose:
            print(f"Fitting individual model for '{feat_name}' with {len(feat_list)} features...")
        start = time()
        # use bayesian search if there are too many hyperparameters settings in the grid
        if np.prod([len(i) for i in grid.values()]) > N_BAYESSEARCH_ITER:
            clf = BayesSearchCV(
                prepend_column_subset(pipeline, feat_list),
                grid,
                cv=inner_cv,
                scoring=scoring,
                n_jobs=n_jobs,
                n_iter=N_BAYESSEARCH_ITER,
                n_points=2,  # this works with the default n_jobs=8 and n_cv=4
                refit=True
            )
            clf.fit(df, y, callback=[DeltaXStopper(1e-8), DeltaYStopper(0.001, n_best=3)])
        else:
            clf = GridSearchCV(
                prepend_column_subset(pipeline, feat_list),
                grid,
                cv=inner_cv,
                scoring=scoring,
                n_jobs=n_jobs,
                refit=True
            )
            clf.fit(df, y)
        clfs.append((feat_name, clf.best_estimator_))
        model_name, model = clf.best_estimator_.steps[-1]

        if verbose:
            print(f'Done in {time() - start} seconds.')
            if model_name in {'svc', 'logistic'}:
                if np.all(model.coef_ == 0):
                    print(f'All zeros for {model}, {feat_name}, {y.name}')

    # the stacker itself
    # meta_clf = GridSearchCV(
    #     estimator=meta_pipeline,
    #     param_grid=meta_grid,
    #     cv=inner_cv,
    #     scoring=scoring,
    #     n_jobs=n_jobs,
    #     refit=True,
    # )
    meta_clf = LogisticRegression(max_iter=10000, solver='liblinear', random_state=RANDOM_STATE)

    # the stacked classifier
    clf_stacked = StackingClassifier(
        estimators=clfs,
        final_estimator=meta_clf,
        cv=inner_cv,
        n_jobs=n_jobs,
        verbose=0
    )

    # train the entire model hierarchy
    clf_stacked.fit(df, y)
    clfs.append(('stacked', clf_stacked))
    return clfs


def sklearn_pipeline_evaluator(
    df,
    output_covariate,
    input_covariates_list,
    names_of_covariate_groups,
    pipeline,
    grid,
    groups=None,  # TODO: support groups later if we need to
    outer_cv=None,
    learning_task="binary",
    scoring=None,
    random_state=RANDOM_STATE,
    n_splits=N_SPLITS,
    n_jobs=N_JOBS,
    verbose=False
):
    if scoring == None:
        scoring = get_default_scoring(learning_task)

    scorer = get_scorer(scoring)

    # see here for learning metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
    if outer_cv is None:
        outer_cv = get_cv(learning_task, groups is not None, n_splits=n_splits, random_state=random_state)

    nested_scores = []
    estimators = []
    for i, (train_inds, test_inds) in enumerate(outer_cv):
        start = time()
        if verbose:
            print(f"Working on the fold {i}...")
        df_train = df.iloc[train_inds, :]
        df_test = df.iloc[test_inds, :]

        # have to reuse the same inner split in multiple spots
        # and this is what you have to do in sklearn to actually reuse it
        inner_cv = get_cv(learning_task, groups is not None, n_splits=n_splits, random_state=random_state)
        if groups is None:
            inner_cv = inner_cv.split(df_train, df_train[output_covariate])
        else:
            inner_cv = inner_cv.split(df_train, df_train[output_covariate], groups=groups[train_inds])
        inner_cv = list(inner_cv)

        clfs = stacked_model(
            df_train,
            df_train[output_covariate],
            pipeline,
            grid,
            input_covariates_list,
            names_of_covariate_groups,
            inner_cv,
            scoring,
            n_jobs,
            verbose=False
        )
        estimators.append(clfs)
        individual_scores = [scorer(clf[1], df_test, df_test[output_covariate]) for clf in clfs]
        nested_scores.append(individual_scores)

        if verbose:
            print(f"Done in {time() - start} seconds.\n")

    return {"test_score": np.array(nested_scores), "estimator": estimators}
