# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 14:39:19 2021

@author: sergey feldman
"""
from multiprocessing import freeze_support
import pickle
import numpy as np
import pandas as pd
from evaluate_utils import sklearn_pipeline_evaluator, get_cv
from ml_models import svc_selector_pipeline, svc_selector_grid
from ml_models import lr_selector_pipeline, lr_selector_grid
from ml_models import qda_selector_pipeline, qda_selector_grid
from ml_models import gpr_selector_pipeline, gpr_selector_grid
from constants import RANDOM_STATE, N_SPLITS
import warnings

warnings.filterwarnings("ignore", message="The objective has been evaluated at this point before.")
warnings.simplefilter('always', category=UserWarning)

N_SPLITS_OUTER = 25
SHUFFLE_SPLIT = True


def drop_the_right_rows(df, output_covariate, covariates_to_check):
    # first drop the rows where there is no output
    df = df.copy()
    df = df.loc[~pd.isnull(df[output_covariate]), :]
    # now drop rows where all of the fnirs are missing
    covariate_missing_flag = df[covariates_to_check].isnull().all(1)
    df = df.loc[~covariate_missing_flag, :]
    return df


def run_all_models(
    df,
    output_covariate,
    input_covariates_list,
    names_of_covariate_groups,
    groups=None,
):
    # some kids don't have fnirs, so we have to drop them
    df = drop_the_right_rows(
        df,
        output_covariate,
        covariates_to_check=input_covariates_list[names_of_covariate_groups.index('raw_fnirs')]
    )

    # define cross-validation splits ahead of time so we can reuse it and save it
    # currently the "groups" variable isn't used but it is useful for stratified splitting
    # in case, for example, that you have two rows per subject and you want to make sure
    # all the rows for a subject are either in train OR val OR test but not split up among them
    # todo: enable shuffle_split here and do way more splits for the final run
    outer_cv = get_cv("binary", groups=groups is not None, n_splits=N_SPLITS_OUTER,
                      shuffle_split=SHUFFLE_SPLIT, random_state=RANDOM_STATE)
    if groups is None:
        outer_cv = outer_cv.split(df, df[output_covariate])
    else:
        outer_cv = outer_cv.split(df, df[output_covariate], groups=groups)
    outer_cv = list(outer_cv)

    results = {}

    results["Logistic Regression"] = sklearn_pipeline_evaluator(
        df,
        output_covariate,
        input_covariates_list,
        names_of_covariate_groups,
        lr_selector_pipeline,
        lr_selector_grid,
        groups=None,
        outer_cv=outer_cv,
        random_state=RANDOM_STATE,
        n_splits=N_SPLITS,
    )

    results["Support Vector Machine"] = sklearn_pipeline_evaluator(
        df,
        output_covariate,
        input_covariates_list,
        names_of_covariate_groups,
        svc_selector_pipeline,
        svc_selector_grid,
        groups=None,
        outer_cv=outer_cv,
        random_state=RANDOM_STATE,
        n_splits=N_SPLITS,
    )

    results["QDA"] = sklearn_pipeline_evaluator(
        df,
        output_covariate,
        input_covariates_list,
        names_of_covariate_groups,
        qda_selector_pipeline,
        qda_selector_grid,
        groups=None,
        outer_cv=outer_cv,
        random_state=RANDOM_STATE,
        n_splits=N_SPLITS,
    )

    results["GPR"] = sklearn_pipeline_evaluator(
        df,
        output_covariate,
        input_covariates_list,
        names_of_covariate_groups,
        gpr_selector_pipeline,
        gpr_selector_grid,
        groups=None,
        outer_cv=outer_cv,
        random_state=RANDOM_STATE,
        n_splits=N_SPLITS,
    )

    # print the mean absolute error for each model
    df_displays = []
    print(f"{output_covariate} Mean Absolute Error:")
    print("--------------------------------------------------")
    for model_name, r in results.items():
        df_display = pd.DataFrame({
            'mean_auroc': np.round(np.mean(r["test_score"], axis=0), 2),
            'std_auroc': np.round(np.std(r["test_score"], axis=0), 2),
            'model_features': names_of_covariate_groups + ['stacked']
        }).set_index('model_features')
        df_display.index.rename(model_name, inplace=True)
        print(df_display, '\n')
        df_displays.append(df_display)
    print("\n\n")

    # return all the models and results
    return results, outer_cv, df_displays


if __name__ == "__main__":
    # this prevents some obscure windows bugs...
    freeze_support()

    # load data
    with open("data/processed_data.pickle", "rb") as f:
        (
            df_crypto,
            crypto_output_6_covariates,
            crypto_input_6_covariates,
            crypto_output_24_covariates,
            crypto_input_24_covariates,
            df_provide,
            provide_output_36_covariates,
            provide_input_36_covariates,
            provide_output_60_covariates,
            provide_input_60_covariates,
            names_of_covariate_groups
        ) = pickle.load(f)

    # for each output, we have to make sure that the
    # rows of the input are dropped in places with all NaNs
    # for the fNIRs data

    all_results = {}

    # 6m
    for output in crypto_output_6_covariates:
        all_results[output] = run_all_models(
            df_crypto,
            output,
            crypto_input_6_covariates,
            names_of_covariate_groups,
        )

    # 24m
    for output in crypto_output_24_covariates:
        all_results[output] = run_all_models(
            df_crypto,
            output,
            crypto_input_24_covariates,
            names_of_covariate_groups,
        )

    # 36m
    for output in provide_output_36_covariates:
        all_results[output] = run_all_models(
            df_provide,
            output,
            provide_input_36_covariates,
            names_of_covariate_groups,
        )

    # 60m
    for output in provide_output_60_covariates:
        all_results[output] = run_all_models(
            df_provide,
            output,
            provide_input_60_covariates,
            names_of_covariate_groups,
        )

    with open("data/ml_results.pickle", 'wb') as f:
        pickle.dump(all_results, f)
