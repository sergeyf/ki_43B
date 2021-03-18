# -*- coding: utf-8 -*-
"""
Created on Tue Jan 5 13:40:03 2021

@author: sergey feldman
"""
import re
import pickle
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.linalg import svdvals


pd.options.mode.chained_assignment = None  # default='warn'

# import data from excel files
# note that all values of 99, 999, 9999 will be treated as missing
# some of the 9s are also missing for certain covariates, but these are left alone
# because many columns have "real" 9s and we don't want to NaN those
# another note: there are 2 datasets inside this data!
df_non_neural = pd.read_excel(
    "data/DataRequest_Sanchez-Alonso_12.10.20.xlsx",
    na_values=[99, "99", 999, "999", 9999, "9999", "999.99", 999.99, "999.9", 999.9]
)

# these are the only variables in our chosen list where a 9 means missing
df_non_neural.loc[df_non_neural['diar_1'] == 9, 'diar_1'] = np.nan
df_non_neural.loc[df_non_neural['diar_2'] == 9, 'diar_2'] = np.nan

# secondary data
df_anthro = pd.read_excel("data/DataRequest_Sanchez.Alonso_3.12.21_correctIDs.xlsx")
df_anthro.drop(['SEX'], axis=1, inplace=True)

# join the two
df_raw = pd.merge(left=df_non_neural, right=df_anthro, on='FSID', how='left')

# import train/test splits
test_train_crypto = pd.read_excel("data/BEAN_CRYPTO_testing_training_n130.xlsx")
test_train_provide = pd.read_excel("data/BEAN_PROVIDE_testing_training_n130.xlsx")

# this entire sprint will not touch the test data, so all we need is the training data
df_crypto = df_raw[df_raw["FSID"].isin(
    test_train_crypto.loc[test_train_crypto["Dataset"] == "Training", "ID"])]
df_provide = df_raw[df_raw["FSID"].isin(
    test_train_provide.loc[test_train_provide["Dataset"] == "Training", "ID"])]

# outputs are binary classification of bottom quartile vs not
for col in ["rlraw_6", "elraw_6", "rlraw_24", "elraw_24"]:
    df_crypto[col + '_top_75'] = df_crypto[col] <= np.nanpercentile(df_crypto[col], 25)
    df_crypto.loc[df_crypto[col + '_top_75'].isnull(), col + '_top_75'] = np.nan

for col in ["rlraw_36", "elraw_36", "vbrawsc", "comrawsc"]:
    df_provide[col + '_top_75'] = df_provide[col] <= np.nanpercentile(df_provide[col], 25)
    df_provide.loc[df_provide[col + '_top_75'].isnull(), col + '_top_75'] = np.nan


"""
HAZ for crypto:
AGEM_AN03 has the actual age of the child for the measurement stored in HAZ_AN01
we have to exclude HAZ values that happened after the MULLEN scores at 6m and 24m

for predicting 6m mullen, can use: HAZ_AN01 through HAZ_AN03 (HAZ_AN04 can happen at 6m or after)
for predicing 24m mullen, can use: HAZ_AN01 through HAZ_AN09 (HAZ_AN10 can happen at 24m or after)
caveat: HAZ_AN07 has missing values, so we're excluding that one
"""
haz_covariates_for_6m = [f"HAZ_AN0{i}" for i in [1, 2, 3]]
df_crypto["stunted_frac_up_to_6m"] = np.nanmean(df_crypto[haz_covariates_for_6m] < -2, axis=1)

haz_covariates_for_24m = [f"HAZ_AN0{i}" for i in [1, 2, 3, 4, 5, 6, 8, 9]]
df_crypto["stunted_frac_up_to_24m"] = np.nanmean(df_crypto[haz_covariates_for_24m] < -2, axis=1)


"""
HAZ for provide:
no need to compute age. haz_36 and lower will work for 36m and haz_60 and lower will work for the rest
note: there are nans, so have to do nanmean
"""
haz_covariates_for_36m = [f'haz_{i}' for i in list(range(1, 17)) + [30, 36]]
df_provide["stunted_frac_up_to_36m"] = np.nanmean(df_provide[haz_covariates_for_36m] < -2, axis=1)

haz_covariates_for_60m = [f'haz_{i}' for i in list(range(1, 17)) + [30, 36, 42, 48, 54, 60]]
df_provide["stunted_frac_up_to_60m"] = np.nanmean(df_provide[haz_covariates_for_60m] < -2, axis=1)


"""
baseline variables
"""
baseline_covariates = [
    "FSID",
    "SEX",
    "WT_1",
    "WT_2",
    "HT_1",
    "HT_2",
    "inco_1",
    "inco_2",
    "watr_1",
    "watr_2",
    "room_1",
    "room_2",
    "medu_1",
    "medu_2",
    "fedu_1",
    "fedu_2",
    "toil_1",
    "toil_2",
    "hhfd_1",
    "hhfd_2",
    "agfd_1",
    "agfd_2",
    "watrt_1",
    "watrt_2",
    "tshar_1",
    "tshar_2",
    "ckpl_1",
    "ckpl_2",
    "nail_1",
    "nail_2",
    "fuel_1",
    "fuel_2",
    "drain_1",
    "drain_2",
    "diar_1",
    "diar_2",
    'G_AGE',
    'mage',
    'mrno',
    'mrag',
    'mnag',
    'prgag',
]

crypto_covariates = [
    'WT',
    'HT',
    "stunted_frac_up_to_6m",
    "stunted_frac_up_to_24m",
    "rlraw_6_top_75",
    "elraw_6_top_75",
    "rlraw_24_top_75",
    "elraw_24_top_75",
]

provide_covariates = [
    'mwtkg',
    'mhtcm',
    "stunted_frac_up_to_36m",
    "stunted_frac_up_to_60m",
    "rlraw_36_top_75",
    "elraw_36_top_75",
    "vbrawsc_top_75",
    "comrawsc_top_75"
]

df_crypto = df_crypto[crypto_covariates + baseline_covariates].dropna(axis='columns', how='all')

df_provide = df_provide[provide_covariates + baseline_covariates].dropna(axis='columns', how='all')

# many variables are categorical and need to be converted to one-hot
categorical_covariates_crypto = [
    "room_1",
    "toil_1",
    "hhfd_1",
    "agfd_1",
    "watrt_1",
    "tshar_1",
    "ckpl_1",
    "nail_1",
    "fuel_1"
]

categorical_covariates_provide = [
    "room_1",
    "room_2",
    "toil_1",
    "toil_2",
    "hhfd_1",
    "hhfd_2",
    "agfd_1",  # agfd_2 is binary so excluded here
    "watrt_1",
    "watrt_2",
    "tshar_1",
    "ckpl_1",
    "ckpl_2",
    "nail_1",
    "nail_2",
    "fuel_1",
    "fuel_2",
    "diar_1",
    "diar_2",
]

df_crypto = pd.get_dummies(df_crypto, dummy_na=True, columns=categorical_covariates_crypto)
df_provide = pd.get_dummies(df_provide, dummy_na=True, columns=categorical_covariates_provide)

# can drop columns with no variance
df_crypto = df_crypto.loc[:, (df_crypto != df_crypto.iloc[0]).any()]
df_provide = df_provide.loc[:, (df_provide != df_provide.iloc[0]).any()]

# the column names sometimes have float trailing ".0" which are removed here
df_crypto.columns = [re.sub(r'\.0$', '', i) for i in df_crypto.columns]
df_provide.columns = [re.sub(r'\.0$', '', i) for i in df_provide.columns]

# summarize all of the baseline variables for crypto and provide
shared_crypto_baseline_covariates = [
    'SEX', 'WT_1', 'WT_2', 'HT_1', 'HT_2', 'inco_1', 'watr_1', 'medu_1', 'fedu_1',
    'drain_1', 'room_1_1', 'room_1_2', 'room_1_3', 'room_1_4', 'room_1_5',
    'toil_1_1', 'toil_1_2', 'toil_1_3', 'toil_1_4', 'hhfd_1_1', 'hhfd_1_2',
    'hhfd_1_3', 'hhfd_1_4', 'agfd_1_1', 'agfd_1_4', 'agfd_1_9', 'watrt_1_1',
    'watrt_1_5', 'watrt_1_6', 'tshar_1_1', 'tshar_1_2', 'ckpl_1_1',
    'ckpl_1_2', 'ckpl_1_3', 'nail_1_1', 'nail_1_2', 'nail_1_3', 'fuel_1_1',
    'fuel_1_2', 'fuel_1_3', 'fuel_1_4', 'G_AGE', 'mage', 'mrno', 'mrag', 'mnag',
    'prgag', 'WT', 'HT',
]

crypto_baseline_6_covariates = shared_crypto_baseline_covariates + ['stunted_frac_up_to_6m']
crypto_baseline_24_covariates = shared_crypto_baseline_covariates + ['stunted_frac_up_to_24m']

#
shared_provide_baseline_covariates = [
    'SEX', 'inco_1', 'inco_2', 'watr_1', 'watr_2', 'medu_1', 'medu_2',
    'fedu_1', 'fedu_2', 'agfd_2', 'tshar_2', 'drain_1', 'drain_2', 'room_1_1',
    'room_1_2', 'room_1_3', 'room_1_4', 'room_1_5', 'room_2_1', 'room_2_2',
    'room_2_3', 'room_2_5', 'toil_1_1', 'toil_1_2', 'toil_1_5', 'toil_2_1',
    'toil_2_2', 'toil_2_3', 'toil_2_4', 'toil_2_5', 'hhfd_1_1', 'hhfd_1_2',
    'hhfd_1_3', 'hhfd_1_4', 'hhfd_2_1', 'hhfd_2_2', 'hhfd_2_3', 'hhfd_2_4',
    'agfd_1_1', 'agfd_1_4', 'agfd_1_5', 'agfd_1_9', 'watrt_1_1',
    'watrt_1_3', 'watrt_1_5', 'watrt_1_7', 'watrt_2_1', 'watrt_2_3',
    'watrt_2_5', 'tshar_1_1', 'tshar_1_2', 'ckpl_1_1', 'ckpl_1_2',
    'ckpl_1_3', 'ckpl_2_1', 'ckpl_2_2', 'ckpl_2_3', 'nail_1_1', 'nail_1_2',
    'nail_1_3', 'nail_2_1', 'nail_2_2', 'nail_2_3', 'fuel_1_1', 'fuel_1_2',
    'fuel_1_3', 'fuel_2_1', 'fuel_2_2', 'fuel_2_3', 'diar_1_1', 'diar_1_2',
    'diar_1_nan', 'diar_2_1', 'diar_2_2', 'diar_2_nan', 'G_AGE', 'mage', 'mrno',
    'mrag', 'mnag', 'prgag', 'mwtkg', 'mhtcm',
]

provide_baseline_36_covariates = shared_provide_baseline_covariates + ['stunted_frac_up_to_36m']
provide_baseline_60_covariates = shared_provide_baseline_covariates + ['stunted_frac_up_to_60m']


"""
Quality control variables
"""


def qt_features(x):
    per_channel_mean = np.nanmean(x, 1)
    per_channel_std = np.nanstd(x, 1)
    overall_stats = [np.nanmean(x), np.nanmean(np.abs(x)), np.nanstd(x)]
    return list(np.hstack([per_channel_mean, per_channel_std, overall_stats]))


def get_qt_features(df_matlab, column_name_suffix=''):
    child_ids = [i[0][0][0].split('peekaboo')[0] for i in df_matlab['nirs_files']]

    qt_features_1 = [qt_features(i[0]) for i in df_matlab['quality_array'][0]]
    num_channels = (len(qt_features_1[0]) - 2) // 2
    prefix = 'qt_sci'
    column_names = [f'{prefix}_channel_{i}_mean_{column_name_suffix}' for i in range(1, num_channels + 1)]
    column_names += [f'{prefix}_channel_{i}_std_{column_name_suffix}' for i in range(1, num_channels + 1)]
    column_names += [
        f'{prefix}_overall_mean_{column_name_suffix}',
        f'{prefix}_overall_abs_mean_{column_name_suffix}',
        f'{prefix}_overall_std_{column_name_suffix}'
    ]

    qt_features_2 = [qt_features(i[1]) for i in df_matlab['quality_array'][0]]
    prefix = 'qt_power'
    column_names += [f'{prefix}_channel_{i}_mean_{column_name_suffix}' for i in range(1, num_channels + 1)]
    column_names += [f'{prefix}_channel_{i}_std_{column_name_suffix}' for i in range(1, num_channels + 1)]
    column_names += [
        f'{prefix}_overall_mean_{column_name_suffix}',
        f'{prefix}_overall_abs_mean_{column_name_suffix}',
        f'{prefix}_overall_std_{column_name_suffix}'
    ]
    qt_features_both = np.hstack([qt_features_1, qt_features_2])
    df = pd.DataFrame(qt_features_both, columns=column_names)
    df['FSID'] = child_ids
    return df


df_qt_6 = get_qt_features(loadmat('data/fnirs/QT-NIRS_analysis_6mo_LIpeekaboo.mat'), '6m')
df_crypto = pd.merge(left=df_crypto, right=df_qt_6, on='FSID', how='left')
crypto_qt_covariates_6 = list(df_qt_6.columns)
crypto_qt_covariates_6.remove('FSID')

df_qt_24 = get_qt_features(loadmat('data/fnirs/QT-NIRS_analysis_24mo_LIpeekaboo.mat'), '24m')
df_crypto = pd.merge(left=df_crypto, right=df_qt_24, on='FSID', how='left')
crypto_qt_covariates_24 = list(df_qt_24.columns)
crypto_qt_covariates_24.remove('FSID')

df_qt_36 = get_qt_features(loadmat('data/fnirs/QT-NIRS_analysis_36mo_LIpeekaboo.mat'), '36m')
df_provide = pd.merge(left=df_provide, right=df_qt_36, on='FSID', how='left')
provide_qt_covariates_36 = list(df_qt_36.columns)
provide_qt_covariates_36.remove('FSID')

df_qt_60 = get_qt_features(loadmat('data/fnirs/QT-NIRS_analysis_60mo_LIpeekaboo.mat'), '60m')
df_provide = pd.merge(left=df_provide, right=df_qt_60, on='FSID', how='left')
provide_qt_covariates_60 = list(df_qt_60.columns)
provide_qt_covariates_60.remove('FSID')


"""
Lobe variables
"""


def get_lobe_features(df_matlab, column_name_suffix=''):
    child_ids = [i[0][0] for i in df_matlab['subjects']]

    X = df_matlab['deoCorrFirOut']
    a, b = np.triu_indices(X.shape[0], 1)
    deocorr_features = X[a, b, :].T

    X = df_matlab['oxyCorrFirOut']
    a, b = np.triu_indices(X.shape[0], 1)
    oxycorr_features = X[a, b, :].T

    lobe_features = np.hstack([deocorr_features, oxycorr_features])
    columns = [f'lobe_deocorr_{i}_entry_{column_name_suffix}' for i in range(1, deocorr_features.shape[1] + 1)]
    columns += [f'lobe_oxycorr_{i}_entry_{column_name_suffix}' for i in range(1, oxycorr_features.shape[1] + 1)]

    df = pd.DataFrame(lobe_features, columns=columns)
    df['FSID'] = child_ids
    return df


df_lobe_6 = get_lobe_features(loadmat('data/fnirs/bgfcLobe6moPeekaboo.mat'), '6m')
df_crypto = pd.merge(left=df_crypto, right=df_lobe_6, on='FSID', how='left')
crypto_lobe_covariates_6 = list(df_lobe_6.columns)
crypto_lobe_covariates_6.remove('FSID')

df_lobe_24 = get_lobe_features(loadmat('data/fnirs/bgfcLobe24moPeekaboo.mat'), '24m')
df_crypto = pd.merge(left=df_crypto, right=df_lobe_24, on='FSID', how='left')
crypto_lobe_covariates_24 = list(df_lobe_24.columns)
crypto_lobe_covariates_24.remove('FSID')

df_lobe_36 = get_lobe_features(loadmat('data/fnirs/bgfcLobe36moPeekaboo.mat'), '36m')
df_provide = pd.merge(left=df_provide, right=df_lobe_36, on='FSID', how='left')
provide_lobe_covariates_36 = list(df_lobe_36.columns)
provide_lobe_covariates_36.remove('FSID')

df_lobe_60 = get_lobe_features(loadmat('data/fnirs/bgfcLobe60moPeekaboo.mat'), '60m')
df_provide = pd.merge(left=df_provide, right=df_lobe_60, on='FSID', how='left')
provide_lobe_covariates_60 = list(df_lobe_60.columns)
provide_lobe_covariates_60.remove('FSID')


"""
fNIRS features:
(1) raw
(2) eigenspectrum of graph laplacian, misc measures like per-channel mean, std, entropy
"""


def matrix_features(A_all):
    features = []
    for i in range(A_all.shape[-1]):
        A = A_all[:, :, i]
        np.fill_diagonal(A, 1)
        # graph laplacian
        L = np.diag(np.sum(A, 1)) - A
        if pd.isnull(L).all():
            feats = [np.nan] * (A_all.shape[0] * 3 + 4)
        else:
            # for entropy
            hist_counts, hist_bins = np.histogram(np.reshape(A, len(A) ** 2), bins=10, range=(-1, 1), density=True)
            overall_stats = [
                np.nanmean(A),
                np.nanmean(np.abs(A)),
                np.nanstd(A),
                -(hist_counts * np.ma.log(np.abs(hist_counts))).sum()
            ]
            feats = np.hstack([
                svdvals(L),
                np.nanmean(A, 1),
                np.nanstd(A, 1),
                overall_stats
            ])
        features.append(feats)
    return np.array(features)


def get_matrix_features(df_matlab, column_name_suffix=''):
    child_ids = [i[0][0] for i in df_matlab['subjects']]

    # raw
    X = df_matlab['deoCorr']
    a, b = np.triu_indices(X.shape[0], 1)
    deocorr_raw_features = X[a, b, :].T

    X = df_matlab['oxyCorr']
    a, b = np.triu_indices(X.shape[0], 1)
    oxycorr_raw_features = X[a, b, :].T

    raw_features = np.hstack([deocorr_raw_features, oxycorr_raw_features])
    columns = [f'raw_deocorr_{i}_entry_{column_name_suffix}' for i in range(1, deocorr_raw_features.shape[1] + 1)]
    columns += [f'raw_oxycorr_{i}_entry_{column_name_suffix}' for i in range(1, oxycorr_raw_features.shape[1] + 1)]

    df_raw = pd.DataFrame(raw_features, columns=columns)
    df_raw['FSID'] = child_ids

    # features
    deocorr_derived_features = matrix_features(df_matlab['deoCorr'])
    oxycorr_derived_features = matrix_features(df_matlab['oxyCorr'])

    num_channels = df_matlab['oxyCorr'].shape[0]

    prefix = 'raw_deocorr'
    column_names = [f'{prefix}_laplacian_svdval_{i}_{column_name_suffix}' for i in range(1, num_channels + 1)]
    column_names += [f'{prefix}_channel_{i}_mean_{column_name_suffix}' for i in range(1, num_channels + 1)]
    column_names += [f'{prefix}_channel_{i}_std_{column_name_suffix}' for i in range(1, num_channels + 1)]
    column_names += [
        f'{prefix}_overall_mean_{column_name_suffix}',
        f'{prefix}_overall_abs_mean_{column_name_suffix}',
        f'{prefix}_overall_std_{column_name_suffix}',
        f'{prefix}_overall_entropy_{column_name_suffix}',
    ]

    prefix = 'raw_oxycorr'
    column_names += [f'{prefix}_laplacian_svdval_{i}_{column_name_suffix}' for i in range(1, num_channels + 1)]
    column_names += [f'{prefix}_channel_{i}_mean_{column_name_suffix}' for i in range(1, num_channels + 1)]
    column_names += [f'{prefix}_channel_{i}_std_{column_name_suffix}' for i in range(1, num_channels + 1)]
    column_names += [
        f'{prefix}_overall_mean_{column_name_suffix}',
        f'{prefix}_overall_abs_mean_{column_name_suffix}',
        f'{prefix}_overall_std_{column_name_suffix}',
        f'{prefix}_overall_entropy_{column_name_suffix}',
    ]
    derived_features = np.hstack([deocorr_derived_features, oxycorr_derived_features])

    df_derived = pd.DataFrame(derived_features, columns=column_names)
    df_derived['FSID'] = child_ids

    return df_raw, df_derived


df_raw_6, df_derived_6 = get_matrix_features(loadmat('data/fnirs/bgfc6moPeekaboo.mat'), '6m')
df_crypto = pd.merge(left=df_crypto, right=df_raw_6, on='FSID', how='left')
crypto_raw_covariates_6 = list(df_raw_6.columns)
crypto_raw_covariates_6.remove('FSID')
df_crypto = pd.merge(left=df_crypto, right=df_derived_6, on='FSID', how='left')
crypto_derived_covariates_6 = list(df_derived_6.columns)
crypto_derived_covariates_6.remove('FSID')

df_raw_24, df_derived_24 = get_matrix_features(loadmat('data/fnirs/bgfc24moPeekaboo.mat'), '24m')
df_crypto = pd.merge(left=df_crypto, right=df_raw_24, on='FSID', how='left')
crypto_raw_covariates_24 = list(df_raw_24.columns)
crypto_raw_covariates_24.remove('FSID')
df_crypto = pd.merge(left=df_crypto, right=df_derived_24, on='FSID', how='left')
crypto_derived_covariates_24 = list(df_derived_24.columns)
crypto_derived_covariates_24.remove('FSID')

df_raw_36, df_derived_36 = get_matrix_features(loadmat('data/fnirs/bgfc36moPeekaboo.mat'), '36m')
df_provide = pd.merge(left=df_provide, right=df_raw_36, on='FSID', how='left')
provide_raw_covariates_36 = list(df_raw_36.columns)
provide_raw_covariates_36.remove('FSID')
df_provide = pd.merge(left=df_provide, right=df_derived_36, on='FSID', how='left')
provide_derived_covariates_36 = list(df_derived_36.columns)
provide_derived_covariates_36.remove('FSID')

df_raw_60, df_derived_60 = get_matrix_features(loadmat('data/fnirs/bgfc60moPeekaboo.mat'), '60m')
df_provide = pd.merge(left=df_provide, right=df_raw_60, on='FSID', how='left')
provide_raw_covariates_60 = list(df_raw_60.columns)
provide_raw_covariates_60.remove('FSID')
df_provide = pd.merge(left=df_provide, right=df_derived_60, on='FSID', how='left')
provide_derived_covariates_60 = list(df_derived_60.columns)
provide_derived_covariates_60.remove('FSID')

"""
assemble everything into reasonable formats
"""


crypto_output_6_covariates = ['rlraw_6_top_75', 'elraw_6_top_75']
crypto_input_6_covariates = [
    crypto_baseline_6_covariates,
    crypto_qt_covariates_6,
    crypto_lobe_covariates_6,
    crypto_raw_covariates_6,
    crypto_derived_covariates_6

]

crypto_output_24_covariates = ['rlraw_24_top_75', 'elraw_24_top_75']
crypto_input_24_covariates = [
    crypto_baseline_24_covariates,
    crypto_qt_covariates_24,
    crypto_lobe_covariates_24,
    crypto_raw_covariates_24,
    crypto_derived_covariates_24,

]

provide_output_36_covariates = ['rlraw_36_top_75', 'elraw_36_top_75']
provide_input_36_covariates = [
    provide_baseline_36_covariates,
    provide_qt_covariates_36,
    provide_lobe_covariates_36,
    provide_raw_covariates_36,
    provide_derived_covariates_36
]

provide_output_60_covariates = ['vbrawsc_top_75', 'comrawsc_top_75']
provide_input_60_covariates = [
    provide_baseline_60_covariates,
    provide_qt_covariates_60,
    provide_lobe_covariates_60,
    provide_raw_covariates_60,
    provide_derived_covariates_60
]

names_of_covariate_groups = ['baseline', 'quality_fnirs', 'lobe_fnirs', 'raw_fnirs', 'derived_fnirs']

"""
sanity check to make sure we don't have unexpected variables left out
"""
all_crypto_vars = sum(crypto_input_6_covariates, []) + sum(crypto_input_24_covariates, []) + \
    crypto_output_6_covariates + crypto_output_24_covariates
assert set(df_crypto.columns) - set(all_crypto_vars) == {'FSID'}

all_provide_vars = sum(provide_input_36_covariates, []) + sum(provide_input_60_covariates,
                                                              []) + provide_output_36_covariates + provide_output_60_covariates
assert set(df_provide.columns) - set(all_provide_vars) == {'FSID'}


# save to disk
with open("data/processed_data.pickle", "wb") as f:
    pickle.dump((
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
    ), f)
