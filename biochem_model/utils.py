import copy

import numpy as np
import pandas as pd
from scipy import stats


def sigmoid(vals):
    return 1 / (1 + np.exp(-1 * vals))


def expand_features_4D(transcripts, mirs, max_nsites, feature_list, feature_df):
    features_4D = np.zeros([len(transcripts), len(mirs), max_nsites, len(feature_list)])
    for ix, transcript in enumerate(transcripts):
        for iy, mir in enumerate(mirs):
            try:
                temp = feature_df.loc[(transcript, mir)]
                nsites = len(temp)
                features_4D[ix, iy, :nsites, :] = temp[feature_list].values
            except KeyError:
                continue

    mask = ((np.abs(np.sum(features_4D, axis=3))) != 0).astype(int)

    return features_4D, mask


def split_vals(vals_4D, zero_indices):
    """
    Given a 4D matrix with ka values and features, split into ka_vals (3D), features (4D), and nosite_features (4D)
    """

    ka_vals_3D = vals_4D[:, :, :, 0]
    features_4D = vals_4D[:, :, :, 1:]
    nosite_features_4D = copy.copy(features_4D)
    for ix in zero_indices:
        nosite_features_4D[:, :, :, ix - 1] = 0

    return ka_vals_3D, features_4D, nosite_features_4D
