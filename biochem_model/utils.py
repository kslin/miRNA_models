import numpy as np
import pandas as pd
from scipy import stats

def norm_matrix(mat):
    means = np.mean(mat, axis=1).reshape([-1, 1])
    return mat - means

def sigmoid(vals):
    return 1 / (1 + np.exp(-1 * vals))

def calc_r2(xs, ys):
    return stats.linregress(xs.flatten(), ys.flatten())[2]**2

def get_r2_unnormed(preds, labels):
    preds_normed = norm_matrix(preds)
    labels_normed = norm_matrix(labels)
    return calc_r2(preds_normed, labels_normed)


def get_nsites(features):
    nsites = features.reset_index()
    nsites['nsites'] = 1
    nsites = nsites.groupby(['transcript','mir']).agg({'nsites': np.sum})
    return nsites


def get_cdf(logFC_list, bins=None):
    """
    Parameters:
    ==========
    logFC_list: list of floats, list of log fold-change values to plot CDF
    
    Returns:
    =======
    list of floats: bin indices
    list of floats: cdf values corresponding to the bin indices
    """
    if len(logFC_list) < 5:
        return [],[]
    if bins is None:
        num_bins = int(len(logFC_list)/5)
    else:
        num_bins = bins
    counts,bin_edges = np.histogram(logFC_list,bins=num_bins)
    counts = counts / float(sum(counts))
    return bin_edges[1:], np.cumsum(counts)


def boolean_indexing(v, zero_val=0):
    """Turns a list of lists into a matrix with zero-padding (or any value-padding). Also returns a mask."""
    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape, zero_val, dtype=float)
    out[mask] = np.concatenate(v)
    return out, mask


def expand_features_3D(transcripts, mirs, feature_df):
    all_sites = []
    for ix, transcript in enumerate(transcripts):
        for iy, mir in enumerate(mirs):
            try:
                temp = feature_df.loc[(transcript, mir)]
                nsites = len(temp)
                all_sites.append(list(temp['log_KA'].values))
            except KeyError:
                all_sites.append([])
                
    return boolean_indexing(all_sites)

def expand_features_4D_fast(transcripts, mirs, feature_list, feature_df):
    all_sites = []
    for ix, transcript in enumerate(transcripts):
        for iy, mir in enumerate(mirs):
            try:
                temp = feature_df.loc[(transcript, mir)]
                nsites = len(temp)
                all_sites.append(list(temp[feature_list].values.flatten()))
            except KeyError:
                all_sites.append([])
                
    return boolean_indexing(all_sites)
                

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

def expand_feats_stypes(features, stypes, expand_vars, single_vars):
    expanded_features = []
    for stype in stypes:
        temp = features[expand_vars]
        stype_filter = features[[stype]].values
        temp.columns = [x + '_' + stype for x in temp.columns]
        temp *= stype_filter
        expanded_features.append(temp)

    expanded_features.append(features[single_vars])
    expanded_features = pd.concat(expanded_features, axis=1, join='inner')

    # get rid of columns of all zeros, for example 6mer PCT
    for col in expanded_features.columns:
        if np.std(expanded_features[col].values) < 0.00001:
            expanded_features = expanded_features.drop(columns=[col])

    return expanded_features


def bayesian_slope(xs, ys, mu_0, prior_strength):
    """Calculates slope for Bayesian linear regression"""
    xs = xs - np.mean(xs)
    ys = ys - np.mean(ys)

    slope = stats.linregress(xs, ys)[0]
    return (1 / (np.sum(xs**2) + prior_strength)) * ((np.sum(xs**2) * slope) + (mu_0 * prior_strength))



def bayesian_intercept(xs, ys, mu_0, prior_strength):
    """Uses the slope from Bayesian linear regression to get the intercept"""
    intercepts = []
    for ix in range(xs.shape[0]):
        temp_x = xs[ix,:]
        temp_y = ys[ix,:]
        if np.var(temp_x) < 1e-10:
            intercepts.append(np.mean(temp_y))
        else:
            b_slope = bayesian_slope(temp_x, temp_y, mu_0, prior_strength)
            intercepts.append(np.mean(temp_y) - (b_slope * np.mean(temp_x)))
        
    return np.array(intercepts)