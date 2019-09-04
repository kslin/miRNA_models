import numpy as np
import pandas as pd
from scipy import stats


def site_to_ints(site):
    nt_dict = {
        'A': 0,
        'T': 1,
        'C': 2,
        'G': 3,
        'X': -1
    }

    return [nt_dict[x] for x in site]

def mir_site_pair_to_ints(mir, site):
    nt_dict = {
        'A': 0,
        'T': 1,
        'C': 2,
        'G': 3,
    }

    ints = []
    for nt1 in mir:
        if nt1 == 'X':
            ints += [-1] * len(site)
            continue
        for nt2 in site:
            if nt2 == 'X':
                ints.append(-1)

            else:
                ints.append((nt_dict[nt1] * 4) + nt_dict[nt2])

    return ints

# def get_nsites(features):
#     nsites = features.reset_index()
#     nsites['nsites'] = 1
#     nsites = nsites.groupby(['transcript','mir']).agg({'nsites': np.sum})
#     return nsites


def sigmoid(vals):
    return 1.0 / (1.0 + np.exp(-1 * vals))


def rev_comp(seq):
    """ Get reverse complement of sequence"""

    match_dict = {'A': 'T',
                  'T': 'A',
                  'C': 'G',
                  'G': 'C'}

    return ''.join([match_dict[x] for x in seq][::-1])

def one_hot_encode(seq):
    if len(seq) == 0:
        return []
    """ 1-hot encode ATCG sequence """
    nt_dict = {
        'A': 0,
        'T': 1,
        'C': 2,
        'G': 3,
        'X': 4
    }
    targets = np.ones([5, 4]) / 4.0
    targets[:4, :] = np.eye(4)
    seq = [nt_dict[nt] for nt in seq]
    return list(targets[seq].flatten())


def generate_random_seq(length):
    """Generate random sequence"""
    nts = ['A', 'T', 'C', 'G']
    seq = np.random.choice(nts, size=length, replace=True)
    return ''.join(seq)


def get_best_stype(site8, seq):
    if site8 in seq:
        return '8mer'
    elif site8[:-1] in seq:
        return '7mer-m8'
    elif site8[1:] in seq:
        return '7mer-a1'
    elif site8[1:-1] in seq:
        return '6mer'
    elif site8[:-2] in seq:
        return '6mer-m8'
    elif site8[2:] in seq:
        return '6mer-a1'
    else:
        return 'no site'


def get_centered_stype(site8, seq):
    if site8 == seq[2:-2]:
        return '8mer'
    elif site8[:-1] == seq[2:-3]:
        return '7mer-m8'
    elif site8[1:] == seq[3:-2]:
        return '7mer-a1'
    elif site8[1:-1] == seq[3:-3]:
        return '6mer'
    elif site8[:-2] == seq[2:-4]:
        return '6mer-m8'
    elif site8[2:] == seq[4:-2]:
        return '6mer-a1'
    else:
        return 'no site'


# def norm_matrix(mat):
#     means = np.mean(mat, axis=1).reshape([-1, 1])
#     return mat - means


# def get_r2_unnormed(preds, labels):
#     preds_normed = norm_matrix(preds)
#     labels_normed = norm_matrix(labels)
#     return calc_r2(preds_normed, labels_normed)

