import re
import operator

import numpy as np
import pandas as pd
import tensorflow as tf


### General sequences functions ###
def rev_comp(seq):
    """ Get reverse complement of sequence"""

    match_dict = {'A': 'T',
                  'T': 'A',
                  'C': 'G',
                  'G': 'C'}

    return ''.join([match_dict[x] for x in seq][::-1])


def one_hot_encode(seq):
    """ 1-hot encode ATCG sequence """
    nt_dict = {
        'A': 0,
        'T': 1,
        'C': 2,
        'G': 3
    }
    targets = np.eye(4)

    seq = [nt_dict[nt] for nt in seq]
    return targets[seq].flatten().astype(int)


def generate_random_seq(length):
    """Generate random sequence"""
    nts = ['A', 'T', 'C', 'G']
    seq = np.random.choice(nts, size=length, replace=True)
    return ''.join(seq)


def get_target_no_match(mirna_sequence, length):
    """Given a miRNA sequence, return a random target sequence without 4 nt of contiguous pairing"""
    rc = rev_comp(mirna_sequence[1:8]) + 'A'
    off_limits = [rc[ix:ix + 4] for ix in range(5)]
    while True:
        target = generate_random_seq(length)
        keep = True
        for subseq in off_limits:
            if subseq in target:
                keep = False
                break

        if keep:
            return target


### TFRecords Functions ###
def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


### UTR sequence functions ###
def _priority_order(locs, overlap_dist):

    # make dictionary of loc occurences and order
    loc_dict = {}
    for ix, l in enumerate(locs):
        if l not in loc_dict:
            loc_dict[l] = (-1, ix)
        else:
            temp_count, temp_ix = loc_dict[l]
            loc_dict[l] = (temp_count - 1, temp_ix)

    loc_tuples = [(l, count, ix) for (l, (count, ix)) in loc_dict.items()]
    loc_tuples.sort(key=operator.itemgetter(1, 2))

    unique_locs = [t[0] for t in loc_tuples]
    nonoverlapping_locs = []
    prev = -100
    for l in unique_locs:
        if abs(l - prev) > overlap_dist:
            nonoverlapping_locs.append(l)
            prev = l

    return nonoverlapping_locs


def get_sites_from_utr(utr, site, overlap_dist, only_canon):
    """
    Given a UTR sequence and the site sequence of an 8mer, return the location of all potential sites with >= 4nt of the 8mer.

    Parameters:
        utr (string): UTR sequence
        site (string): 8mer site sequence
        overlap_dist (int): overlap distance allowed between sites
        only_canon (bool): if true, only return location of canonical sites

    Returns:
        list of ints: location of all desired sites in a UTR
    """
    if only_canon:
        # get site locations of all canonical sites
        locs0 = [m.start() - 1 for m in re.finditer(site[2:], utr)]  # pos 1-6
        locs1 = [m.start() for m in re.finditer(site[1:-1], utr)]  # pos 2-7 (start of 6mer site)
        locs2 = [m.start() + 1 for m in re.finditer(site[:-2], utr)]  # pos 3-8
        locs = (locs1 + locs2 + locs0)

    else:
        # get site locations of all 4mer subsequences of the 8mer site
        locs0 = [m.start() - 3 for m in re.finditer(site[4:], utr)]  # pos 1-4
        locs1 = [m.start() - 2 for m in re.finditer(site[3:-1], utr)]  # pos 2-5
        locs2 = [m.start() - 1 for m in re.finditer(site[2:-2], utr)]  # pos 3-6
        locs3 = [m.start() for m in re.finditer(site[1:-3], utr)]  # pos 4-7 (start of 6mer site)
        locs4 = [m.start() + 1 for m in re.finditer(site[:-4], utr)]  # pos 5-8
        locs = (locs1 + locs2 + locs0 + locs3 + locs4)

    # get rid of any that would put the 6mer site outside the bounds of the UTR
    locs = [l for l in locs if ((l >= 0) and ((l + 6) <= len(utr)))]

    # if 1 or fewer sites found, return list as is
    if len(locs) > 1:
        locs = _priority_order(locs, overlap_dist)

    utr_ext = ('TTT' + utr + 'TTT')
    seqs = [utr_ext[l:l + 12] for l in locs]

    # sites already listed in order of priority, except longer sites take precedent
    return seqs, locs


### TS7 Feature functions ###
def calculate_threep_score(mirseq, utr, site_start, upstream_limit):
    """
    Calculate the three-prime pairing score

    Parameters
    ----------
    mirseq: string, miRNA sequence
    utr: string, utr sequence
    site_start: int, start of 12mer site
    upstream_limit: int, how far upstream to look for 3p pairing

    Output
    ------
    float: 3' pairing score
    """
    if site_start <= 0:
        return 0

    # get the 3' region of the mirna and the corresponding utr seq
    mirseq_3p = mirseq[8:]  # miRNA sequence from position 9 onward
    trailing = utr[max(0, site_start - upstream_limit): site_start + 2]  # site sequence up to edges of possible 8mer site
    utr_5p = utils.rev_comp(trailing)

    # initiate array for dynamic programming search
    scores = np.empty((len(utr_5p) + 1, len(mirseq_3p) + 1))
    scores.fill(np.nan)
    possible_scores = [0]

    # fill in array
    for i, nt1 in enumerate(utr_5p):
        for j, nt2 in enumerate(mirseq_3p):
            if nt1 == nt2:
                new_score = 0.5 + 0.5 * ((j > 3) & (j < 8))
                if not np.isnan(scores[i, j]):
                    new_score += scores[i, j]
                    scores[i + 1, j + 1] = new_score
                    possible_scores.append(new_score)
                else:
                    offset_penalty = max(0, (abs(i - j) - 2) * 0.5)
                    scores[i + 1, j + 1] = new_score - offset_penalty
            else:
                scores[i + 1, j + 1] = float('NaN')

    return np.nanmax(possible_scores)


def calculate_local_au(utr, site_start):
    """
    Calculate the local AU score

    Parameters
    ----------
    utr: string, utr sequence
    site_start: int, start of 12mer site

    Output
    ------
    float: local AU score
    """
    # find A, U and weights upstream of site
    upstream = utr[max(0, site_start - 30): max(0, site_start)]
    upstream = [int(x in ['A', 'U']) for x in upstream]
    upweights = [1.0 / (x + 1) for x in range(len(upstream))][::-1]

    # find A,U and weights downstream of site
    downstream = utr[site_start + 12:min(len(utr), site_start + 42)]
    downstream = [int(x in ['A', 'U']) for x in downstream]
    downweights = [1.0 / (x + 1) for x in range(len(downstream))]

    weighted = np.dot(upstream, upweights) + np.dot(downstream, downweights)
    total = float(sum(upweights) + sum(downweights))

    return weighted / total


def get_ts7_features(mirseq, locs, utr, utr_len, orf_len, upstream_limit, rnaplfold_data):
    # calculate TS7 features
    features = []
    for loc in locs:

        # get ts7 features
        local_au = calculate_local_au(utr, loc - 3)
        threep = calculate_threep_score(mirseq, utr, loc - 3, upstream_limit)
        min_dist = min(loc, utr_len - (loc + 6))
        assert (min_dist >= 0), (loc, utr_len)

        # use the rnaplfold data to calculate the site accessibility
        site_start_for_SA = loc + 7
        if (site_start_for_SA) not in rnaplfold_data.index:
            sa_score = 0
        else:
            row_vals = rnaplfold_data.loc[site_start_for_SA].values[:14]  # pos 1-14 unpaired, Agarwal 2015
            # row_vals = rnaplfold_data.loc[site_start_for_SA].values[:10]  # pos 1-10 unpaired, Sean

            for raw_sa_score in row_vals[::-1]:
                if not np.isnan(raw_sa_score):
                    break

            if np.isnan(raw_sa_score):
                sa_score = np.nan
            elif raw_sa_score <= 0:
                sa_score = -5.0
                print("warning, nan sa_score")
            else:
                sa_score = np.log10(raw_sa_score)

        features += [local_au, threep, min_dist/10000, sa_score, utr_len/10000, orf_len/10000]

    return np.array(features).astype(float)
