import re
import operator

import numpy as np
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