import re
import operator

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


### SEQUENCE FUNCTIONS ###

def rev_comp(seq):
    match_dict = {'A': 'T',
                  'T': 'A',
                  'C': 'G',
                  'G': 'C'}

    return ''.join([match_dict[x] for x in seq][::-1])


def count_num_canon(utr, sitem8):
    # Six canonical sites, remove double-counted sites
    num_6m8 = utr.count(sitem8[:-1])
    num_6a1 = utr.count(sitem8[2:] + 'A')
    num_6 = utr.count(sitem8[1:])
    num_7m8 = utr.count(sitem8)
    num_7a1 = utr.count(sitem8[1:] + 'A')

    return num_6m8 + num_6a1 + num_6 - num_7m8 - num_7a1


def get_stype_six_canon(site8, seq):
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


def generate_random_seq(length):
    nts = ['A', 'T', 'C', 'G']
    seq = np.random.choice(nts, size=length, replace=True)
    return ''.join(seq)


def get_target_no_match(mirna_sequence, length):
    """Given a miRNA sequence, return a random target sequence without 4 nt of contiguous pairing"""
    rc = rev_comp(mirna_sequence[:8])
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


### SEQUENCE ENCODING HELPERS ###

def one_hot_encode(seq, nt_dict, targets):
    seq = [nt_dict[nt] for nt in seq]
    return targets[seq].flatten()


def priority_order(locs, overlap_dist):

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


def get_seqs(utr, site, overlap_dist, only_canon):
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
        locs = priority_order(locs, overlap_dist)

    utr_ext = ('TTT' + utr + 'TTT')
    seqs = [utr_ext[l:l + 12] for l in locs]

    # sites already listed in order of priority, except longer sites take precedent
    return seqs


### GRAPHING FUNCTIONS ###

def graph_convolutions(conv_weights, xlabels, ylabels, fname):
    vmin, vmax = np.min(conv_weights), np.max(conv_weights)
    dim = conv_weights.shape
    nrows = dim[2]
    ncols = dim[3]
    h, w = dim[0], dim[1]

    if xlabels is None:
        xlabels = [str(x) for x in (np.arange(w) + 1)[::-1]]

    if ylabels is None:
        ylabels = [str(y) for y in (np.arange(h) + 1)[::-1]]

    plot_num = 1
    fig = plt.figure(figsize=(w * ncols, h * nrows))
    for i in range(nrows):
        for j in range(ncols):
            v = conv_weights[:, :, i, j].reshape(h, w)
            ax = plt.subplot(nrows, ncols, plot_num)
            sns.heatmap(v, xticklabels=xlabels, yticklabels=ylabels,
                        cmap=plt.cm.bwr, vmin=vmin, vmax=vmax, ax=ax)
            plot_num += 1

    # plt.colorbar()
    plt.tight_layout()
    fig.savefig(fname)
    plt.close()
