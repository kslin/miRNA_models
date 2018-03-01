import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import regex


def calc_rsq(predicted, actual):
    SS_res = np.sum((predicted - actual)**2)
    SS_tot = np.sum((actual - np.mean(actual))**2)

    return 1.0 - (SS_res/SS_tot)

### SEQUENCE FUNCTIONS ###

def rev_comp(seq):
    match_dict = {'A': 'T',
                  'T': 'A',
                  'C': 'G',
                  'G': 'C'}

    return ''.join([match_dict[x] for x in seq])[::-1]

def complementary(seq):
    match_dict = {'A': 'T',
                  'T': 'A',
                  'C': 'G',
                  'G': 'C'}

    return ''.join([match_dict[x] for x in seq])


def get_color(sitem8, seq):
    if (sitem8 + 'A') in seq:
        return 'blue'
    elif sitem8 in seq:
        return 'green'
    elif (sitem8[1:] + 'A') in seq:
        return 'orange'
    elif (sitem8[1:]) in seq:
        return 'red'
    else:
        return 'grey'


def one_hot_encode_nt(seq, nt_order):
    """Convert RNA sequence to one-hot encoding"""
    
    one_hot = [list(np.array(nt_order == nt, dtype=int)) for nt in seq]
    one_hot = [item for sublist in one_hot for item in sublist]
    
    return np.array(one_hot)


def one_hot_encode_nt_new(seq, nt_order):
    """Convert RNA sequence to one-hot encoding"""
    
    one_hot = np.zeros([len(seq) * 4])
    for i, nt in enumerate(seq):
        one_hot[i*4 + np.argmax(nt_order == nt)] = 2.0
    
    return one_hot


def make_square(seq1, seq2):
    """Given two sequences, calculate outer product of one-hot encodings"""

    # noise = np.random.normal(loc=0, scale=0.01, size=16*len(seq1)*len(seq2)).reshape((4*len(seq1), 4*len(seq2)))

    square = np.outer(one_hot_encode_nt(seq1, np.array(['A','T','C','G'])),
                    one_hot_encode_nt(seq2, np.array(['T','A','G','C'])))

    square = ((square*4) - 0.25)#.reshape((4*len(seq1), 4*len(seq2), 1))

    return square# + noise


# def get_seqs(utr, site, only_canon=False):
#     if only_canon:
#         locs = [m.start() + 1 for m in re.finditer(site, utr)]

#     else:
#         locs1 = [m.start() for m in re.finditer(site[1:-1], utr)]
#         locs2 = [(m.start() - 1) for m in re.finditer(site[2:], utr)]
#         locs = list(set(locs1 + locs2))

#     seqs = [utr[loc-4:loc+8] for loc in locs if (loc-4 >=0) and (loc+8 <= len(utr))]
#     return seqs

def get_seqs(utr, site, only_canon=False):
    if only_canon:
        locs = [m.start() + 1 for m in re.finditer(site, utr)]

    # else:
    #     locs1 = [m.start() for m in re.finditer(site[1:], utr)]
    #     locs2 = [(m.start() + 1) for m in re.finditer(site[:-1], utr)]
    #     locs = list(set(locs1 + locs2))

    else:
        locs1 = [m.start() - 1 for m in re.finditer(site[2:], utr)]
        locs2 = [(m.start()) for m in re.finditer(site[1:-1], utr)]
        locs3 = [(m.start() + 1) for m in re.finditer(site[1:-2], utr)]
        locs = list(set(locs1 + locs2 + locs3))

    seqs = [utr[loc-4:loc+8] for loc in locs if (loc-4 >=0) and (loc+8 <= len(utr))]
    return seqs


def generate_random_seq(len):
    nts = ['A','T','C','G']
    seq = np.random.choice(nts, size=len, replace=True)
    return ''.join(seq)


def make_pretrain_data(size, mirlen, seqlen=12):

    mirseqs = [generate_random_seq(mirlen) for _ in range(size)]
    stypes = np.random.choice(['8mer','7m8','7a1','6mer', '5mer','4mer'], size=size, replace=True)
    stype_dict = {'8mer': 6, '7m8': 4.8, '7a1': 3.6, '6mer': 2.4, '5mer': 0.7, '4mer': -0.3}
    nts = ['A','T','C','G']
    sites, upflanks, downflanks = [], [], []
    for stype, m in zip(stypes, mirseqs):
        if stype == '8mer':
            sites.append(complementary(m[-8:-1]) + 'A')

            upflanks.append(generate_random_seq(2))
            downflanks.append(generate_random_seq(2))
        elif stype == '7m8':
            sites.append(complementary(m[-8:-1]))

            upflanks.append(generate_random_seq(2))
            downflanks.append(np.random.choice(['T','C','G']) + generate_random_seq(2))
        elif stype == '7a1':
            possibilities = [nt for nt in nts if nt != complementary(m[-8])]
            sites.append(complementary(m[-7:-1]) + 'A')

            upflanks.append(generate_random_seq(2) + np.random.choice(possibilities))
            downflanks.append(generate_random_seq(2))
        elif stype == '6mer':
            possibilities = [nt for nt in nts if nt != complementary(m[-8])]
            sites.append(complementary(m[-7:-1]))

            upflanks.append(generate_random_seq(2) + np.random.choice(possibilities))
            downflanks.append(np.random.choice(['T','C','G']) + generate_random_seq(2))

        elif stype == '5mer':
            temp_site = list(complementary(m[-7:-1]))
            random_mutation_site = np.random.choice([0,5])
            possibilities = [nt for nt in nts if nt != temp_site[random_mutation_site]]
            temp_site[random_mutation_site] = np.random.choice(possibilities)

            sites.append(''.join(temp_site))

            upflanks.append(generate_random_seq(3))
            downflanks.append(generate_random_seq(3))

        else:
            temp_site = list(complementary(m[-7:-1]))
            which_4mer = np.random.choice([0,1])
            if which_4mer == 0:
                random_mutation_sites = [0,5]
            else:
                random_mutation_sites = [0,1]

            for rms in random_mutation_sites:
                possibilities = [nt for nt in nts if nt != temp_site[rms]]
                temp_site[rms] = np.random.choice(possibilities)

            sites.append(''.join(temp_site))

            upflanks.append(generate_random_seq(3))
            downflanks.append(generate_random_seq(3))

    flanks, seqs = [], []
    for (x, y, z) in zip(upflanks, sites, downflanks):
        flanks.append(x+z)
        seqs.append(x+y+z)

    flanking_AU = [((f.count('A') + f.count('T'))*1.6 / len(f)) - 0.8 for f in flanks]

    batch_y = np.array([stype_dict[s] for s in stypes]) + np.array(flanking_AU) + ((np.random.random(size=size) - 0.5))

    batch_x = np.zeros((size, 4*mirlen, 4*seqlen))
    for i, (mirseq, seq) in enumerate(zip(mirseqs, seqs)):
        batch_x[i, :, :] = make_square(mirseq, seq)

    return np.expand_dims(batch_x, 3), np.expand_dims(batch_y, 1)


### TF functions ###

def get_conv_params(dim1, dim2, in_channels, out_channels, layer_name):

    # create variables for weights and biases
    with tf.name_scope('weights'):
        weights = tf.get_variable("{}_weight".format(layer_name),
                               shape=[dim1, dim2, in_channels, out_channels],
                               initializer=tf.truncated_normal_initializer(stddev=0.1))

        # add variable to collection of variables
        tf.add_to_collection('weight', weights)
    with tf.name_scope('biases'):
        biases = tf.get_variable("{}_bias".format(layer_name), shape=[out_channels],
                              initializer=tf.constant_initializer(0.0))

        # add variable to collection of variables
        tf.add_to_collection('bias', biases)

    return weights, biases


### GRAPHING FUNCTION ###


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
    fig = plt.figure(figsize=(w*ncols, h*nrows))
    for i in range(nrows):
        for j in range(ncols):
            v = conv_weights[:,:,i,j].reshape(h,w)
            ax = plt.subplot(nrows, ncols, plot_num)
            sns.heatmap(v, xticklabels=xlabels, yticklabels=ylabels,
                        cmap=plt.cm.bwr, vmin=vmin, vmax=vmax)
            plot_num += 1

    # plt.colorbar()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()