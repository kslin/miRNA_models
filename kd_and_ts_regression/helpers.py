import re
import operator

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import regex

MIR_NTS = np.array(['A','T','C','G'])
SEQ_NTS = np.array(['T','A','G','C'])

MIR_NT_DICT = {nt:ix for (ix, nt) in enumerate(MIR_NTS)}
SEQ_NT_DICT = {nt:ix for (ix, nt) in enumerate(SEQ_NTS)}
TARGETS = np.eye(4)


### TFRecords FUNCTIONS ###
def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


### SEQUENCE FUNCTIONS ###

def rev_comp(seq):
    match_dict = {'A': 'T',
                  'T': 'A',
                  'C': 'G',
                  'G': 'C'}

    return ''.join([match_dict[x] for x in seq][::-1])


def one_hot_encode(seq, nt_dict, targets):
    seq = [nt_dict[nt] for nt in seq]
    return targets[seq].flatten()


def generate_random_seq(length):
    nts = ['A','T','C','G']
    seq = np.random.choice(nts, size=length, replace=True)
    return ''.join(seq)


def get_target_no_match(mirna_sequence, length):
    """Given a miRNA sequence, return a random target sequence without 4 nt of contiguous pairing"""
    rc = rev_comp(mirna_sequence[:8])
    off_limits = [rc[ix:ix+4] for ix in range(5)]
    while True:
        target = generate_random_seq(length)
        keep = True
        for subseq in off_limits:
            if subseq in target:
                keep = False
                break

        if keep:
            return target


def remove_overlaps(locs, distance=6):
    if len(locs) < 2:
        return locs

    diffs = np.diff(locs)
    while np.min(diffs) < distance:
        overlap = np.argmax(diffs < distance)
        # keep_which = np.random.randint(2)
        keep_which = 1
        del locs[overlap + keep_which]

        if len(locs) == 1:
            break

        diffs = np.diff(locs)

    return locs

def priority_order(locs, overlap_dist):

    # make dictionary of loc occurences and order
    loc_dict = {}
    for ix, l in enumerate(locs):
        if l not in loc_dict:
            loc_dict[l] = (-1,ix)
        else:
            temp_count, temp_ix = loc_dict[l]
            loc_dict[l] = (temp_count - 1, temp_ix)

    loc_tuples = [(l, count, ix) for (l, (count, ix)) in loc_dict.items()]
    loc_tuples.sort(key = operator.itemgetter(1, 2))

    unique_locs = [t[0] for t in loc_tuples]
    nonoverlapping_locs = []
    prev = -100
    for l in unique_locs:
        if abs(l - prev) > overlap_dist:
            nonoverlapping_locs.append(l)
            prev = l
 
    return nonoverlapping_locs


def get_locs(utr, site, overlap_dist, only_canon):
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
        locs0 = [m.start() - 1 for m in re.finditer(site[2:], utr)] # pos 1-6
        locs1 = [m.start() for m in re.finditer(site[1:-1], utr)] # pos 2-7 (start of 6mer site)
        locs2 = [m.start() + 1 for m in re.finditer(site[:-2], utr)] # pos 3-8
        locs = (locs1 + locs2 + locs0)

    else:
        # get site locations of all 4mer subsequences of the 8mer site
        locs0 = [m.start() - 3 for m in re.finditer(site[4:], utr)] # pos 1-4
        locs1 = [m.start() - 2 for m in re.finditer(site[3:-1], utr)] # pos 2-5
        locs2 = [m.start() - 1 for m in re.finditer(site[2:-2], utr)] # pos 3-6
        locs3 = [m.start() for m in re.finditer(site[1:-3], utr)] # pos 4-7 (start of 6mer site)
        locs4 = [m.start() + 1 for m in re.finditer(site[:-4], utr)] # pos 5-8
        locs = (locs1 + locs2 + locs0 + locs3 + locs4)

    # get rid of any that would put the 6mer site outside the bounds of the UTR
    locs = [l for l in locs if ((l >= 0) and ((l+6) <= len(utr)))]

    # if 1 or fewer sites found, return list as is
    if len(locs) <= 1:
        return locs

    # sites already listed in order of priority, except longer sites take precedent
    return priority_order(locs, overlap_dist)


# ### TS7 features ###
# def get_rnaplfold_data(gene, utr, rnaplfold_folder):
#     """
#     Run RNAplfold and get pairing probabilities for a utr

#     Parameters
#     ----------
#     gene: string, name of gene
#     utr: string, utr sequence

#     Output
#     ------
#     pandas DataFrame: pairing probabilities at each position
#     """

#     # sanitize name of file so we don't break the shell
#     gene_name = shlex.split(gene)[0]

#     # navigate to the folder for RNAplfold data
#     cwd = os.getcwd()
#     os.chdir(rnaplfold_folder)

#     # write sequence to a temporary file
#     mytempfile = 'temp_{}.fa'.format(gene_name)
#     with open(mytempfile, 'wb') as f:
#         f.write('>{}\n{}'.format(gene_name, utr))

#     # call RNAplfold
#     length = min(40, len(utr))
#     window = min(80, len(utr))
#     mycall = 'RNAplfold -L {} -W {} -u 14 < {}'.format(length, window,
#                                                        mytempfile)
#     subprocess.call([mycall], shell=True, stdout=subprocess.PIPE)
#     lunp_file = '{}_lunp'.format(gene_name)

#     # read data and convert to a dataframe
#     rnaplfold_data = pd.read_csv(lunp_file, sep='\t',
#                                  header=1).set_index(' #i$')

#     os.remove(mytempfile)
#     os.remove(lunp_file)
#     os.remove('{}_dp.ps'.format(gene_name))

#     os.chdir(cwd)

#     return rnaplfold_data

def calculate_threep_score(mirna, utr, site_start, upstream_limit):
    """
    Calculate the three-prime pairing score

    Parameters
    ----------
    mirna: string, miRNA sequence
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
    mirna_3p = mirna[8:] # miRNA sequence from position 9 onward
    trailing = utr[max(0, site_start-upstream_limit): site_start+2] # site sequence up to edges of possible 8mer site
    utr_5p = rev_comp(trailing)

    # initiate array for dynamic programming search
    scores = np.empty((len(utr_5p) + 1, len(mirna_3p) + 1))
    scores.fill(np.nan)
    possible_scores = [0]

    # fill in array
    for i, nt1 in enumerate(utr_5p):
        for j, nt2 in enumerate(mirna_3p):
            if nt1 == nt2:
                new_score = 0.5 + 0.5 * ((j > 3) & (j < 8))
                if np.isnan(scores[i, j]) == False:
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