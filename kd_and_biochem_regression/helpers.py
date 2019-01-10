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


# def calc_rsq(predicted, actual):
#     SS_res = np.sum((predicted - actual)**2)
#     SS_tot = np.sum((actual - np.mean(actual))**2)

#     return 1.0 - (SS_res/SS_tot)

# def sigmoid(x):
#     return 1.0 / (1.0 + np.exp(-1*x))

# ### DATA LOADING FUNCTIONS ###

# def load_biochem(filename):
#     biochem_data = pd.read_csv(filename, sep='\t')
#     biochem_data.columns = ['mir','mirseq_full','seq','log kd','stype']

#     # compute Ka's
#     biochem_data['log ka'] = (-1.0 * biochem_data['log kd'])
#     biochem_data['mirseq'] = [config.MIRSEQ_DICT_MIRLEN[mir] for mir in biochem_data['mir']]
#     biochem_data['sitem8'] = [helpers.rev_comp(mirseq[1:8]) for mirseq in biochem_data['mirseq_full']]
#     biochem_data['color'] = [helpers.get_color(sitem8, seq) for (sitem8, seq) in zip(biochem_data['sitem8'], biochem_data['seq'])]
#     biochem_data['color2'] = [helpers.get_color(sitem8, seq[2:10]) for (sitem8, seq) in zip(biochem_data['sitem8'], biochem_data['seq'])]

#     # get rid of sequences with sites out of register
#     biochem_data = biochem_data[biochem_data['color'] == biochem_data['color2']].drop('color2',1)

#     return biochem_data

### SEQUENCE FUNCTIONS ###

def rev_comp(seq):
    match_dict = {'A': 'T',
                  'T': 'A',
                  'C': 'G',
                  'G': 'C'}

    return ''.join([match_dict[x] for x in seq][::-1])

def complementary(seq):
    match_dict = {'A': 'T',
                  'T': 'A',
                  'C': 'G',
                  'G': 'C'}

    return ''.join([match_dict[x] for x in seq])


def count_num_canon(utr, sitem8):
    # Six canonical sites, remove double-counted sites
    num_6m8 = utr.count(sitem8[:-1])
    num_6a1 = utr.count(sitem8[2:] + 'A')
    num_6 = utr.count(sitem8[1:])
    num_7m8 = utr.count(sitem8)
    num_7a1 = utr.count(sitem8[1:] + 'A')
    num_8 = utr.count(sitem8 + 'A')
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


# def get_color(sitem8, seq):
#     if (sitem8 + 'A') in seq:
#         return 'blue'
#     elif sitem8 in seq:
#         return 'green'
#     elif (sitem8[1:] + 'A') in seq:
#         return 'orange'
#     elif (sitem8[1:]) in seq:
#         return 'red'
#     else:
#         return 'grey'


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


# def one_hot_encode_nt(seq, nt_order):
#     """Convert RNA sequence to one-hot encoding"""
    
#     one_hot = [list(np.array(nt_order == nt, dtype=int)) for nt in seq]
#     one_hot = [item for sublist in one_hot for item in sublist]
    
#     return np.array(one_hot)


# def one_hot_encode_nt_new(seq, nt_order):
#     """Convert RNA sequence to one-hot encoding"""
    
#     one_hot = np.zeros([len(seq) * 4])
#     for i, nt in enumerate(seq):
#         one_hot[i*4 + np.argmax(nt_order == nt)] = 2.0
    
#     return one_hot


# def make_square(seq1, seq2):
#     """Given two sequences, calculate outer product of one-hot encodings"""

#     # noise = np.random.normal(loc=0, scale=0.01, size=16*len(seq1)*len(seq2)).reshape((4*len(seq1), 4*len(seq2)))

#     square = np.outer(one_hot_encode_nt(seq1, np.array(['A','T','C','G'])),
#                     one_hot_encode_nt(seq2, np.array(['T','A','G','C'])))

#     square = ((square*4) - 0.25)#.reshape((4*len(seq1), 4*len(seq2), 1))

#     return square# + noise


# def get_seqs(utr, site, only_canon=False):
#     if only_canon:
#         locs = [m.start() + 1 for m in re.finditer(site, utr)]

#     else:
#         locs1 = [m.start() for m in re.finditer(site[1:-1], utr)]
#         locs2 = [(m.start() - 1) for m in re.finditer(site[2:], utr)]
#         locs = list(set(locs1 + locs2))

#     seqs = [utr[loc-4:loc+8] for loc in locs if (loc-4 >=0) and (loc+8 <= len(utr))]
#     return seqs


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


# def get_seqs(utr, site, only_canon=False):
#     utr_len = len(utr)
#     utr_ext = utr + 'TTT'

#     if only_canon:
#         locs2 = remove_overlaps([(m.start()) for m in re.finditer(site[2:], utr)])
#         locs3 = remove_overlaps([(m.start() + 1) for m in re.finditer(site[1:-1], utr)])
#         locs4 = remove_overlaps([(m.start() + 2) for m in re.finditer(site[:-2], utr)])
#         locs = np.array(locs3 + locs2 + locs4)

#     else:
#         locs0 = remove_overlaps([m.start() - 2 for m in re.finditer(site[4:], utr)])
#         locs1 = remove_overlaps([m.start() - 1 for m in re.finditer(site[3:-1], utr)])
#         locs2 = remove_overlaps([(m.start()) for m in re.finditer(site[2:-2], utr)])
#         locs3 = remove_overlaps([(m.start() + 1) for m in re.finditer(site[1:-3], utr)])
#         locs4 = remove_overlaps([(m.start() + 2) for m in re.finditer(site[:-4], utr)])
#         locs = np.array(locs0 + locs1 + locs2 + locs3 + locs4)

#         # locs1 = remove_overlaps([m.start() - 1 for m in re.finditer(site[3:], utr)])
#         # locs2 = remove_overlaps([(m.start()) for m in re.finditer(site[2:-1], utr)])
#         # locs3 = remove_overlaps([(m.start() + 1) for m in re.finditer(site[1:-2], utr)])
#         # locs4 = remove_overlaps([(m.start() + 2) for m in re.finditer(site[:-3], utr)])
#         # locs = np.array(locs1 + locs2 + locs3 + locs4)

#     if len(locs) == 0:
#         return []
#     elif len(locs) == 1:
#         real_locs = locs
#     else:
#         real_locs = [locs[0]]
#         for i, l in enumerate(locs[1:]):
#             if min([abs(l - rl) for rl in real_locs]) >= 7:
#                 real_locs.append(l)

#     seqs = [utr_ext[loc-4:loc+8] for loc in real_locs if (loc-4 >=0) and (loc+5 <= utr_len)]

#     return seqs

# def get_seqs_new(utr, site, only_canon=False):
#     utr_len = len(utr)
#     utr_ext = utr + 'TTT'


#     locs2 = remove_overlaps([(m.start()) for m in re.finditer(site[2:], utr)])
#     locs3 = remove_overlaps([(m.start() + 1) for m in re.finditer(site[1:-1], utr)])
#     locs4 = remove_overlaps([(m.start() + 2) for m in re.finditer(site[:-2], utr)])
#     locs = locs3 + locs2 + locs4

#     if only_canon == False:
#         locs0 = remove_overlaps([m.start() - 2 for m in re.finditer(site[4:], utr)])
#         locs1 = remove_overlaps([m.start() - 1 for m in re.finditer(site[3:-1], utr)])
#         locs2 = remove_overlaps([(m.start()) for m in re.finditer(site[2:-2], utr)])
#         locs3 = remove_overlaps([(m.start() + 1) for m in re.finditer(site[1:-3], utr)])
#         locs4 = remove_overlaps([(m.start() + 2) for m in re.finditer(site[:-4], utr)])
#         locs += (locs0 + locs1 + locs2 + locs3 + locs4)

#     locs = np.array(locs)

#     if len(locs) == 0:
#         return []
#     elif len(locs) == 1:
#         real_locs = locs
#     else:
#         real_locs = [locs[0]]
#         for i, l in enumerate(locs[1:]):
#             if min([abs(l - rl) for rl in real_locs]) >= 7:
#                 real_locs.append(l)

#     seqs = [utr_ext[loc-4:loc+8] for loc in real_locs if (loc-4 >=0) and (loc+5 <= utr_len)]

#     return seqs


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


def get_seqs_new(utr, site, overlap_dist, only_canon):
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
    if len(locs) > 1:
        locs = priority_order(locs, overlap_dist)

    utr_ext = ('TTT' + utr + 'TTT')
    seqs = [utr_ext[l:l+12] for l in locs]

    # sites already listed in order of priority, except longer sites take precedent
    return seqs


def make_pretrain_data(size, mirlen, seqlen=12):

    mirseqs = [generate_random_seq(mirlen) for _ in range(size)]
    stypes = np.random.choice(['8mer','7m8','7a1','6mer', '5mer','4mer'], size=size, replace=True)
    stype_dict = {'8mer': 6, '7m8': 4.8, '7a1': 3.6, '6mer': 2.4, '5mer': 0.7, '4mer': -0.3}
    nts = ['A','T','C','G']
    sites, upflanks, downflanks = [], [], []
    for stype, m in zip(stypes, mirseqs):
        upflanks.append(generate_random_seq(2))
        downflanks.append(generate_random_seq(2))
        template = complementary(m[-8:-1]) + 'A'
        if stype == '8mer':
            sites.append(template)
            continue
        elif stype == '7m8':
            rms = [[7]]
        elif stype == '7a1':
            rms = [[0]]
        elif stype == '6mer':
            rms = [[0,1], [0,7], [6,7]]
        elif stype == '5mer':
            rms = [[0,1,2], [0,1,7], [0,6,7], [5,6,7]]
        else:
            rms = [[0,1,2,3], [0,1,2,7], [0,1,6,7], [0,5,6,7], [4,5,6,7]]

        template = list(template)
        rm = rms[np.random.choice(np.arange(len(rms)))]
        for mut in rm:
            possibilities = [nt for nt in nts if nt != template[mut]]
            template[mut] = np.random.choice(possibilities)
        
        sites.append(''.join(template))

    flanks, seqs = [], []
    for (x, y, z) in zip(upflanks, sites, downflanks):
        flanks.append(x+z)
        seqs.append(x+y+z)

    flanking_AU = [((f.count('A') + f.count('T'))*1.6 / len(f)) - 0.8 for f in flanks]

    batch_y = np.array([stype_dict[s] for s in stypes]) + np.array(flanking_AU) + ((np.random.random(size=size) - 0.5))

    batch_x = np.zeros((size, 4*mirlen, 4*seqlen))
    for i, (mirseq, seq) in enumerate(zip(mirseqs, seqs)):
        batch_x[i, :, :] = (np.outer(one_hot_encode(mirseq, MIR_NT_DICT, TARGETS),
                                            one_hot_encode(seq, SEQ_NT_DICT, TARGETS))*4) - 0.25

    return np.expand_dims(batch_x, 3), np.expand_dims(batch_y, 1)


def make_pretrain_data_old(size, mirlen, seqlen=12):

    mirseqs = [generate_random_seq(mirlen) for _ in range(size)]
    stypes = np.random.choice(['8mer','7m8','7a1','6mer', '5mer','4mer'], size=size, replace=True)
    stype_dict = {'8mer': 6, '7m8': 4.8, '7a1': 3.6, '6mer': 2.4, '5mer': 0.7, '4mer': -0.3}
    nts = ['A','T','C','G']
    sites, upflanks, downflanks = [], [], []
    for stype, m in zip(stypes, mirseqs):
        upflanks.append(generate_random_seq(2))
        downflanks.append(generate_random_seq(2))
        template = complementary(m[-8:-1]) + 'A'
        if stype == '8mer':
            sites.append(template)
            continue
        elif stype == '7m8':
            rms = [[7]]
        elif stype == '7a1':
            rms = [[0]]
        elif stype == '6mer':
            rms = [[0,1], [0,7], [6,7]]
        elif stype == '5mer':
            rms = [[0,1,2], [0,1,7], [0,6,7], [5,6,7]]
        else:
            rms = [[0,1,2,3], [0,1,2,7], [0,1,6,7], [0,5,6,7], [4,5,6,7]]

        template = list(template)
        rm = rms[np.random.choice(np.arange(len(rms)))]
        for mut in rm:
            possibilities = [nt for nt in nts if nt != template[mut]]
            template[mut] = np.random.choice(possibilities)
        
        sites.append(''.join(template))

    flanks, seqs = [], []
    for (x, y, z) in zip(upflanks, sites, downflanks):
        flanks.append(x+z)
        seqs.append(x+y+z)

    flanking_AU = [((f.count('A') + f.count('T'))*1.6 / len(f)) - 0.8 for f in flanks]

    batch_y = np.array([stype_dict[s] for s in stypes]) + np.array(flanking_AU) + ((np.random.random(size=size) - 0.5))

    batch_x = np.zeros((size, 4*mirlen, 4*seqlen))
    for i, (mirseq, seq) in enumerate(zip(mirseqs, seqs)):
        batch_x[i, :, :] = (np.outer(one_hot_encode(mirseq, MIR_NT_DICT, TARGETS),
                                            one_hot_encode(seq, SEQ_NT_DICT, TARGETS))*4) - 0.25

    return np.expand_dims(batch_x, 3), np.expand_dims(batch_y, 1)


# def make_pretrain_data(size, mirlen, seqlen=12):

#     mirseqs = [generate_random_seq(mirlen) for _ in range(size)]
#     stypes = np.random.choice(['8mer','7m8','7a1','6mer', '5mer','4mer'], size=size, replace=True)
#     stype_dict = {'8mer': 6, '7m8': 4.8, '7a1': 3.6, '6mer': 2.4, '5mer': 0.7, '4mer': -0.3}
#     nts = ['A','T','C','G']
#     sites, upflanks, downflanks = [], [], []
#     for stype, m in zip(stypes, mirseqs):
#         if stype == '8mer':
#             sites.append(complementary(m[-8:-1]) + 'A')

#             upflanks.append(generate_random_seq(2))
#             downflanks.append(generate_random_seq(2))
#         elif stype == '7m8':
#             sites.append(complementary(m[-8:-1]))

#             upflanks.append(generate_random_seq(2))
#             downflanks.append(np.random.choice(['T','C','G']) + generate_random_seq(2))
#         elif stype == '7a1':
#             possibilities = [nt for nt in nts if nt != complementary(m[-8])]
#             sites.append(complementary(m[-7:-1]) + 'A')

#             upflanks.append(generate_random_seq(2) + np.random.choice(possibilities))
#             downflanks.append(generate_random_seq(2))
#         elif stype == '6mer':
#             possibilities = [nt for nt in nts if nt != complementary(m[-8])]
#             sites.append(complementary(m[-7:-1]))

#             upflanks.append(generate_random_seq(2) + np.random.choice(possibilities))
#             downflanks.append(np.random.choice(['T','C','G']) + generate_random_seq(2))

#         elif stype == '5mer':
#             temp_site = list(complementary(m[-7:-1]))
#             random_mutation_site = np.random.choice([0,5])
#             possibilities = [nt for nt in nts if nt != temp_site[random_mutation_site]]
#             temp_site[random_mutation_site] = np.random.choice(possibilities)

#             sites.append(''.join(temp_site))

#             upflanks.append(generate_random_seq(3))
#             downflanks.append(generate_random_seq(3))

#         else:
#             temp_site = list(complementary(m[-7:-1]))
#             which_4mer = np.random.choice([0,1])
#             if which_4mer == 0:
#                 random_mutation_sites = [0,5]
#             else:
#                 random_mutation_sites = [0,1]

#             for rms in random_mutation_sites:
#                 possibilities = [nt for nt in nts if nt != temp_site[rms]]
#                 temp_site[rms] = np.random.choice(possibilities)

#             sites.append(''.join(temp_site))

#             upflanks.append(generate_random_seq(3))
#             downflanks.append(generate_random_seq(3))

#     flanks, seqs = [], []
#     for (x, y, z) in zip(upflanks, sites, downflanks):
#         flanks.append(x+z)
#         seqs.append(x+y+z)

#     flanking_AU = [((f.count('A') + f.count('T'))*1.6 / len(f)) - 0.8 for f in flanks]

#     batch_y = np.array([stype_dict[s] for s in stypes]) + np.array(flanking_AU) + ((np.random.random(size=size) - 0.5))

#     batch_x = np.zeros((size, 4*mirlen, 4*seqlen))
#     for i, (mirseq, seq) in enumerate(zip(mirseqs, seqs)):
#         batch_x[i, :, :] = make_square(mirseq, seq)

#     return np.expand_dims(batch_x, 3), np.expand_dims(batch_y, 1)


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