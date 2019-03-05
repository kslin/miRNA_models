import numpy as np
import pandas as pd
from scipy import stats


### functions from Namita
import subprocess

def GetdG_dotbracket(seq1, seq2):
    # output_ = subprocess.check_output('echo \'' + seq +'\' | RNAfold -d 0', shell=True).split('\n')
    out = subprocess.check_output('echo \'' + seq1 +'&' + seq2 + '\' | RNAcofold -d2',  shell=True)
    dG = float(out[-8:-2])
    dotbracket =  out.split()[1]
    return dG, dotbracket

def threepscore_NB(mirnaseq, UTR, index):
    #index is where the seed pairing starts 
    #5' to 3'

    #window to search for 3p pairing, start 8 nucleotides away and search a total of a looplen up to 12nt + 9mer of pairing = 21nt total
    window = 21
    index_start = max(0, index-8-window)
    index_end = min(index-8, len(UTR))
    UTRsearch = UTR[index_start:index_end]
    if len(UTRsearch) < 4:
        return 0.0, 0, 0

    dG, dotbracket = GetdG_dotbracket(mirnaseq[8:], UTRsearch)
    dotbracket = dotbracket.decode()
    try:
        dotbracket_mirna = dotbracket.split('&')[0]
        dotbracket_target = dotbracket.split('&')[1]

    except:
        print(UTRsearch, index_start, index_end, mirnaseq[8:], dotbracket)
        raise ValueError()

    #need atleast 3 predicted basepairs
    if dotbracket_target.count(')') > 3:
        looplen = dotbracket_target[::-1].find(')')
        register = dotbracket_mirna.find('(') + 9
        dGout = dG
    else:
        looplen = 0
        dGout = 0
        register = 0
    

    return dGout, looplen, register

def test_threepscore_NB():

    mirnaseq = 'UGAGGUAGUAGGUUGUAUAGUU'
    index = 41
    #A good 3p site: "UACAACC-N5-CCACCUCA" 5p to 3p
    # UTR = 'AGCUGAGUCGUAGCUGAUCGA UACAACC AAGUU CCACCUCA AUUGUAUGUCU'
    #5' to 3'
    UTR = 'AGCUGAGUCGUAGCUGAUCGAUACAACCAAGUUCCACCUCAAUUGUAUGUCU'

    dG, looplen, register = threepscore_NB(mirnaseq,UTR, index)

    print(dG, looplen, register)

# test_threepscore_NB()

### end functions from Namita


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


def calc_r2(xs, ys):
    return stats.linregress(xs.flatten(), ys.flatten())[2]**2


def get_nsites(features):
    nsites = features.reset_index()
    nsites['nsites'] = 1
    nsites = nsites.groupby(['transcript','mir']).agg({'nsites': np.sum})
    return nsites


def sigmoid(vals):
    return 1.0 / (1.0 + np.exp(-1 * vals))


def rev_comp(seq):
    """ Get reverse complement of sequence"""

    match_dict = {'A': 'T',
                  'T': 'A',
                  'C': 'G',
                  'G': 'C'}

    return ''.join([match_dict[x] for x in seq][::-1])


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


def norm_matrix(mat):
    means = np.mean(mat, axis=1).reshape([-1, 1])
    return mat - means


def get_r2_unnormed(preds, labels):
    preds_normed = norm_matrix(preds)
    labels_normed = norm_matrix(labels)
    return calc_r2(preds_normed, labels_normed)
