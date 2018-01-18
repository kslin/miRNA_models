import os
import shlex
import subprocess

import numpy as np
import pandas as pd
import tensorflow as tf


def generate_random_seq(length):
    return ''.join(np.random.choice(['A','T','C','G'], size=length))


def one_hot_encode_nt(seq, nt_order):
    """Convert RNA sequence to one-hot encoding"""
    
    one_hot = [list(np.array(nt_order == nt, dtype=int)) for nt in seq]
    # one_hot = [item for sublist in one_hot for item in sublist]
    
    return np.array(one_hot)


def rev_comp(seq):
    match_dict = {'A': 'T',
                  'T': 'A',
                  'C': 'G',
                  'G': 'C'}

    return ''.join([match_dict[x] for x in seq])[::-1]


def weight_variable(shape, name=None):
    var = tf.get_variable(
                            name="name",
                            initializer=tf.random_normal_initializer(stddev=0.1),
                            shape=shape
    )
    return var


def bias_variable(shape, name=None):
    var = tf.get_variable(
                            name=name,
                            initializer=tf.constant_initializer(0.1),
                            shape=shape
    )
    return var


def get_rnaplfold_data(gene, utr, temp_folder, fold_len):
    """
    Run RNAplfold and get pairing probabilities for a utr

    Parameters
    ----------
    gene: string, name of gene

    utr: string, utr sequence

    Output
    ------
    pandas DataFrame: pairing probabilities at each position
    """

    # sanitize name of file so we don't break the shell
    gene_name = shlex.split(gene)[0]

    # navigate to the folder for RNAplfold data
    cwd = os.getcwd()
    os.chdir(temp_folder)

    # write sequence to a temporary file
    mytempfile = os.path.join(temp_folder, 'temp_{}.fa'.format(gene_name))
    with open(mytempfile, 'w') as f:
        f.write('>{}\n{}'.format(gene_name, utr))

    # call RNAplfold
    length = min(40, len(utr))
    window = min(80, len(utr))
    mycall = 'RNAplfold -L {} -W {} -u {} < {}'.format(length, window, fold_len,
                                                       mytempfile)
    subprocess.call([mycall], shell=True, stdout=subprocess.PIPE)
    lunp_file = '{}_lunp'.format(gene_name)

    # read data and convert to a dataframe
    rnaplfold_data = pd.read_csv(lunp_file, sep='\t', skiprows=2,
                                 header=None).set_index(0)

    os.remove(mytempfile)
    os.remove(lunp_file)
    os.remove('{}_dp.ps'.format(gene_name))

    os.chdir(cwd)

    return rnaplfold_data