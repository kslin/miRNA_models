import json
from optparse import OptionParser
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import tensorflow as tf

import config, helpers, data_objects_endog

np.set_printoptions(threshold=np.inf, linewidth=200)
pd.options.mode.chained_assignment = None

def get_features(mir, utrs, mirlen, seqlen):
    all_seqs = []
    mirseq_one_hot = config.ONE_HOT_DICT[mir]
    site = config.SITE_DICT[mir]

    utr_num_seqs = []
    for utr in utrs:
        seqs = helpers.get_seqs(utr, site)
        all_seqs.append(seqs)
        utr_num_seqs.append(len(seqs))

    combined_x = np.zeros([np.sum(utr_num_seqs), 4*mirlen, 4*seqlen])
    
    current_ix = 0
    for seq_list in all_seqs:
        if len(seq_list) > 0:
            for seq in seq_list:
                temp = np.outer(mirseq_one_hot, helpers.one_hot_encode(seq, config.SEQ_NT_DICT, config.TARGETS))
                combined_x[current_ix, :, :] = temp
                current_ix += 1
    
    combined_x = np.expand_dims((combined_x * 4) - 0.25, 3)
    
    return combined_x, utr_num_seqs


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-t", "--tpmfile", dest="TPM_FILE", help="tpm data")
    parser.add_option("-m", "--mir", dest="MIR", help="tpm data")
    parser.add_option("-l", "--logdir", dest="LOGDIR", help="directory for writing logs")
    parser.add_option("-o", "--outfile", dest="OUTFILE", help="output file")

    (options, args) = parser.parse_args()

    PRETRAIN_SAVE_PATH = os.path.join(options.LOGDIR, 'pretrain_saved')
    SAVE_PATH = os.path.join(options.LOGDIR, 'saved')

    if not os.path.isdir(options.LOGDIR):
        os.makedirs(options.LOGDIR)

    ### READ PREFIT DATA ###
    prefit = pd.read_csv(options.PREFIT, sep='\t', index_col=0)

    ### READ let-7 sites ###
    let7_sites = pd.read_csv(options.LET7_SITES, sep='\t', index_col=0)
    let7_mask = pd.read_csv(options.LET7_MASK, sep='\t', index_col=0)
    let7_num_kds = len(let7_sites.columns)

    ### READ EXPRESSION DATA ###
    tpm = pd.read_csv(options.TPM_FILE, sep='\t', index_col=0)

    assert(options.MIR in tpm.columns)

    all_utrs = tpm['sequence'].values
    batch_size = 50
    current_ix = 0
    last_batch = False
    output_dict = {options.MIR: {'KDs': [], 'num_seqs': []}, options.MIR+'*': {'KDs': [], 'num_seqs': []}}
    starting_ix_guide = 0
    starting_ix_pass = 0

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        latest = tf.train.latest_checkpoint('{}/{}/saved'.format(LOGDIR, mir))

        saver = tf.train.import_meta_graph(latest + '.meta')
        saver.restore(sess, latest)

        _keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')
        _phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        _combined_x = tf.get_default_graph().get_tensor_by_name('biochem_x:0')
        _prediction = tf.get_default_graph().get_tensor_by_name('final_layer/add:0')

        while True:
            if (current_ix + batch_size) >= len(all_utrs):
                last_batch = True
                current_utrs = utrs[current_ix:]
            else:
                current_utrs = utrs[current_ix: current_ix + batch_size]
                current_ix += batch_size

            combined_x_guide, utr_num_seqs_guide = get_features(options.MIR, utrs, config.MIRLEN, config.SEQLEN)
            combined_x_pass, utr_num_seqs_pass = get_features(options.MIR + '*', utrs, config.MIRLEN, config.SEQLEN)

            feed_dict = {
                            _keep_prob: 1.0,
                            _phase_train: False,
                            _combined_x: combined_x_guide
                        }

            pred_guide = sess.run(_prediction, feed_dict=feed_dict) * config.NORM_RATIO
            output_dict[options.MIR]['KDs'] += list(pred_guide.flatten())
            output_dict[options.MIR]['num_seqs'] += utr_num_seqs_guide

            feed_dict = {
                            _keep_prob: 1.0,
                            _phase_train: False,
                            _combined_x: combined_x_pass
                        }

            pred_pass = sess.run(_prediction, feed_dict=feed_dict) * config.NORM_RATIO
            output_dict[options.MIR+'*']['KDs'] += list(pred_pass.flatten())
            output_dict[options.MIR+'*']['num_seqs'] += utr_num_seqs_pass

    with open(options.OUTFILE, 'w') as outfile:
        json.dump(output_dict, outfile)


