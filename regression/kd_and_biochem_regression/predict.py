from optparse import OptionParser
import os
import pickle
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
        print(seqs)
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
    parser.add_option("-m", "--mir", dest="MIR", help="left_out_mir")
    parser.add_option("-l", "--logdir", dest="LOGDIR", help="directory for writing logs")
    parser.add_option("-o", "--outdir", dest="OUTDIR", help="output file")

    (options, args) = parser.parse_args()

    PRETRAIN_SAVE_PATH = os.path.join(options.LOGDIR, 'pretrain_saved')
    SAVE_PATH = os.path.join(options.LOGDIR, 'saved')

    if not os.path.isdir(options.LOGDIR):
        os.makedirs(options.LOGDIR)

    ### READ EXPRESSION DATA ###
    tpm = pd.read_csv(options.TPM_FILE, sep='\t', index_col=0)

    MIRS = [x for x in tpm.columns if ('mir' in x) or ('lsy' in x)]

    all_utrs = tpm['sequence'].values
    batch_size = 50
    output_dict = {}

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        latest = tf.train.latest_checkpoint('{}/{}/saved'.format(options.LOGDIR, options.MIR))
        # print(latest)
        saver = tf.train.import_meta_graph(latest + '.meta')
        saver.restore(sess, latest)

        _keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')
        _phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        _combined_x = tf.get_default_graph().get_tensor_by_name('biochem_x:0')
        _prediction = tf.get_default_graph().get_tensor_by_name('final_layer/pred_kd:0')
        # _freeAGO_all_trainable = tf.get_default_graph().get_tensor_by_name('freeAGO_all_trainable:0')
        # _decay_trainable = tf.get_default_graph().get_tensor_by_name('decay_trainable:0')
        # _utr_coef_trainable = tf.get_default_graph().get_tensor_by_name('utr_coef_trainable:0')
        # _freeAGO_let7_trainable = tf.get_default_graph().get_tensor_by_name('freeAGO_let7_trainable:0')

        # current_freeAGO = sess.run(_freeAGO_all_trainable)
        # current_freeAGO_let7 = sess.run(_freeAGO_let7_trainable)
        # current_decay = sess.run(_decay_trainable)
        # current_utr_coef = sess.run(_utr_coef_trainable)

        # current_freeAGO = current_freeAGO.reshape([NUM_TRAIN, 2])
        # freeAGO_df = pd.DataFrame({'mir': train_mirs,
        #                            'guide': current_freeAGO[:, 0],
        #                            'passenger': current_freeAGO[:, 1]})

        # freeAGO_df.to_csv(os.path.join(options.OUTDIR, 'freeAGO_final_{}.txt'.format(options.MIR)), sep='\t', index=False)
        # with open(os.path.join(options.OUTDIR, 'fitted_params_{}.txt'.format(options.MIR)), 'w') as outfile:
        #     outfile.write('freeAGO_let7\t{}\n'.format(current_freeAGO_let7.flatten()[0]))
        #     outfile.write('decay\t{}\n'.format(current_decay))
        #     outfile.write('utr_coef\t{}\n'.format(current_utr_coef))

        # print(current_utr_coef)
        for mir in MIRS:
            current_ix = 0
            last_batch = False
            output_dict[mir] = {'KDs': [], 'num_seqs': []}
            output_dict[mir+'*'] = {'KDs': [], 'num_seqs': []}

            while True:
                if (current_ix + batch_size) >= len(all_utrs):
                    last_batch = True
                    current_utrs = all_utrs[current_ix:]
                else:
                    current_utrs = all_utrs[current_ix: current_ix + batch_size]
                    current_ix += batch_size

                combined_x_guide, utr_num_seqs_guide = get_features(mir, current_utrs, config.MIRLEN, config.SEQLEN)
                combined_x_pass, utr_num_seqs_pass = get_features(mir + '*', current_utrs, config.MIRLEN, config.SEQLEN)

                feed_dict = {
                                _keep_prob: 1.0,
                                _phase_train: False,
                                _combined_x: combined_x_guide
                            }

                pred_guide = sess.run(_prediction, feed_dict=feed_dict)
                output_dict[mir]['KDs'] += list(pred_guide.flatten())
                output_dict[mir]['num_seqs'] += utr_num_seqs_guide

                feed_dict = {
                                _keep_prob: 1.0,
                                _phase_train: False,
                                _combined_x: combined_x_pass
                            }

                pred_pass = sess.run(_prediction, feed_dict=feed_dict)
                output_dict[mir+'*']['KDs'] += list(pred_pass.flatten())
                output_dict[mir+'*']['num_seqs'] += utr_num_seqs_pass

                if last_batch:
                    break

    with open(os.path.join(options.OUTDIR, '{}.pickle'.format(options.MIR)), 'wb') as outfile:
        pickle.dump(output_dict, outfile)



