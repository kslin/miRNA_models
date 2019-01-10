import os
import pickle
import sys

import numpy as np
import pandas as pd

import helpers, config, predict_helpers


shuffle_num = int(sys.argv[1])
SHUFFLE_ORDER_DIR = '/lab/bartel4_ata/kathyl/RNA_Seq/analysis/data/no_baseline_analysis/convnet_shuffle'
with open(os.path.join(SHUFFLE_ORDER_DIR, 'shuffle_orders_{}.pickle'.format(shuffle_num)), 'rb') as infile:
    SHUFFLE_ORDER = pickle.load(infile)

PREDICT_DIR = '/lab/bartel4_ata/kathyl/RNA_Seq/analysis/data/final/predictions_simple_xval_4_16_16_with_seqs/'
NEW_PREDICT_DIR = '/lab/bartel4_ata/kathyl/RNA_Seq/analysis/data/final/predictions_simple_xval_4_16_16_shuffled_noncanon_temp_{}/'.format(shuffle_num)
NEW_LOG_DIR = '/lab/bartel4_ata/kathyl/RNA_Seq/analysis/data/no_baseline_analysis/model_preds_shuffled_noncanon_retrain_temp_{}/'.format(shuffle_num)
# BASELINES_FILE = '/lab/bartel4_ata/kathyl/RNA_Seq/analysis/data/no_baseline_analysis/convnet_baselines.pickle'
PRED_DF_DIR = '/lab/bartel4_ata/kathyl/RNA_Seq/analysis/data/final/predictions_simple_xval_4_16_16_shuffled_pred_dfs'

TEST_MIRS = sorted(['mir153','mir139','mir144','mir223','mir137',
             'mir205','mir143','mir182','mir199a','mir204','mir216b'])

MIRS5 = ['mir1','mir124','mir155','mir7','lsy6']
ALL_MIRS = MIRS5 + TEST_MIRS
MERGED = pd.read_csv('/lab/bartel4_ata/kathyl/RNA_Seq/analysis/data/no_baseline_analysis/merged.txt', sep='\t', index_col=0)

if not os.path.isdir(NEW_PREDICT_DIR):
    os.makedirs(NEW_PREDICT_DIR)

if not os.path.isdir(NEW_LOG_DIR):
    os.makedirs(NEW_LOG_DIR)

if not os.path.isdir(PRED_DF_DIR):
    os.makedirs(PRED_DF_DIR)

num_mirs = len(ALL_MIRS)
num_genes = len(MERGED)
num_kds_guide = 362
num_kds_pass = 418
input_utr_len = MERGED[['utr_length']].values

for shuffle_ix in range(len(SHUFFLE_ORDER['mir137'])):

    for test_ix, test_mir in enumerate(TEST_MIRS):
        site8 = config.SITE_DICT[test_mir]
        next_mir = SHUFFLE_ORDER[test_mir][shuffle_ix][0]
        site8_next = config.SITE_DICT[next_mir]
        other_mirs = [m for m in ALL_MIRS if m != test_mir]
        other_shuffles = SHUFFLE_ORDER[test_mir][shuffle_ix][1]
        
        output_dict = {}
        x_guide, mask_guide = np.zeros([num_genes, num_mirs-1, num_kds_guide]), np.zeros([num_genes, num_mirs-1, num_kds_guide])
        x_pass, mask_pass = np.zeros([num_genes, num_mirs-1, num_kds_pass]), np.zeros([num_genes, num_mirs-1, num_kds_pass])
        y_vals = MERGED[other_mirs].values

        with open(os.path.join(PREDICT_DIR, '{}.pickle'.format(test_mir)), 'rb') as infile:
            info = pickle.load(infile)
        
        with open(os.path.join(PREDICT_DIR, '{}.pickle'.format(next_mir)), 'rb') as infile:
            info_next = pickle.load(infile)
            
        kds_guide, num_seqs_guide = predict_helpers.get_shuffled_data(info[test_mir], info_next[next_mir], site8, site8_next)
        kds_pass, num_seqs_pass = predict_helpers.get_shuffled_data(info[test_mir+"*"], info_next[next_mir+"*"], site8, site8_next)
        output_dict[test_mir] = {'KDs': [-1*x for x in kds_guide], 'num_seqs': num_seqs_guide}
        output_dict[test_mir+'*'] = {'KDs': [-1*x for x in kds_pass], 'num_seqs': num_seqs_pass}
            
        for this_mir_ix, this_mir in enumerate(other_mirs):
            site8 = config.SITE_DICT[this_mir]
            next_mir = other_shuffles[this_mir_ix]
            assert (next_mir != this_mir)
            assert (next_mir != test_mir)
            site8_next = config.SITE_DICT[next_mir]
            
            kds_guide, num_seqs_guide = predict_helpers.get_shuffled_data(info[this_mir], info[next_mir], site8, site8_next)
            kds_pass, num_seqs_pass = predict_helpers.get_shuffled_data(info[this_mir+"*"], info[next_mir+"*"], site8, site8_next)
            
            current_ix = 0
            for ix, num_seq in enumerate(num_seqs_guide):
                x_guide[ix, this_mir_ix, :num_seq] = kds_guide[current_ix: current_ix + num_seq]
                mask_guide[ix, this_mir_ix, :num_seq] = 1
                current_ix += num_seq
                
            current_ix = 0
            for ix, num_seq in enumerate(num_seqs_pass):
                x_pass[ix, this_mir_ix, :num_seq] = kds_pass[current_ix: current_ix + num_seq]
                mask_pass[ix, this_mir_ix, :num_seq] = 1
                current_ix += num_seq
            
            output_dict[this_mir] = {'KDs': [-1*x for x in kds_guide], 'num_seqs': num_seqs_guide}
            output_dict[this_mir+'*'] = {'KDs': [-1*x for x in kds_pass], 'num_seqs': num_seqs_pass}
            
        with open(os.path.join(NEW_PREDICT_DIR, '{}.pickle'.format(test_mir)), 'wb') as outfile:
            pickle.dump(output_dict, outfile)
                
        params = predict_helpers.refit_last_layer(x_guide, x_pass,
                                  mask_guide, mask_pass,
                                  y_vals, input_utr_len, num_genes, num_mirs-1, maxiter=200)
        freeAGO_df = pd.DataFrame({'mir': other_mirs,
                                   'guide': params[2].flatten(),
                                   'passenger': params[3].flatten()})

        log_dir_mir = os.path.join(NEW_LOG_DIR, test_mir)
        if not os.path.isdir(log_dir_mir):
            os.makedirs(log_dir_mir)
        freeAGO_df.to_csv(os.path.join(log_dir_mir, 'freeAGO_final.txt'), sep='\t', index=False)
        with open(os.path.join(log_dir_mir, 'fitted_params.txt'), 'w') as outfile:
            outfile.write('decay\t{}\n'.format(np.log(params[1])))
            outfile.write('utr_coef\t{}\n'.format(params[0]))
            print(params[0], params[1])

    shuffle_results = predict_helpers.get_pred_df(NEW_LOG_DIR, NEW_PREDICT_DIR, MERGED, TEST_MIRS, ALL_MIRS, None)
    shuffle_results[0].to_csv(os.path.join(PRED_DF_DIR, '{}_{}.txt'.format(shuffle_num, shuffle_ix)), sep='\t', index=False)

# sns.set_style('ticks')
# sns.set_context('poster')
# OUTFILE = '/lab/bartel4_ata/kathyl/RNA_Seq/analysis/figures/new_analysis/nn_shuffled_noncanon_{}.png'.format(offset)
# shuffled_noncanon = plot_results(NEW_LOG_DIR, NEW_PREDICT_DIR, MERGED, TEST_MIRS, ALL_MIRS, OUTFILE, baselines_dict)
# print(offset, stats.linregress(shuffled_noncanon[1].flatten(), shuffled_noncanon[2].flatten())[2]**2)
# print(shuffled_noncanon[0])
