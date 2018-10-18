import os
import pickle
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import tensorflow as tf

import config

from matplotlib.colorbar import ColorbarBase
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap


def make_sylamer_files(df, mir, nosite, outdir):
    temp = df[[mir, nosite, 'sequence']].dropna()
    temp['logfc'] = temp[mir] - temp[nosite]
    temp = temp.sort_values('logfc')
    
    with open(os.path.join(outdir, 'utrs.fa'.format(mir)), 'w') as outfile1:
        with open(os.path.join(outdir, '{}_universe.txt'.format(mir)), 'w') as outfile2:
            for row in temp.iterrows():
                outfile1.write('>{}\n{}\n'.format(row[0], row[1]['sequence']))
                outfile2.write('{}\n'.format(row[0]))
                
#     with open(os.path.join(outdir, '{}_universe.txt'.format(mir)), 'w') as outfile3:
#         for gene in temp.index:
#             outfile3.write('{}\n'.format(gene))


def bayesian_slope(xs, ys, mu_0, prior_strength):
    xs = xs - np.mean(xs)
    ys = ys - np.mean(ys)

    slope = stats.linregress(xs, ys)[0]
    return (1 / (np.sum(xs**2) + prior_strength)) * ((np.sum(xs**2) * slope) + (mu_0 * prior_strength))

def get_bayes_intercept(xs, ys, mu_0, prior_strength):
    
    intercepts = []
    for ix in range(xs.shape[0]):
        temp_x = xs[ix,:]
        temp_y = ys[ix,:]
        if np.mean(temp_x) == 0:
            intercepts.append(np.mean(temp_y))
        else:
            b_slope = bayesian_slope(temp_x, temp_y, mu_0, prior_strength)
            intercepts.append(np.mean(temp_y) - (b_slope * np.mean(temp_x)))
        
    return np.array(intercepts)

def discrete_colorbar(ax, cmap, num_colors, orientation):
    cmaplist = [cmap(i*int(np.floor(256/num_colors))) for i in range(num_colors)][::-1]
    new_cmap = LinearSegmentedColormap.from_list('Custom cmap', cmaplist, num_colors)
    bounds = np.arange(num_colors+1)
    ticks = np.arange(num_colors) + 0.5

    cbar = ColorbarBase(ax, cmap=new_cmap, norm=colors.BoundaryNorm(bounds, num_colors),
                 spacing='proportional', ticks=ticks, boundaries=bounds, format='%1i', orientation=orientation)
    
    if orientation == 'horizontal':
        cbar.ax.set_xticklabels([str(x) for x in bounds[:-2]] + [r'$\geqslant$ {}'.format(bounds[-2])], fontsize=18)
    else:
        cbar.ax.set_yticklabels([str(x) for x in bounds[:-2]] + [r'$\geqslant$ {}'.format(bounds[-2])], fontsize=18)
    
    return new_cmap

def count_num_canon(utr, sitem8):
    # Six canonical sites, remove double-counted sites
    num_6m8 = utr.count(sitem8[:-1])
    num_6a1 = utr.count(sitem8[2:] + 'A')
    num_6 = utr.count(sitem8[1:])
    num_7m8 = utr.count(sitem8)
    num_7a1 = utr.count(sitem8[1:] + 'A')
    num_8 = utr.count(sitem8 + 'A')
    return num_6m8 + num_6a1 + num_6 - num_7m8 - num_7a1

def calc_nbound(kds, num_seqs, freeAgo):
    nbound_ind = 1 / (1 + np.exp(kds - freeAgo))
    nbound = np.zeros(len(num_seqs))
    current_ix = 0
    for ix, num_seq in enumerate(num_seqs):
        nbound[ix] = np.sum(nbound_ind[current_ix: current_ix + num_seq])
        current_ix += num_seq
        
    assert(current_ix == len(kds))
    return nbound

def calc_rsquared(pred, actual):
    SS_err = np.sum((pred-actual)**2)
    y_bar = np.mean(actual)
    SS_tot = np.sum((y_bar-actual)**2)
    
    return 1 - (SS_err/SS_tot)


def get_logfc_simple(kds_guide, num_seqs_guide, kds_pass, num_seqs_pass, utr_len, params):
    nbound_guide = calc_nbound(kds_guide, num_seqs_guide, params['freeAgo_guide'])
    nbound_pass = calc_nbound(kds_pass, num_seqs_pass, params['freeAgo_pass'])
    nbound_init_utr = utr_len * np.exp(params['utr_coef'])
    
    pred_init = np.log1p((nbound_init_utr) * np.exp(params['decay']))
    pred = np.log1p((nbound_guide + nbound_pass + nbound_init_utr) * np.exp(params['decay']))
    
    return -1 * (pred - pred_init)

def has_canon(seq, site8):
    return ((site8[:-2] in seq) or (site8[1:-1] in seq) or (site8[2:] in seq))

def get_shuffled_data(pred_dict, pred_dict_next, site8, site8_next):
    kds = -1 * np.array(pred_dict['KDs'])
    seqs = pred_dict['seqs']
    num_seqs = pred_dict['num_seqs']
    
    kds_next = -1 * np.array(pred_dict_next['KDs'])
    seqs_next = pred_dict_next['seqs']
    num_seqs_next = pred_dict_next['num_seqs']
    
    current_ix1, current_ix2 = 0,0
    new_kds, new_num_seqs = [], []
    for ns1,ns2 in zip(num_seqs, num_seqs_next):
        temp_kds1 = kds[current_ix1: current_ix1 + ns1]
        temp_seqs1 = seqs[current_ix1: current_ix1 + ns1]
        
        temp_kds2 = kds_next[current_ix2: current_ix2 + ns2]
        temp_seqs2 = seqs_next[current_ix2: current_ix2 + ns2]
        
        new_canon = [k for (k,s) in zip(temp_kds1, temp_seqs1) if has_canon(s,site8)]
        new_noncanon = [k for (k,s) in zip(temp_kds2, temp_seqs2) if not (has_canon(s,site8) or has_canon(s, site8_next))]
        new_kds += new_canon + new_noncanon
        new_num_seqs.append(len(new_canon) + len(new_noncanon))
        
        current_ix1 += ns1
        current_ix2 += ns2
    return new_kds, new_num_seqs


def refit_last_layer(kds_guide, kds_pass, mask_guide, mask_pass, ys, input_utr_len, num_genes, num_mirs, maxiter=200):
    tf.reset_default_graph()

    with tf.Session() as sess:
        x_3d_guide = tf.placeholder(tf.float32, shape=[num_genes, num_mirs, None], name='log_kds_guide')
        x_3d_pass = tf.placeholder(tf.float32, shape=[num_genes, num_mirs, None], name='log_kds_pass')
        mask_3d_guide = tf.placeholder(tf.float32, shape=[num_genes, num_mirs, None], name='mask_guide')
        mask_3d_pass = tf.placeholder(tf.float32, shape=[num_genes, num_mirs, None], name='mask_pass')

        y_all = tf.placeholder(tf.float32, shape=[num_genes, num_mirs], name='y')
        utr_len = tf.placeholder(tf.float32, shape=[num_genes, 1], name='utr_len')

        decay = tf.exp(tf.get_variable('decay', shape=(), initializer=tf.constant_initializer(0.0)))
        utr_coef = tf.get_variable('utr_coef', shape=(), initializer=tf.constant_initializer(-8.5))
        freeAgo_guide = tf.get_variable('f_guide', shape=[1,num_mirs,1], initializer=tf.constant_initializer(-5.0))
        freeAgo_pass = tf.get_variable('f_pass', shape=[1,num_mirs,1], initializer=tf.constant_initializer(-6.0))
        
        nbound_guide = tf.reduce_sum(tf.multiply(tf.sigmoid(freeAgo_guide - x_3d_guide), mask_3d_guide), axis=2)
        nbound_pass = tf.reduce_sum(tf.multiply(tf.sigmoid(freeAgo_pass - x_3d_pass), mask_3d_pass), axis=2)
        nbound_init_utr = utr_len * tf.exp(utr_coef)

        nbound_all = nbound_guide + nbound_pass + nbound_init_utr

        pred_init = tf.log1p(nbound_init_utr * decay)
        pred = -1 * (tf.log1p(nbound_all * decay) - pred_init)
        
        pred_mean = tf.reduce_mean(pred, axis=1)
        y_mean = tf.reduce_mean(y_all, axis=1)

        intercept = y_mean - pred_mean
        y_normed = y_all - tf.reshape(intercept, [-1,1])

        loss = tf.nn.l2_loss((pred - y_normed))
        train_step = tf.train.AdamOptimizer(0.1).minimize(loss)
            

        sess.run(tf.global_variables_initializer())

        feed_dict = {x_3d_guide: kds_guide,
                     x_3d_pass: kds_pass,
                     mask_3d_guide: mask_guide,
                     mask_3d_pass: mask_pass,
                     utr_len: input_utr_len,
                     y_all: ys
                    }

        losses = []
        prev = 0
        for i in range(maxiter):
            train_loss, _ = sess.run([loss, train_step], feed_dict=feed_dict)


            train_loss = (train_loss / num_genes)
            losses.append(train_loss)

            prev = train_loss
            
        return sess.run([utr_coef, decay, freeAgo_guide, freeAgo_pass])


def get_training_pred_df(test_mir, other_mirs, log_dir, predict_dir, merged, input_intercept_dict=None):
    freeago = pd.read_csv('{}/{}/freeAGO_final.txt'.format(log_dir, test_mir), sep='\t').set_index('mir')
    utr_list = list(merged['sequence'].values)
    utr_len = merged['utr_length'].values
    paramfile = pd.read_csv('{}/{}/fitted_params.txt'.format(log_dir, test_mir), sep='\t', index_col=0, header=None)

    all_preds, all_tpms = np.zeros([len(merged), len(other_mirs)]), np.zeros([len(merged), len(other_mirs)])
    with open(os.path.join(predict_dir, '{}.pickle'.format(test_mir)), 'rb') as infile:
    #         print(infile)
        preds = pickle.load(infile)
        for ix, mir in enumerate(other_mirs):
            pred_guide = preds[mir]
            kds_guide = -1 * np.array(pred_guide['KDs'])
            num_seqs_guide = pred_guide['num_seqs']

            pred_pass = preds[mir+'*']
            kds_pass = -1 * np.array(pred_pass['KDs'])
            num_seqs_pass = pred_pass['num_seqs']

            params = {'freeAgo_guide': freeago.loc[mir]['guide'],
                          'freeAgo_pass': freeago.loc[mir]['passenger'],
                          'utr_coef': paramfile.loc['utr_coef'][1],
                          'decay': paramfile.loc['decay'][1]}

            logfc = get_logfc_simple(kds_guide, num_seqs_guide, kds_pass, num_seqs_pass, utr_len, params)

            all_preds[:, ix] = logfc
            all_tpms[:, ix] = merged[mir].values

        all_preds_df = pd.DataFrame(all_preds, columns=other_mirs).set_index(merged.index)
        all_tpms_df = pd.DataFrame(all_tpms, columns=other_mirs).set_index(merged.index)
    
    return all_preds_df, all_tpms_df


def get_pred_df(log_dir, predict_dir, merged, test_mirs, all_mirs, input_intercept_dict=None):
    freeAgo_guide, freeAgo_pass = [], []
    for tm in test_mirs:
        try:
            temp = pd.read_csv('{}/{}/freeAGO_final.txt'.format(log_dir, tm), sep='\t').set_index('mir')
        except:
            continue
        freeAgo_guide.append(temp[['guide']].rename(columns={'guide': tm}))
        freeAgo_pass.append(temp[['passenger']].rename(columns={'passenger': tm}))

    freeAgo_guide = pd.concat(freeAgo_guide, axis=1, join='outer')
    freeAgo_pass = pd.concat(freeAgo_pass, axis=1, join='outer')
    vals = freeAgo_guide.values
    freeAgo_guide['mean'], freeAgo_guide['std'] = np.nanmean(vals, axis=1), np.nanstd(vals, axis=1)
    vals = freeAgo_pass.values
    freeAgo_pass['mean'], freeAgo_pass['std'] = np.nanmean(vals, axis=1), np.nanstd(vals, axis=1)

    intercept_dict = {}
    utr_list = list(merged['sequence'].values)
    r2_dict_bayes = {}
    pred_df = []

    for IX, TEST_MIR in enumerate(test_mirs):
        utr_len = merged['utr_length'].values
        try:
            paramfile = pd.read_csv('{}/{}/fitted_params.txt'.format(log_dir, TEST_MIR), sep='\t', index_col=0, header=None)
        except:
            continue

        all_preds, all_tpms = np.zeros([len(merged), len(all_mirs)]), np.zeros([len(merged), len(all_mirs)])
        with open(os.path.join(predict_dir, '{}.pickle'.format(TEST_MIR)), 'rb') as infile:
    #         print(infile)
            preds = pickle.load(infile)
            for ix, mir in enumerate(all_mirs):
                pred_guide = preds[mir]
                kds_guide = -1 * np.array(pred_guide['KDs'])
                num_seqs_guide = pred_guide['num_seqs']

                pred_pass = preds[mir+'*']
                kds_pass = -1 * np.array(pred_pass['KDs'])
                num_seqs_pass = pred_pass['num_seqs']

                if mir == TEST_MIR:
                    params = {'freeAgo_guide': freeAgo_guide.loc[TEST_MIR]['mean'],
                              'freeAgo_pass': freeAgo_pass.loc[TEST_MIR]['mean'],
                              'utr_coef': paramfile.loc['utr_coef'][1],
                              'decay': paramfile.loc['decay'][1]}
                else:
                    params = {'freeAgo_guide': freeAgo_guide.loc[mir][TEST_MIR],
                              'freeAgo_pass': freeAgo_pass.loc[mir][TEST_MIR],
                              'utr_coef': paramfile.loc['utr_coef'][1],
                              'decay': paramfile.loc['decay'][1]}

                logfc = get_logfc_simple(kds_guide, num_seqs_guide, kds_pass, num_seqs_pass, utr_len, params)

                all_preds[:, ix] = logfc
                all_tpms[:, ix] = merged[mir].values

        all_preds_df = pd.DataFrame(all_preds, columns=all_mirs).set_index(merged.index)
        all_tpms_df = pd.DataFrame(all_tpms, columns=all_mirs).set_index(merged.index)

        other_mirs = [m for m in all_mirs if m != TEST_MIR]
        if input_intercept_dict is None:
            intercept = get_bayes_intercept(all_preds_df[other_mirs].values, all_tpms_df[other_mirs].values, 0, 0.01)
        else:
            intercept = input_intercept_dict[TEST_MIR]
        intercept_dict[TEST_MIR] = intercept
        pred = all_preds_df[TEST_MIR].values / np.log(2)
        tpms = all_tpms_df[TEST_MIR] / np.log(2)
        log2_intercept = intercept / np.log(2)
        actual = tpms - log2_intercept
        sitem8 = config.SITE_DICT[TEST_MIR][:-1]
        num_canon = [count_num_canon(utr, sitem8) for utr in utr_list]
        
        temp = pd.DataFrame({'mir': TEST_MIR,
                             'pred': pred,
                             'log2fc_bayes': actual,
                             'num_canon': num_canon,
                             'log2_tpm': tpms,
                             'log2_bayes_intercept': log2_intercept,
                             'nosite3': merged['nosite3'].values / np.log(2),
                             'log2fc_method3': tpms - merged['nosite3'].values / np.log(2)
                            })
        
        pred_df.append(temp)
        
        r2 = stats.linregress(pred, actual)[2]**2
        r2_dict_bayes[TEST_MIR] = r2

    pred_df = pd.concat(pred_df)
    
    return pred_df, r2_dict_bayes, freeAgo_guide, freeAgo_pass, intercept_dict

def plot_results(pred_df, num_colors, new_cmap, test_mirs, outfile):
    # sns.set_style('ticks')
    # sns.set_context('poster')
    # num_colors = 9

    fig = plt.figure(figsize=(18,15))
#     ax_colorbar = fig.add_axes([0.9, 0.15, 0.01, 0.75])
    # ax_colorbar = fig.add_axes([1.5, 0.07, 0.01, 0.25])
    # new_cmap = discrete_colorbar(ax_colorbar, plt.cm.plasma, 9, 'vertical')

    kwargs = {'vmin': 0,
              'vmax': num_colors-1,
              'cmap': new_cmap,
              's':15,
              'alpha': 0.5}
    
    for IX, TEST_MIR in enumerate(test_mirs):
        ax = plt.subplot(3,4,IX+1)
        ax.set_xlim(-0.025,1.5)
        ax.set_ylim(-4,2)
        
        temp = pred_df[pred_df['mir'] == TEST_MIR]
        pred = -1 * temp['pred'].values
        actual = temp['log2fc_bayes'].values
        num_canon = temp['num_canon'].values
        
        ax.scatter(pred, actual, c=num_canon, **kwargs)
        ax.set_title(TEST_MIR.replace('mir','miR-'))
        r2 = stats.linregress(pred, actual)[2]**2
        ax.text(0.7,0.9,r"$r^2$ = {:.2g}".format(r2), transform=ax.transAxes)
        # ax.set_ylabel(r'Measured fold-change (log$_2$)')
        # ax.set_xlabel(r'Predicted repression (log$_2$)')
        sns.despine()
        
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)