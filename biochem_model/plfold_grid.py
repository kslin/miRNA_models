import numpy as np
import pandas as pd
import tensorflow as tf

import models
import utils

TPM_FILE = '/lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt'
all_tpms = pd.read_csv(TPM_FILE, sep='\t', index_col=0)
all_tpms.index.name = 'transcript'

all_transcripts = list(all_tpms.index)
use_mirs = sorted(['mir1','mir124','mir155','mir7','lsy6'])


all_features = pd.read_csv('/lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/all_features.txt', sep='\t')
all_features['log_KA'] = -1 * all_features['log_kd']

NUM_SITES = all_features.copy()
NUM_SITES['nsites'] = 1
NUM_SITES = NUM_SITES.groupby(['transcript','mir']).agg({'nsites': np.sum})
MAX_NSITES = np.max(NUM_SITES['nsites'])

all_plfold_info = pd.read_csv('/lab/bartel4_ata/kathyl/RNA_Seq/outputs/rnaplfold/compiled/L40W80u25.txt', sep='\t')
all_plfold_info = all_plfold_info.set_index(['transcript', 'loc'])

# loc_pads = np.arange(-15, 16)
# lengths = np.arange(1, 27)
# loc_pads = np.arange(6, 12)
# lengths = np.arange(10, 14)
loc_pads = np.arange(2, 6)
lengths = np.arange(13, 17)

with open('/lab/bartel4_ata/kathyl/RNA_Seq/outputs/rnaplfold/compiled/L40W80u25_r2s_2.txt', 'wb', buffering=0) as outfile:
    for loc_pad in loc_pads:
        for length in lengths:
            temp = all_features.copy()
            temp['loc'] = (temp['loc'] + loc_pad).astype(int)
            temp['logSA'] = all_plfold_info.reindex(temp[['transcript','loc']].values)['{}'.format(length)].values
            temp['logSA_diff'] = temp['logSA'] - temp['logSA_bg']
            mean_SA_diff = np.mean(temp.dropna()['logSA_diff'])
            temp['logSA_diff'] = temp['logSA_diff'].fillna(mean_SA_diff)
            temp['logSA_bg'] = temp['logSA'] - temp['logSA_diff']
            temp = temp.set_index(['transcript', 'mir'])


            all_features_4D, mask_3D = utils.expand_features_4D(all_transcripts, use_mirs, MAX_NSITES,
                                                                ['log_KA', 'in_ORF', 'logSA_diff'],
                                                                temp)

            ka_vals_3D = all_features_4D[:, :, :, 0]
            features_4D = all_features_4D[:, :, :, 1:]

            tf.reset_default_graph()
            NUM_FEATS = features_4D.shape[-1]

            ka_tensor = tf.placeholder(tf.float32, shape=[None, None, None], name='ka_vals')
            feature_tensor = tf.placeholder(tf.float32, shape=[None, None, None, NUM_FEATS], name='orf_ka')
            mask_tensor = tf.placeholder(tf.float32, shape=[None, None, None], name='mask')
            labels_tensor = tf.placeholder(tf.float32, shape=[None, None], name='labels')

            data = {
                'ka_vals': ka_tensor,
                'mask': mask_tensor,
                'features': feature_tensor,
                'labels': labels_tensor
            }

            feed_dict = {
                ka_tensor: ka_vals_3D,
                mask_tensor: mask_3D,
                feature_tensor: features_4D,
                labels_tensor: all_tpms.loc[all_transcripts][use_mirs].values
            }

            mod = models.OccupancyWithFeaturesModel(len(use_mirs), NUM_FEATS, init_bound=True)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                mod.fit(sess, data, feed_dict, maxiter=200)
                row = [loc_pad, length, mod.r2, mod.final_loss] + list(mod.vars_evals['feature_coefs'].flatten())
                write_str = '{}\t{}\t{}\t{}\t{}\t{}\n'.format(loc_pad, length, mod.r2, mod.final_loss, row[4], row[5])
                outfile.write(write_str.encode('utf8'))
                print(mod.vars_evals)
