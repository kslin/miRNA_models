import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface

import utils


def expand_features(transcripts, mirs, max_nsites, feature_list, feature_df):
    features_4D = np.zeros([len(transcripts), len(mirs), max_nsites, len(feature_list)])
    for ix, transcript in enumerate(transcripts):
        for iy, mir in enumerate(mirs):
            try:
                temp = feature_df.loc[(transcript, mir)]
                nsites = len(temp)
                features_4D[ix, iy, :nsites, :] = temp[feature_list].values
            except KeyError:
                continue

    mask = ((np.abs(np.sum(features_4D, axis=3))) != 0).astype(int)

    return features_4D, mask



def cross_val(tpms, features, nsites, train_mirs, val_mirs, all_vars, vars_to_normalize, model, maxiter, one_site=False):

    num_features = len(all_vars)
    batches = tpms['batch'].unique()
    train_r2s = []
    val_r2s = []

    conf = tf.ConfigProto(inter_op_parallelism_threads=4, intra_op_parallelism_threads=24) 
    with tf.Session(config=conf) as sess:

        features_tensor = tf.placeholder(tf.float32, shape=[None, None, None, num_features], name='features')
        mask_tensor = tf.placeholder(tf.float32, shape=[None, None, None], name='nsites')
        labels_tensor = tf.placeholder(tf.float32, shape=[None, None], name='labels')

        if one_site:
            black_list = nsites[nsites['nsites'] > 1].reset_index()['transcript'].unique()    

        for batch in batches:
            train_transcripts = sorted(list(tpms[tpms['batch'] != batch].index))
            val_transcripts = sorted(list(tpms[tpms['batch'] == batch].index))

            if one_site:
                train_transcripts = [x for x in train_transcripts if x not in black_list]

            print(len(train_transcripts), len(val_transcripts))

            # get training and validation labels
            train_labels = tpms.loc[train_transcripts][train_mirs].values
            val_labels = tpms.loc[val_transcripts][val_mirs].values

            # get training and validation features
            train_features = features.query('transcript in @train_transcripts & mir in @train_mirs')
            val_features = features.query('transcript in @val_transcripts & mir in @val_mirs')

            # get max nsites for training and validation sets
            train_nsites = nsites.query('transcript in @train_transcripts & mir in @train_mirs')
            val_nsites = nsites.query('transcript in @val_transcripts & mir in @val_mirs')

            train_max_nsites = np.max(train_nsites['nsites'].values)
            val_max_nsites = np.max(val_nsites['nsites'].values)

            print(train_max_nsites, val_max_nsites)

            # normalize continuous variables using training set
            norm_means = np.mean(train_features[vars_to_normalize].values, axis=0).reshape([1, -1])
            norm_stds = np.std(train_features[vars_to_normalize].values, axis=0).reshape([1, -1])
            train_features[vars_to_normalize] = (train_features[vars_to_normalize] - norm_means) / norm_stds
            val_features[vars_to_normalize] = (val_features[vars_to_normalize] - norm_means) / norm_stds

            # expand features
            train_features_4D, train_mask = expand_features(train_transcripts, train_mirs, train_max_nsites, all_vars, train_features)
            assert(np.sum(train_mask) == np.sum(train_nsites['nsites'].values))

            # expand features
            val_features_4D, val_mask = expand_features(val_transcripts, val_mirs, val_max_nsites, all_vars, val_features)
            assert(np.sum(val_mask) == np.sum(val_nsites['nsites'].values))

            print('Train feature shape: {}'.format(train_features_4D.shape))
            print('Val feature shape: {}'.format(val_features_4D.shape))

            train_feed_dict = {
                    features_tensor: train_features_4D,
                    mask_tensor: train_mask,
                    labels_tensor: train_labels
                }

            val_feed_dict = {
                    features_tensor: val_features_4D,
                    mask_tensor: val_mask,
                    labels_tensor: val_labels
                }

            sess.run(tf.global_variables_initializer())

            model.fit(sess, features_tensor, mask_tensor, labels_tensor, train_feed_dict, maxiter)

            train_r2s.append(model.r2)

            val_pred = model.predict(sess, features_tensor, mask_tensor, val_feed_dict)
            valr2 = utils.get_r2_unnormed(val_pred, val_labels)
            print(valr2)
            val_r2s.append(valr2)

            return train_r2s, val_r2s, None

        # train on all data and report predictions
        all_transcripts = sorted(list(tpms.index))
        all_mirs = train_mirs
        all_features = features.query('transcript in @all_transcripts & mir in @all_mirs')
        all_nsites = nsites.query('transcript in @all_transcripts & mir in @all_mirs')
        all_labels = tpms.loc[all_transcripts][all_mirs].values

        # normalize continuous variables
        norm_means = np.mean(all_features[vars_to_normalize].values, axis=0).reshape([1, -1])
        norm_stds = np.std(all_features[vars_to_normalize].values, axis=0).reshape([1, -1])
        all_features[vars_to_normalize] = (all_features[vars_to_normalize] - norm_means) / norm_stds

        max_nsites = np.max(all_nsites['nsites'].values)

        # expand features
        all_features_4D, all_mask = expand_features(all_transcripts, all_mirs, max_nsites, all_vars, all_features)
        assert(np.sum(all_mask) == np.sum(all_nsites['nsites'].values))

        print(all_features_4D.shape)

        all_feed_dict = {
                features_tensor: all_features_4D,
                mask_tensor: all_mask,
                labels_tensor: all_labels
            }

        sess.run(tf.global_variables_initializer())
        model.fit(sess, features_tensor, mask_tensor, labels_tensor, all_feed_dict, maxiter)

    pred_df = pd.DataFrame({
        'transcript': np.repeat(all_transcripts, len(all_mirs)),
        'mir': all_mirs * len(all_transcripts),
        'pred': model.eval_pred.flatten(),
        'pred_normed': model.eval_pred_normed.flatten(),
        'label': all_labels.flatten(),
        'label_normed': model.eval_label_normed.flatten()
    })

    return train_r2s, val_r2s, pred_df


class Model():
    def __init__(self):
        self.vars = {}
        self.vars_evals = {}

    def get_ind_pred(self, features):
        raise NotImplementedError()

    def get_pred(self, features, mask):
        ind_pred = self.get_ind_pred(features)
        return tf.reduce_sum(tf.multiply(ind_pred, mask), axis=2)

    def get_loss(self, pred, labels):
        pred_normed = pred - tf.reshape(tf.reduce_mean(pred, axis=1), [-1, 1])
        labels_normed = labels - tf.reshape(tf.reduce_mean(labels, axis=1), [-1, 1])

        loss = tf.nn.l2_loss(pred_normed - labels_normed)
        return loss, pred_normed, labels_normed

    def fit(self, sess, features, mask, labels, feed_dict, maxiter):
        pred = self.get_pred(features, mask)
        loss, pred_normed, labels_normed = self.get_loss(pred, labels)
        optimizer = ScipyOptimizerInterface(loss, options={'maxiter': maxiter})

        optimizer.minimize(sess, feed_dict=feed_dict)
        for name, var in self.vars.items():
            self.vars_evals[name] = sess.run(var)

        self.eval_pred, self.eval_pred_normed, self.eval_label_normed = sess.run([pred, pred_normed, labels_normed], feed_dict=feed_dict)
        self.r2 = stats.linregress(self.eval_pred_normed.flatten(), self.eval_label_normed.flatten())[2]**2


    def predict(self, sess, features, mask, feed_dict):
        pred = self.get_pred(features, mask)
        return sess.run(pred, feed_dict=feed_dict)


class LinearModel(Model):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.vars['coefs'] = tf.get_variable('coefs', shape=[1, 1, 1, num_features],
            initializer=tf.constant_initializer(-0.1))

    # def _collapse_features(self, vals, split_sizes, num_mirs):
    #     vals_split = tf.split(vals, split_sizes)
    #     collapsed = tf.reshape(tf.stack([tf.reduce_sum(x, axis=0) for x in vals_split]), [-1, num_mirs, self.num_features])
    #     return collapsed

    def get_ind_pred(self, features):
        return tf.reduce_sum(tf.multiply(features, self.vars['coefs']), axis=3)



class BoundedLinearModel(LinearModel):

    def get_ind_pred(self, features):
        return tf.reduce_sum(tf.multiply(features[:, :, :, :-1], self.vars['coefs']), axis=3)

    def predict(self, sess, features, mask, feed_dict):
        bounds = features[:, :, :, -1]
        ind_pred = self.get_ind_pred(features)
        bounded = tf.minimum(ind_pred, bounds)
        pred = tf.reduce_sum(tf.multiply(bounded, mask), axis=2)
        return sess.run(pred, feed_dict=feed_dict)


class SigmoidModel(Model):
    def __init__(self, num_features_pre_sigmoid, num_features_post_sigmoid, num_mirs):
        super().__init__()
        self.num_features_pre_sigmoid = num_features_pre_sigmoid
        self.num_features_post_sigmoid = num_features_post_sigmoid
        self.num_mirs = num_mirs
        self.vars['coefs_pre_sigmoid'] = tf.get_variable('coefs_pre_sigmoid', shape=[1, 1, 1, num_features_pre_sigmoid],
            initializer=tf.constant_initializer(1.0))
        self.vars['coefs_post_sigmoid'] = tf.get_variable('coefs_post_sigmoid', shape=[1, 1, 1, num_features_post_sigmoid],
            initializer=tf.constant_initializer(0.01))
        self.vars['bias1'] = tf.get_variable('bias1', shape=(), initializer=tf.constant_initializer(-5.0))

        self.vars['decay'] = tf.get_variable('decay', shape=(), initializer=tf.constant_initializer(-1.0))

    def get_pred(self, features, mask):
        features1 = features[:, :, :, :self.num_features_pre_sigmoid]
        features2 = features[:, :, :, self.num_features_pre_sigmoid:]
        weighted1 = self.vars['decay'] * tf.sigmoid(tf.reduce_sum(tf.multiply(features1, self.vars['coefs_pre_sigmoid']), axis=3) + self.vars['bias1'])
        weighted2 = tf.reduce_sum(tf.multiply(features2, self.vars['coefs_post_sigmoid']), axis=3)

        weighted = weighted1 + weighted2
        pred = tf.reduce_sum(tf.multiply(weighted, mask), axis=2)
        return pred

class DoubleSigmoidModel(SigmoidModel):
    def __init__(self, num_features_pre_sigmoid, num_features_post_sigmoid, num_mirs):
        super().__init__(num_features_pre_sigmoid, num_features_post_sigmoid, num_mirs)
        
        self.vars['bias2'] = tf.get_variable('bias2', shape=(), initializer=tf.constant_initializer(0.0))

    def get_pred(self, features, mask):
        features1 = features[:, :, :, :self.num_features_pre_sigmoid]
        features2 = features[:, :, :, self.num_features_pre_sigmoid:]
        weighted1 = self.vars['decay'] * tf.sigmoid(tf.reduce_sum(tf.multiply(features1, self.vars['coefs_pre_sigmoid']), axis=3) + self.vars['bias1'])
        weighted2 = tf.sigmoid(tf.reduce_sum(tf.multiply(features2, self.vars['coefs_post_sigmoid']), axis=3) + self.vars['bias2'])

        weighted = tf.multiply(weighted1, weighted2)
        pred = tf.reduce_sum(tf.multiply(weighted, mask), axis=2)
        return pred


class SigmoidFreeAGOModel(Model):
    def __init__(self, num_features_pre_sigmoid, num_features_post_sigmoid, num_mirs):
        super().__init__()
        self.num_features_pre_sigmoid = num_features_pre_sigmoid
        self.num_features_post_sigmoid = num_features_post_sigmoid
        self.num_mirs = num_mirs
        self.vars['coefs_pre_sigmoid'] = tf.get_variable('coefs_pre_sigmoid', shape=[1, 1, 1, num_features_pre_sigmoid],
            initializer=tf.constant_initializer(1.0))
        self.vars['coefs_post_sigmoid'] = tf.get_variable('coefs_post_sigmoid', shape=[1, 1, 1, num_features_post_sigmoid],
            initializer=tf.constant_initializer(0.01))
        self.vars['freeAgo'] = tf.get_variable('freeAgo', shape=[1, num_mirs, 1],
            initializer=tf.constant_initializer(-5.0))

        self.vars['decay'] = tf.get_variable('decay', shape=(), initializer=tf.constant_initializer(-1.0))

    def get_pred(self, features, mask):
        features1 = features[:, :, :, :self.num_features_pre_sigmoid]
        features2 = features[:, :, :, self.num_features_pre_sigmoid:]
        weighted1 = self.vars['decay'] * tf.sigmoid(tf.reduce_sum(tf.multiply(features1, self.vars['coefs_pre_sigmoid']), axis=3) + self.vars['freeAgo'])
        weighted2 = tf.reduce_sum(tf.multiply(features2, self.vars['coefs_post_sigmoid']), axis=3)

        weighted = weighted1 + weighted2
        pred = tf.reduce_sum(tf.multiply(weighted, mask), axis=2)
        return pred


class DoubleSigmoidFreeAGOModel(SigmoidFreeAGOModel):
    def __init__(self, num_features_pre_sigmoid, num_features_post_sigmoid, num_mirs):
        super().__init__(num_features_pre_sigmoid, num_features_post_sigmoid, num_mirs)
        
        self.vars['bias'] = tf.get_variable('bias', shape=(), initializer=tf.constant_initializer(0.0))

    def get_pred(self, features, mask):
        features1 = features[:, :, :, :self.num_features_pre_sigmoid]
        features2 = features[:, :, :, self.num_features_pre_sigmoid:]
        weighted1 = self.vars['decay'] * tf.sigmoid(tf.reduce_sum(tf.multiply(features1, self.vars['coefs_pre_sigmoid']), axis=3) + self.vars['freeAgo'])
        weighted2 = tf.sigmoid(tf.reduce_sum(tf.multiply(features2, self.vars['coefs_post_sigmoid']), axis=3) + self.vars['bias'])

        weighted = tf.multiply(weighted1, weighted2)
        pred = tf.reduce_sum(tf.multiply(weighted, mask), axis=2)
        return pred


class DoubleSigmoidFreeAGOWithORFModel(DoubleSigmoidFreeAGOModel):
    def __init__(self, num_features_pre_sigmoid, num_features_post_sigmoid, num_mirs):
        super().__init__(num_features_pre_sigmoid, num_features_post_sigmoid, num_mirs)
        
        self.vars['orf_bias'] = tf.get_variable('orf_bias', shape=(), initializer=tf.constant_initializer(0.0))

    def get_orf_pred(self, features, mask):
        weighted = self.vars['decay'] * tf.sigmoid(features + self.vars['freeAgo'] + self.vars['orf_bias'])
        pred = tf.reduce_sum(tf.multiply(weighted, mask), axis=2)
        return pred

    def fit(self, sess, features, mask, orf_features, orf_mask, labels, feed_dict, maxiter):
        utr_pred = self.get_pred(features, mask)
        orf_pred = self.get_orf_pred(orf_features, orf_mask)
        pred = utr_pred + orf_pred
        loss, pred_normed, labels_normed = self.get_loss(pred, labels)
        optimizer = ScipyOptimizerInterface(loss, options={'maxiter': maxiter})

        optimizer.minimize(sess, feed_dict=feed_dict)
        for name, var in self.vars.items():
            self.vars_evals[name] = sess.run(var)

        self.eval_pred, self.eval_pred_normed, self.eval_label_normed = sess.run([pred, pred_normed, labels_normed], feed_dict=feed_dict)
        self.r2 = stats.linregress(self.eval_pred_normed.flatten(), self.eval_label_normed.flatten())[2]**2


    def predict(self, sess, features, mask, orf_features, orf_mask, feed_dict):
        utr_pred = self.get_pred(features, mask)
        orf_pred = self.get_orf_pred(orf_features, orf_mask)
        pred = utr_pred + orf_pred
        return sess.run(pred, feed_dict=feed_dict)







