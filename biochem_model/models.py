import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface


class Model():
    def __init__(self):
        self.vars = {}
        self.vars_evals = {}

    def get_loss(self, pred, labels):
        pred_normed = pred - tf.reshape(tf.reduce_mean(pred, axis=1), [-1, 1])
        labels_normed = labels - tf.reshape(tf.reduce_mean(labels, axis=1), [-1, 1])

        loss = tf.nn.l2_loss(pred_normed - labels_normed)
        return loss, pred_normed, labels_normed

    def get_pred(self, data):
        raise NotImplementedError()

    def fit(self, sess, data, feed_dict, maxiter):
        pred = self.get_pred(data)
        loss, pred_normed, labels_normed = self.get_loss(pred, data['labels'])
        optimizer = ScipyOptimizerInterface(loss, options={'maxiter': maxiter})
        self.losses = []
        def append_loss(loss):
            self.losses.append(loss)

        optimizer.minimize(sess, feed_dict=feed_dict, loss_callback=append_loss, fetches=[loss])
        for name, var in self.vars.items():
            self.vars_evals[name] = sess.run(var)

        self.eval_pred, self.eval_pred_normed, self.eval_label, self.eval_label_normed = sess.run([pred, pred_normed, data['labels'], labels_normed], feed_dict=feed_dict)
        self.r2 = stats.linregress(self.eval_pred_normed.flatten(), self.eval_label_normed.flatten())[2]**2
        self.final_loss = sess.run(loss, feed_dict=feed_dict)


    def predict(self, sess, data, feed_dict):
        pred = self.get_pred(data)
        return sess.run(pred, feed_dict=feed_dict)


# FINAL MODEL
class OccupancyWithFeaturesModel(Model):
    def __init__(self, num_guides, num_features, init_bound=False, fit_background=False, passenger=False, set_vars={}):
        super().__init__()

        if 'set_freeAGO' in set_vars:
            FA_initializer = set_vars['set_freeAGO']
            fitFA = False
        else:
            FA_initializer = -5.5
            fitFA = True

        if passenger:
            self.vars['freeAGO'] = tf.get_variable('freeAGO', shape=[1, num_guides * 2, 1],
                initializer=tf.constant_initializer(FA_initializer), trainable=fitFA)
        else:
            self.vars['freeAGO'] = tf.get_variable('freeAGO', shape=[1, num_guides, 1],
                initializer=tf.constant_initializer(FA_initializer), trainable=fitFA)

        if 'log_decay' in set_vars:
            self.vars['log_decay'] = tf.get_variable('log_decay', shape=(), initializer=tf.constant_initializer(set_vars['log_decay']), trainable=False)
        else:
            self.vars['log_decay'] = tf.get_variable('log_decay', shape=(), initializer=tf.constant_initializer(0.0))

        if 'feature_coefs' in set_vars:
            self.vars['feature_coefs'] = tf.get_variable('feature_coefs', shape=[1, 1, 1, num_features],
                initializer=tf.constant_initializer(set_vars['feature_coefs']), trainable=False)
        else:
            self.vars['feature_coefs'] = tf.get_variable('feature_coefs', shape=[1, 1, 1, num_features], initializer=tf.constant_initializer(-0.1))

        if init_bound:
            self.vars['nosite_conc'] = tf.get_variable('nosite_conc', shape=(),
                initializer=tf.constant_initializer(0.0), trainable=fit_background)

        self.init_bound = init_bound
        self.passenger = passenger

    def get_pred(self, data):

        # if freeAGOs supplied, use those
        if 'freeAGO' in data:
            freeAGOs = data['freeAGO']
        else:
            freeAGOs = self.vars['freeAGO']

        feature_contribution = tf.reduce_sum(tf.multiply(data['features'], self.vars['feature_coefs']), axis=3)
        nosite_feature_contribution = tf.reduce_sum(tf.multiply(data['nosite_features'], self.vars['feature_coefs']), axis=3)

        occ = tf.sigmoid(data['ka_vals'] + feature_contribution + freeAGOs)
        nbound = tf.reduce_sum(tf.multiply(occ, data['mask']), axis=2)

        if self.passenger:
            nbound = tf.reduce_sum(tf.reshape(nbound, [-1, data['num_guides'], 2]), axis=2)

        if not self.init_bound:
            pred = -1 * tf.log1p(tf.exp(self.vars['log_decay']) * nbound)

        else:
            occ_init = tf.sigmoid(nosite_feature_contribution + freeAGOs + self.vars['nosite_conc'])
            nbound_init = tf.reduce_sum(tf.multiply(occ_init, data['mask']), axis=2)
            if self.passenger:
                nbound_init = tf.reduce_sum(tf.reshape(nbound_init, [-1, data['num_guides'], 2]), axis=2)
        
            pred = tf.log1p(tf.exp(self.vars['log_decay']) * nbound_init) - tf.log1p(tf.exp(self.vars['log_decay']) * nbound)

        return pred


class OccupancyOnlyModel(Model):
    def __init__(self, num_mirs, withORF, withUTR5, with_init):
        super().__init__()
        self.vars['freeAgo'] = tf.get_variable('freeAgo', shape=[1, num_mirs, 1],
            initializer=tf.constant_initializer(-5.5))
        self.vars['log_decay'] = tf.get_variable('log_decay', shape=(), initializer=tf.constant_initializer(0.0))

        self.with_init = with_init
        if self.with_init:
            self.vars['freeAgo_init_val'] = tf.get_variable('freeAgo_init_val', shape=[1], initializer=tf.constant_initializer(-8.0))
            self.vars['freeAgo_init'] = tf.reshape(tf.concat([tf.constant([-100.0] * (num_mirs - 1)), self.vars['freeAgo_init_val']], axis=0), [1, num_mirs, 1])

        self.withORF = withORF
        if self.withORF:
            self.vars['orf_ka_offset'] = tf.get_variable('orf_ka_offset', shape=(), initializer=tf.constant_initializer(-2.0))

        self.withUTR5 = withUTR5
        if self.withUTR5:
            self.vars['utr5_ka_offset'] = tf.get_variable('utr5_ka_offset', shape=(), initializer=tf.constant_initializer(-2.0))

    def get_nbound(self, ka_vals, offset, mask, freeAgo):
        occ = tf.sigmoid((ka_vals + offset) + freeAgo) - tf.sigmoid(offset + freeAgo)
        nbound = tf.reduce_sum(tf.multiply(occ, mask), axis=2)
        return nbound

    def get_pred(self, data):
        nbound = self.get_nbound(data['utr3_ka_vals'], 0, data['utr3_mask'], self.vars['freeAgo'])
        if self.withORF:
            nbound_orf = self.get_nbound(data['orf_ka_vals'], self.vars['orf_ka_offset'], data['orf_mask'], self.vars['freeAgo'])
            nbound = nbound + nbound_orf

        if self.withUTR5:
            nbound_utr5 = self.get_nbound(data['utr5_ka_vals'], self.vars['utr5_ka_offset'], data['utr5_mask'], self.vars['freeAgo'])
            nbound = nbound + nbound_utr5

        # pred = -1 * tf.log1p(tf.exp(self.vars['log_decay']) * nbound)
        if self.with_init:
            nbound_init = self.get_nbound(data['utr3_ka_vals'], data['utr3_mask'], self.vars['freeAgo_init'])
            if self.withORF:
                nbound_orf_init = self.get_nbound(data['orf_ka_vals'], self.vars['orf_ka_offset'], data['orf_mask'], self.vars['freeAgo_init'])
                nbound_init = nbound_init + nbound_orf_init

            if self.withUTR5:
                nbound_utr5_init = self.get_nbound(data['utr5_ka_vals'], self.vars['utr5_ka_offset'], data['utr5_mask'], self.vars['freeAgo_init'])
                nbound_init = nbound_init + nbound_utr5_init

            # pred = pred + tf.log1p(tf.exp(self.vars['log_decay']) * nbound_init)
            nbound -= nbound_init

        pred = -1 * tf.log1p(tf.exp(self.vars['log_decay']) * nbound)


        return pred
