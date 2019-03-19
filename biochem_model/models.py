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

        optimizer.minimize(sess, feed_dict=feed_dict)
        for name, var in self.vars.items():
            self.vars_evals[name] = sess.run(var)

        self.eval_pred, self.eval_pred_normed, self.eval_label_normed = sess.run([pred, pred_normed, labels_normed], feed_dict=feed_dict)
        self.r2 = stats.linregress(self.eval_pred_normed.flatten(), self.eval_label_normed.flatten())[2]**2


    def predict(self, sess, data, feed_dict):
        pred = self.get_pred(data)
        return sess.run(pred, feed_dict=feed_dict)


class OccupancyOnlyModel(Model):
    def __init__(self, num_mirs, withORF, withUTR5, with_init):
        super().__init__()
        self.vars['freeAgo'] = tf.get_variable('freeAgo', shape=[1, num_mirs, 1],
            initializer=tf.constant_initializer(-5.5))
        self.vars['log_decay'] = tf.get_variable('log_decay', shape=(), initializer=tf.constant_initializer(0.0))

        self.with_init = with_init
        if self.with_init:
            self.vars['freeAgo_init'] = tf.get_variable('freeAgo_init', shape=[1, num_mirs, 1],
                initializer=tf.constant_initializer(-8.5))

        self.withORF = withORF
        if self.withORF:
            self.vars['orf_ka_offset'] = tf.get_variable('orf_ka_offset', shape=(), initializer=tf.constant_initializer(-2.0))

        self.withUTR5 = withUTR5
        if self.withUTR5:
            self.vars['utr5_ka_offset'] = tf.get_variable('utr5_ka_offset', shape=(), initializer=tf.constant_initializer(-2.0))

    def get_nbound(self, ka_vals, mask, freeAgo):
        occ = tf.sigmoid(ka_vals + freeAgo)
        nbound = tf.reduce_sum(tf.multiply(occ, mask), axis=2)
        return nbound

    def get_pred(self, data):
        nbound = self.get_nbound(data['utr3_ka_vals'], data['utr3_mask'], self.vars['freeAgo'])
        if self.withORF:
            nbound_orf = self.get_nbound(data['orf_ka_vals'] + self.vars['orf_ka_offset'], data['orf_mask'], self.vars['freeAgo'])
            nbound = nbound + nbound_orf

        if self.withUTR5:
            nbound_utr5 = self.get_nbound(data['utr5_ka_vals'] + self.vars['utr5_ka_offset'], data['utr5_mask'], self.vars['freeAgo'])
            nbound = nbound + nbound_utr5

        pred = -1 * tf.log1p(tf.exp(self.vars['log_decay']) * nbound)
        if self.with_init:
            nbound_init = self.get_nbound(data['utr3_ka_vals'], data['utr3_mask'], self.vars['freeAgo_init'])
            if self.withORF:
                nbound_orf_init = self.get_nbound(data['orf_ka_vals'] + self.vars['orf_ka_offset'], data['orf_mask'], self.vars['freeAgo_init'])
                nbound_init = nbound_init + nbound_orf_init

            if self.withUTR5:
                nbound_utr5_init = self.get_nbound(data['utr5_ka_vals'] + self.vars['utr5_ka_offset'], data['utr5_mask'], self.vars['freeAgo_init'])
                nbound_init = nbound_init + nbound_utr5_init

            pred = pred + tf.log1p(tf.exp(self.vars['log_decay']) * nbound_init)


        return pred


class OriginalModel(OccupancyOnlyModel):
    def __init__(self, num_mirs, withORF, withUTR5, with_init):
        super().__init__(num_mirs, withORF, withUTR5, with_init)
        self.vars['log_utr3_coef'] = tf.get_variable('log_utr3_coef', shape=(), initializer=tf.constant_initializer(-2.0))
        if self.withORF:
            self.vars['log_orf_coef'] = tf.get_variable('log_orf_coef', shape=(), initializer=tf.constant_initializer(-2.0))

        if self.withUTR5:
            self.vars['log_utr5_coef'] = tf.get_variable('log_utr5_coef', shape=(), initializer=tf.constant_initializer(-2.0))


    def get_pred(self, data):
        nbound = self.get_nbound(data['utr3_ka_vals'], data['utr3_mask'], self.vars['freeAgo'])
        nbound_endog = tf.exp(self.vars['log_utr3_coef']) * tf.reshape(data['utr3_len'], [-1, 1])
        if self.withORF:
            nbound += self.get_nbound(data['orf_ka_vals'] + self.vars['orf_ka_offset'], data['orf_mask'], self.vars['freeAgo'])
            nbound_endog += tf.exp(self.vars['log_orf_coef']) * tf.reshape(data['orf_len'], [-1, 1])

        if self.withUTR5:
            nbound += self.get_nbound(data['utr5_ka_vals'] + self.vars['utr5_ka_offset'], data['utr5_mask'], self.vars['freeAgo'])
            nbound_endog += tf.exp(self.vars['log_utr5_coef']) * tf.reshape(data['utr5_len'], [-1, 1])

        pred_endog = tf.log1p(nbound_endog)
        pred_transfect = tf.log1p((tf.exp(self.vars['log_decay']) * nbound) + nbound_endog)
        pred = -1 * (pred_transfect - pred_endog)

        if self.with_init:
            nbound_init = self.get_nbound(data['utr3_ka_vals'], data['utr3_mask'], self.vars['freeAgo_init'])
            if self.withORF:
                nbound_orf_init = self.get_nbound(data['orf_ka_vals'] + self.vars['orf_ka_offset'], data['orf_mask'], self.vars['freeAgo_init'])
                nbound_init = nbound_init + nbound_orf_init

            if self.withUTR5:
                nbound_utr5_init = self.get_nbound(data['utr5_ka_vals'] + self.vars['utr5_ka_offset'], data['utr5_mask'], self.vars['freeAgo_init'])
                nbound_init = nbound_init + nbound_utr5_init

            pred += (tf.log1p(nbound_init + nbound_endog) - pred_endog)

        return pred


# class ModifierModel(OccupancyOnlyModel):
#     def __init__(self, num_mirs, withORF):
#         super().__init__(num_mirs, withORF)
#         self.vars['utr_coef'] = tf.get_variable('utr_coef', shape=(), initializer=tf.constant_initializer(-0.1))
#         if self.withORF:
#             self.vars['orf_coef'] = tf.get_variable('orf_coef', shape=(), initializer=tf.constant_initializer(-0.1))

#         self.vars['bias'] = tf.get_variable('bias', shape=(), initializer=tf.constant_initializer(1.0))

#     def get_pred(self, data):
#         modifier = (self.vars['utr_coef'] * data['utr_len']) + (self.vars['orf_coef'] * data['orf_len']) + self.vars['bias']
#         nbound = self.get_nbound(data['utr_ka_vals'] + tf.reshape(modifier, [-1, 1, 1]), data['utr_mask'])
#         if self.withORF:
#             nbound_orf = self.get_nbound(data['orf_ka_vals'] + self.vars['orf_ka_offset'], data['orf_mask'])
#             nbound = nbound + nbound_orf
#         multiplier = (self.vars['utr_coef'] * data['utr_len']) + (self.vars['orf_coef'] * data['orf_len']) + self.vars['bias']
#         print(multiplier)
#         # pred = nbound * tf.reshape(multiplier, [-1, 1])
#         return -1 * tf.log1p(nbound) * tf.reshape(multiplier, [-1, 1])



# class OriginalModelLet7(OriginalModel):
#     def __init__(self, num_mirs):
#         super().__init__(num_mirs)
#         # self.vars['freeAgo_init_let7'] = tf.get_variable('freeAgo_init_let7', shape=(), initializer=tf.constant_initializer(-7.0))
#         self.vars['freeAgo_init'] = tf.get_variable('freeAgo_init', shape=[1, num_mirs, 1],
#             initializer=tf.constant_initializer(-7.5))

#     def get_pred(self, data):
#         occ = tf.sigmoid(data['utr_ka_vals'] + self.vars['freeAgo'])
#         occ_init = tf.sigmoid(data['utr_ka_vals'] + self.vars['freeAgo_init'])

#         nbound = tf.exp(self.vars['log_decay']) * tf.reduce_sum(tf.multiply(occ, data['utr_mask']), axis=2)
#         nbound_init = tf.exp(self.vars['log_decay']) * tf.reduce_sum(tf.multiply(occ_init, data['utr_mask']), axis=2)
#         nbound_endog = tf.exp(self.vars['log_utr_coef']) * tf.reshape(data['utr_len'], [-1, 1])

#         pred = -1 * (tf.log1p(nbound + nbound_endog) - tf.log1p(nbound_init + nbound_endog))


#         # print(nbound, nbound_let7)
#         # nbound_all = tf.exp(self.vars['log_decay']) * tf.concat([nbound[:, :-1], nbound[:, -1:] - nbound_let7], axis=1)
#         # nbound_endog = tf.exp(self.vars['log_utr_coef']) * tf.reshape(data['utr_len'], [-1, 1])
#         # pred_endog = tf.log1p(nbound_endog)
#         # pred_transfect = tf.log1p(nbound_all + nbound_endog)
#         # pred = -1 * (pred_transfect - pred_endog)
#         return pred


class LinearModel(Model):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.vars['coefs'] = tf.get_variable('coefs', shape=[1, 1, num_features],
            initializer=tf.constant_initializer(-0.1))

    def get_pred(self, data):
        pred = tf.reduce_sum(tf.multiply(data['features_collapsed'], self.vars['coefs']), axis=2)
        # pred = tf.reduce_sum(tf.multiply(weighted, data['utr_mask']), axis=2)
        return pred


class BoundedLinearModel(LinearModel):

    def predict(self, sess, data, feed_dict):
        weighted = tf.reduce_sum(tf.multiply(data['features'], tf.expand_dims(self.vars['coefs'], axis=0)), axis=3)
        bounded = tf.minimum(weighted, data['bounds'])
        # pred = tf.reduce_sum(tf.multiply(bounded, data['utr_mask']), axis=2)
        pred = tf.reduce_sum(bounded, axis=2)
        return sess.run(pred, feed_dict=feed_dict)


# class SigmoidModel(Model):
#     def __init__(self, num_features_pre_sigmoid, num_features_post_sigmoid, num_mirs):
#         super().__init__()
#         self.num_features_pre_sigmoid = num_features_pre_sigmoid
#         self.num_features_post_sigmoid = num_features_post_sigmoid
#         self.num_mirs = num_mirs
#         self.vars['coefs_pre_sigmoid'] = tf.get_variable('coefs_pre_sigmoid', shape=[1, 1, 1, num_features_pre_sigmoid],
#             initializer=tf.constant_initializer(1.0))
#         self.vars['coefs_post_sigmoid'] = tf.get_variable('coefs_post_sigmoid', shape=[1, 1, 1, num_features_post_sigmoid],
#             initializer=tf.constant_initializer(0.01))
#         self.vars['bias1'] = tf.get_variable('bias1', shape=(), initializer=tf.constant_initializer(-5.0))

#         self.vars['decay'] = tf.get_variable('decay', shape=(), initializer=tf.constant_initializer(1.0))

#     def get_pred(self, data):
#         features, mask = data['features'], data['utr_mask']
#         features1 = features[:, :, :, :self.num_features_pre_sigmoid]
#         features2 = features[:, :, :, self.num_features_pre_sigmoid:]
#         weighted1 = self.vars['decay'] * tf.sigmoid(tf.reduce_sum(tf.multiply(features1, self.vars['coefs_pre_sigmoid']), axis=3) + self.vars['bias1'])
#         weighted2 = tf.reduce_sum(tf.multiply(features2, self.vars['coefs_post_sigmoid']), axis=3)

#         weighted = weighted1 + weighted2
#         pred = -1 * tf.reduce_sum(tf.multiply(weighted, mask), axis=2)
#         return pred

# class DoubleSigmoidModel(SigmoidModel):
#     def __init__(self, num_features_pre_sigmoid, num_features_post_sigmoid, num_mirs):
#         super().__init__(num_features_pre_sigmoid, num_features_post_sigmoid, num_mirs)
        
#         self.vars['bias2'] = tf.get_variable('bias2', shape=(), initializer=tf.constant_initializer(0.0))

#     def get_pred(self, data):
#         features, mask = data['features'], data['utr_mask']
#         features1 = features[:, :, :, :self.num_features_pre_sigmoid]
#         features2 = features[:, :, :, self.num_features_pre_sigmoid:]
#         weighted1 = self.vars['decay'] * tf.sigmoid(tf.reduce_sum(tf.multiply(features1, self.vars['coefs_pre_sigmoid']), axis=3) + self.vars['bias1'])
#         weighted2 = tf.sigmoid(tf.reduce_sum(tf.multiply(features2, self.vars['coefs_post_sigmoid']), axis=3) + self.vars['bias2'])

#         weighted = tf.multiply(weighted1, weighted2)
#         pred = -1 * tf.reduce_sum(tf.multiply(weighted, mask), axis=2)
#         return pred


# class SigmoidFreeAGOModel(Model):
#     def __init__(self, num_features_pre_sigmoid, num_features_post_sigmoid, num_mirs):
#         super().__init__()
#         self.num_features_pre_sigmoid = num_features_pre_sigmoid
#         self.num_features_post_sigmoid = num_features_post_sigmoid
#         self.num_mirs = num_mirs
#         self.vars['coefs_pre_sigmoid'] = tf.get_variable('coefs_pre_sigmoid', shape=[1, 1, 1, num_features_pre_sigmoid],
#             initializer=tf.constant_initializer(1.0))
#         self.vars['coefs_post_sigmoid'] = tf.get_variable('coefs_post_sigmoid', shape=[1, 1, 1, num_features_post_sigmoid],
#             initializer=tf.constant_initializer(0.01))
#         self.vars['freeAgo'] = tf.get_variable('freeAgo', shape=[1, num_mirs, 1],
#             initializer=tf.constant_initializer(-5.5))

#         self.vars['decay'] = tf.get_variable('decay', shape=(), initializer=tf.constant_initializer(1.0))

#     def get_pred(self, data):
#         features, mask = data['features'], data['utr_mask']
#         features1 = features[:, :, :, :self.num_features_pre_sigmoid]
#         features2 = features[:, :, :, self.num_features_pre_sigmoid:]
#         weighted1 = self.vars['decay'] * tf.sigmoid(tf.reduce_sum(tf.multiply(features1, self.vars['coefs_pre_sigmoid']), axis=3) + self.vars['freeAgo'])
#         weighted2 = tf.reduce_sum(tf.multiply(features2, self.vars['coefs_post_sigmoid']), axis=3)

#         weighted = weighted1 + weighted2
#         pred = -1 * tf.reduce_sum(tf.multiply(weighted, mask), axis=2)
#         return pred


# class DoubleSigmoidFreeAGOModel(SigmoidFreeAGOModel):
#     def __init__(self, num_features_pre_sigmoid, num_features_post_sigmoid, num_mirs):
#         super().__init__(num_features_pre_sigmoid, num_features_post_sigmoid, num_mirs)
        
#         self.vars['bias'] = tf.get_variable('bias', shape=(), initializer=tf.constant_initializer(0.0))

#     def get_pred(self, data):
#         features, mask = data['features'], data['utr_mask']
#         features1 = features[:, :, :, :self.num_features_pre_sigmoid]
#         features2 = features[:, :, :, self.num_features_pre_sigmoid:]
#         weighted1 = tf.sigmoid(tf.reduce_sum(tf.multiply(features1, self.vars['coefs_pre_sigmoid']), axis=3) + self.vars['freeAgo'])
#         weighted2 = tf.sigmoid(tf.reduce_sum(tf.multiply(features2, self.vars['coefs_post_sigmoid']), axis=3) + self.vars['bias'])

#         weighted = tf.multiply(tf.log1p(weighted1), weighted2)
#         pred = -1 * tf.reduce_sum(tf.multiply(weighted, mask), axis=2)
#         return pred


# class DoubleSigmoidFreeAGOLet7Model(SigmoidFreeAGOModel):
#     def __init__(self, num_features_pre_sigmoid, num_features_post_sigmoid, num_mirs):
#         super().__init__(num_features_pre_sigmoid, num_features_post_sigmoid, num_mirs)
        
#         self.vars['bias'] = tf.get_variable('bias', shape=(), initializer=tf.constant_initializer(0.0))
#         self.vars['let7_freeago_init'] = tf.get_variable('let7_freeago_init', shape=(), initializer=tf.constant_initializer(-7.0))

#     def get_pred(self, data):
#         features, mask = data['features'], data['utr_mask']
#         features1 = features[:, :, :, :self.num_features_pre_sigmoid]
#         features2 = features[:, :, :, self.num_features_pre_sigmoid:]
#         weighted1 = tf.reduce_sum(tf.multiply(features1, self.vars['coefs_pre_sigmoid']), axis=3)
#         weighted2 = tf.sigmoid(tf.reduce_sum(tf.multiply(features2, self.vars['coefs_post_sigmoid']), axis=3) + self.vars['bias'])

#         pred1 = tf.multiply(tf.log1p(tf.sigmoid(weighted1 + self.vars['freeAgo'])), weighted2)
#         pred2 = tf.multiply(tf.log1p(tf.sigmoid(weighted1[:, -1:, :] + self.vars['let7_freeago_init'])), weighted2[:, -1:, :])
        
#         pred = tf.concat([pred1[:, :-1, :], pred1[:, -1:, :] - pred2], axis=1)
#         pred = -1 * tf.reduce_sum(tf.multiply(pred, mask), axis=2)
#         return pred


# class DoubleSigmoidFreeAGOWithORFModel(DoubleSigmoidFreeAGOModel):
#     def __init__(self, num_features_pre_sigmoid, num_features_post_sigmoid, num_mirs):
#         super().__init__(num_features_pre_sigmoid, num_features_post_sigmoid, num_mirs)
        
#         self.vars['orf_bias'] = tf.get_variable('orf_bias', shape=(), initializer=tf.constant_initializer(0.0))

#     def get_orf_pred(self, features, mask):
#         weighted = self.vars['decay'] * tf.sigmoid(features + self.vars['freeAgo'] + self.vars['orf_bias'])
#         pred = tf.reduce_sum(tf.multiply(weighted, mask), axis=2)
#         return pred

#     def fit(self, sess, features, mask, orf_features, orf_mask, labels, feed_dict, maxiter):
#         utr_pred = self.get_pred(features, mask)
#         orf_pred = self.get_orf_pred(orf_features, orf_mask)
#         pred = utr_pred + orf_pred
#         loss, pred_normed, labels_normed = self.get_loss(pred, labels)
#         optimizer = ScipyOptimizerInterface(loss, options={'maxiter': maxiter})

#         optimizer.minimize(sess, feed_dict=feed_dict)
#         for name, var in self.vars.items():
#             self.vars_evals[name] = sess.run(var)

#         self.eval_pred, self.eval_pred_normed, self.eval_label_normed = sess.run([pred, pred_normed, labels_normed], feed_dict=feed_dict)
#         self.r2 = stats.linregress(self.eval_pred_normed.flatten(), self.eval_label_normed.flatten())[2]**2


#     def predict(self, sess, features, mask, orf_features, orf_mask, feed_dict):
#         utr_pred = self.get_pred(features, mask)
#         orf_pred = self.get_orf_pred(orf_features, orf_mask)
#         pred = utr_pred + orf_pred
#         return sess.run(pred, feed_dict=feed_dict)







