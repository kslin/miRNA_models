import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf

import models

def sigmoid(vals):
    return 1 / (1 + np.exp(-1 * vals))

def calc_r2(xs, ys):
    return stats.linregress(xs, ys)[2]**2



tf.logging.set_verbosity(tf.logging.DEBUG)


def test_linear_model(num_genes, num_mirs, num_max_sites, num_features, maxiter):

    # generate random data
    np.random.seed(0)

    # get a random number of sites per mRNA/miRNA interaction
    features = np.zeros([num_genes, num_mirs, num_max_sites, num_features])
    for i in range(num_genes):
        for j in range(num_mirs):
            nsites = np.random.choice(num_max_sites)
            features[i,j,:nsites,:] = np.random.rand(nsites, num_features)

    mask = ((np.abs(np.sum(features, axis=3))) != 0).astype(int)

    true_weights = (np.arange(num_features) + 1.0).reshape([1, 1, 1, -1])
    true_weights = (true_weights - np.mean(true_weights)) / np.std(true_weights)
    labels = np.sum(np.multiply(np.sum(np.multiply(features, true_weights), axis=3), mask), axis=2)

    print(features.shape)
    print(mask.shape)
    print(labels.shape)

    tf.reset_default_graph()

    features_tensor = tf.placeholder(tf.float32, shape=[None, None, None, num_features], name='features')
    mask_tensor = tf.placeholder(tf.float32, shape=[None, None, None], name='nsites')
    labels_tensor = tf.placeholder(tf.float32, shape=[None, None], name='labels')

    data = {
        'features': features_tensor,
        'mask': mask_tensor,
        'labels': labels_tensor
    }

    feed_dict = {
            features_tensor: features,
            mask_tensor: mask,
            labels_tensor: labels
        }

    model = models.LinearModel(num_features)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess, data, feed_dict, maxiter)

        print('True weight diff: {}'.format(np.sum(np.abs(model.vars_evals['coefs'] - true_weights))))

        print('Label r2: {}'.format(model.r2))


def test_boundedlinear_model(num_genes, num_mirs, num_max_sites, num_features, maxiter):

    # generate random data
    np.random.seed(0)

    # get a random number of sites per mRNA/miRNA interaction
    features = np.zeros([num_genes, num_mirs, num_max_sites, num_features])
    for i in range(num_genes):
        for j in range(num_mirs):
            nsites = np.random.choice(num_max_sites)
            features[i,j,:nsites,:] = np.random.rand(nsites, num_features) - 0.5

    mask = ((np.abs(np.sum(features, axis=3))) != 0).astype(int)

    bounds = np.full([num_genes, num_mirs, num_max_sites, 1], -0.03)
    features_plus_bounds = np.concatenate([features, bounds], axis=3)

    true_weights = (np.arange(num_features) + 1.0).reshape([1, 1, 1, -1])
    true_weights = (true_weights - np.mean(true_weights)) / np.std(true_weights)
    weighted = np.sum(np.multiply(features, true_weights), axis=3)
    bounded = np.minimum(weighted, np.squeeze(bounds))
    labels = np.sum(np.multiply(weighted, mask), axis=2)
    labels_bounded = np.sum(np.multiply(bounded, mask), axis=2)

    print(features_plus_bounds.shape)
    print(mask.shape)
    print(labels.shape)

    tf.reset_default_graph()

    features_tensor = tf.placeholder(tf.float32, shape=[None, None, None, None], name='features')
    mask_tensor = tf.placeholder(tf.float32, shape=[None, None, None], name='nsites')
    labels_tensor = tf.placeholder(tf.float32, shape=[None, None], name='labels')

    data = {
        'features': features_tensor,
        'mask': mask_tensor,
        'labels': labels_tensor
    }

    feed_dict = {
            features_tensor: features_plus_bounds,
            mask_tensor: mask,
            labels_tensor: labels
        }

    model = models.BoundedLinearModel(num_features)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess, data, feed_dict, maxiter)

        print('True weight diff: {}'.format(np.sum(np.abs(model.vars_evals['coefs'] - true_weights))))
        print('Label r2: {}'.format(model.r2))

        bounded_pred = model.predict(sess, data, feed_dict)
        print(calc_r2(labels_bounded.flatten(), bounded_pred.flatten()))

def test_sigmoid_model(num_genes, num_mirs, num_max_sites, num_pre_features, num_post_features, maxiter):

    # generate random data
    np.random.seed(0)

    num_features = num_pre_features + num_post_features

    # get a random number of sites per mRNA/miRNA interaction
    features = np.zeros([num_genes, num_mirs, num_max_sites, num_features])
    for i in range(num_genes):
        for j in range(num_mirs):
            nsites = np.random.choice(num_max_sites)
            features[i,j,:nsites,:] = np.random.rand(nsites, num_features)

    mask = ((np.abs(np.sum(features, axis=3))) != 0).astype(int)

    true_weights1 = (np.arange(num_pre_features) + 1.0).reshape([1, 1, 1, -1])
    true_weights1 = (true_weights1 - np.mean(true_weights1)) / np.std(true_weights1)
    true_weights2 = (np.arange(num_post_features) + 1.0).reshape([1, 1, 1, -1])
    true_weights2 = (true_weights2 - np.mean(true_weights2)) / np.std(true_weights2)

    true_bias1 = -1
    true_decay = 1.5

    weighted1 = true_decay * sigmoid(np.sum(np.multiply(features[:, :, :, :num_pre_features], true_weights1), axis=3) + true_bias1)
    weighted2 = np.sum(np.multiply(features[:, :, :, num_pre_features:], true_weights2), axis=3)

    weighted = weighted1 + weighted2
    labels = -1 * np.sum(np.multiply(weighted, mask), axis=2)


    print(features.shape)
    print(mask.shape)
    print(labels.shape)

    tf.reset_default_graph()

    features_tensor = tf.placeholder(tf.float32, shape=[None, None, None, num_features], name='features')
    mask_tensor = tf.placeholder(tf.float32, shape=[None, None, None], name='nsites')
    labels_tensor = tf.placeholder(tf.float32, shape=[None, None], name='labels')

    data = {
        'features': features_tensor,
        'mask': mask_tensor,
        'labels': labels_tensor
    }

    feed_dict = {
            features_tensor: features,
            mask_tensor: mask,
            labels_tensor: labels
        }

    model = models.SigmoidModel(num_pre_features, num_post_features, num_mirs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess, data, feed_dict, maxiter)

        print('True weight1 diff: {}'.format(np.sum(np.abs(model.vars_evals['coefs_pre_sigmoid'] - true_weights1))))
        print('True weight2 diff: {}'.format(np.sum(np.abs(model.vars_evals['coefs_post_sigmoid'] - true_weights2))))
        print('True bias1 diff: {}'.format(np.abs(model.vars_evals['bias1'] - true_bias1)))
        print('True decay diff: {}'.format(np.abs(model.vars_evals['decay'] - true_decay)))

        print('Label r2: {}'.format(model.r2))


def test_doublesigmoid_model(num_genes, num_mirs, num_max_sites, num_pre_features, num_post_features, maxiter):

    # generate random data
    np.random.seed(0)

    num_features = num_pre_features + num_post_features

    # get a random number of sites per mRNA/miRNA interaction
    features = np.zeros([num_genes, num_mirs, num_max_sites, num_features])
    for i in range(num_genes):
        for j in range(num_mirs):
            nsites = np.random.choice(num_max_sites)
            features[i,j,:nsites,:] = np.random.rand(nsites, num_features)

    mask = ((np.abs(np.sum(features, axis=3))) != 0).astype(int)

    true_weights1 = (np.arange(num_pre_features) + 1.0).reshape([1, 1, 1, -1])
    true_weights1 = (true_weights1 - np.mean(true_weights1)) / np.std(true_weights1)
    true_weights2 = (np.arange(num_post_features) + 1.0).reshape([1, 1, 1, -1])
    true_weights2 = (true_weights2 - np.mean(true_weights2)) / np.std(true_weights2)

    true_decay = -1.5
    true_bias1 = -1
    true_bias2 = -0.4

    weighted1 = true_decay * sigmoid(np.sum(np.multiply(features[:, :, :, :num_pre_features], true_weights1), axis=3) + true_bias1)
    weighted2 = sigmoid(np.sum(np.multiply(features[:, :, :, num_pre_features:], true_weights2), axis=3) + true_bias2)

    weighted = np.multiply(weighted1, weighted2)
    labels = np.sum(np.multiply(weighted, mask), axis=2)

    print(features.shape)
    print(mask.shape)
    print(labels.shape)

    tf.reset_default_graph()

    features_tensor = tf.placeholder(tf.float32, shape=[None, None, None, num_features], name='features')
    mask_tensor = tf.placeholder(tf.float32, shape=[None, None, None], name='nsites')
    labels_tensor = tf.placeholder(tf.float32, shape=[None, None], name='labels')

    data = {
        'features': features_tensor,
        'mask': mask_tensor,
        'labels': labels_tensor
    }

    feed_dict = {
            features_tensor: features,
            mask_tensor: mask,
            labels_tensor: labels
        }

    model = models.DoubleSigmoidModel(num_pre_features, num_post_features, num_mirs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess, data, feed_dict, maxiter)

        print('True weight1 diff: {}'.format(np.sum(np.abs(model.vars_evals['coefs_pre_sigmoid'] - true_weights1))))
        print('True weight2 diff: {}'.format(np.sum(np.abs(model.vars_evals['coefs_post_sigmoid'] - true_weights2))))
        print('True decay diff: {}'.format(np.abs(model.vars_evals['decay'] - true_decay)))
        print('True bias1 diff: {}'.format(np.abs(model.vars_evals['bias1'] - true_bias1)))
        print('True bias2 diff: {}'.format(np.abs(model.vars_evals['bias2'] - true_bias2)))

        print('Label r2: {}'.format(model.r2))


def test_sigmoidfreeago_model(num_genes, num_mirs, num_max_sites, num_pre_features, num_post_features, maxiter):

    # generate random data
    np.random.seed(0)

    num_features = num_pre_features + num_post_features

    # get a random number of sites per mRNA/miRNA interaction
    features = np.zeros([num_genes, num_mirs, num_max_sites, num_features])
    for i in range(num_genes):
        for j in range(num_mirs):
            nsites = np.random.choice(num_max_sites)
            features[i,j,:nsites,:] = np.random.rand(nsites, num_features)

    mask = ((np.abs(np.sum(features, axis=3))) != 0).astype(int)

    true_weights1 = (np.arange(num_pre_features) + 1.0).reshape([1, 1, 1, -1])
    true_weights1 = (true_weights1 - np.mean(true_weights1)) / np.std(true_weights1)
    true_weights2 = (np.arange(num_post_features) + 1.0).reshape([1, 1, 1, -1])
    true_weights2 = (true_weights2 - np.mean(true_weights2)) / np.std(true_weights2)

    true_freeAgo = np.random.random(num_mirs).reshape([1, -1, 1])
    true_decay = 1.5

    weighted1 = true_decay * sigmoid(np.sum(np.multiply(features[:, :, :, :num_pre_features], true_weights1), axis=3) + true_freeAgo)
    weighted2 = np.sum(np.multiply(features[:, :, :, num_pre_features:], true_weights2), axis=3)

    weighted = weighted1 + weighted2
    labels = -1 * np.sum(np.multiply(weighted, mask), axis=2)


    print(features.shape)
    print(mask.shape)
    print(labels.shape)

    tf.reset_default_graph()

    features_tensor = tf.placeholder(tf.float32, shape=[None, None, None, num_features], name='features')
    mask_tensor = tf.placeholder(tf.float32, shape=[None, None, None], name='nsites')
    labels_tensor = tf.placeholder(tf.float32, shape=[None, None], name='labels')

    data = {
        'features': features_tensor,
        'mask': mask_tensor,
        'labels': labels_tensor
    }

    feed_dict = {
            features_tensor: features,
            mask_tensor: mask,
            labels_tensor: labels
        }

    model = models.SigmoidFreeAGOModel(num_pre_features, num_post_features, num_mirs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess, data, feed_dict, maxiter)

        print(model.vars_evals['coefs_pre_sigmoid'].flatten())
        print(true_weights1.flatten())

        print('True weight1 diff: {}'.format(np.sum(np.abs(model.vars_evals['coefs_pre_sigmoid'] - true_weights1))))
        print('True weight2 diff: {}'.format(np.sum(np.abs(model.vars_evals['coefs_post_sigmoid'] - true_weights2))))
        print('True freeAgo diff: {}'.format(np.sum(np.abs(model.vars_evals['freeAgo'] - true_freeAgo))))
        print('True decay diff: {}'.format(np.abs(model.vars_evals['decay'] - true_decay)))

        print('Label r2: {}'.format(model.r2))


def test_doublesigmoidfreeago_model(num_genes, num_mirs, num_max_sites, num_pre_features, num_post_features, maxiter):

    # generate random data
    np.random.seed(0)

    num_features = num_pre_features + num_post_features

    # get a random number of sites per mRNA/miRNA interaction
    features = np.zeros([num_genes, num_mirs, num_max_sites, num_features])
    for i in range(num_genes):
        for j in range(num_mirs):
            nsites = np.random.choice(num_max_sites)
            features[i,j,:nsites,:] = np.random.rand(nsites, num_features)

    mask = ((np.abs(np.sum(features, axis=3))) != 0).astype(int)

    true_weights1 = (np.arange(num_pre_features) + 1.0).reshape([1, 1, 1, -1])
    true_weights1 = (true_weights1 - np.mean(true_weights1)) / np.std(true_weights1)
    true_weights2 = (np.arange(num_post_features) + 1.0).reshape([1, 1, 1, -1])
    true_weights2 = (true_weights2 - np.mean(true_weights2)) / np.std(true_weights2)

    true_freeAgo = np.random.random(num_mirs).reshape([1, -1, 1])
    true_decay = 1.5
    true_bias = -0.4

    weighted1 = true_decay * sigmoid(np.sum(np.multiply(features[:, :, :, :num_pre_features], true_weights1), axis=3) + true_freeAgo)
    weighted2 = sigmoid(np.sum(np.multiply(features[:, :, :, num_pre_features:], true_weights2), axis=3) + true_bias)

    weighted = np.multiply(weighted1, weighted2)
    labels = -1 * np.sum(np.multiply(weighted, mask), axis=2)

    print(features.shape)
    print(mask.shape)
    print(labels.shape)

    tf.reset_default_graph()

    features_tensor = tf.placeholder(tf.float32, shape=[None, None, None, num_features], name='features')
    mask_tensor = tf.placeholder(tf.float32, shape=[None, None, None], name='nsites')
    labels_tensor = tf.placeholder(tf.float32, shape=[None, None], name='labels')

    data = {
        'features': features_tensor,
        'mask': mask_tensor,
        'labels': labels_tensor
    }

    feed_dict = {
            features_tensor: features,
            mask_tensor: mask,
            labels_tensor: labels
        }

    model = models.DoubleSigmoidFreeAGOModel(num_pre_features, num_post_features, num_mirs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess, data, feed_dict, maxiter)

        print('True weight1 diff: {}'.format(np.sum(np.abs(model.vars_evals['coefs_pre_sigmoid'] - true_weights1))))
        print('True weight2 diff: {}'.format(np.sum(np.abs(model.vars_evals['coefs_post_sigmoid'] - true_weights2))))
        print('True freeAgo diff: {}'.format(np.sum(np.abs(model.vars_evals['freeAgo'] - true_freeAgo))))
        print('True decay diff: {}'.format(np.abs(model.vars_evals['decay'] - true_decay)))
        print('True bias diff: {}'.format(np.abs(model.vars_evals['bias'] - true_bias)))

        print('Label r2: {}'.format(model.r2))


def test_doublesigmoidfreeagolet7_model(num_genes, num_mirs, num_max_sites, num_pre_features, num_post_features, maxiter):

    # generate random data
    np.random.seed(0)

    num_features = num_pre_features + num_post_features

    # get a random number of sites per mRNA/miRNA interaction
    features = np.zeros([num_genes, num_mirs, num_max_sites, num_features])
    for i in range(num_genes):
        for j in range(num_mirs):
            nsites = np.random.choice(num_max_sites)
            features[i,j,:nsites,:] = np.random.rand(nsites, num_features)

    mask = ((np.abs(np.sum(features, axis=3))) != 0).astype(int)

    true_weights1 = (np.arange(num_pre_features) + 1.0).reshape([1, 1, 1, -1])
    true_weights1 = (true_weights1 - np.mean(true_weights1)) / np.std(true_weights1)
    true_weights2 = (np.arange(num_post_features) + 1.0).reshape([1, 1, 1, -1])
    true_weights2 = (true_weights2 - np.mean(true_weights2)) / np.std(true_weights2)

    true_freeAgo = np.random.random(num_mirs).reshape([1, -1, 1])
    true_freeAgolet7 = true_freeAgo[0,-1,0] - 1
    true_decay = 1.5
    true_bias = -0.4

    weighted1 = np.sum(np.multiply(features[:, :, :, :num_pre_features], true_weights1), axis=3)
    occ1 = sigmoid(weighted1 + true_freeAgo)
    print(np.mean(np.mean(occ1, axis=2), axis=0))
    occ1[:, -1, :] -= sigmoid(weighted1[:, -1, :] + true_freeAgolet7)
    print(np.mean(np.mean(occ1, axis=2), axis=0))
    print(np.min(occ1))

    occ1 *= true_decay
    weighted2 = sigmoid(np.sum(np.multiply(features[:, :, :, num_pre_features:], true_weights2), axis=3) + true_bias)

    weighted = np.multiply(occ1, weighted2)
    labels = -1 * np.sum(np.multiply(weighted, mask), axis=2)

    print(features.shape)
    print(mask.shape)
    print(labels.shape)

    tf.reset_default_graph()

    features_tensor = tf.placeholder(tf.float32, shape=[None, None, None, num_features], name='features')
    mask_tensor = tf.placeholder(tf.float32, shape=[None, None, None], name='nsites')
    labels_tensor = tf.placeholder(tf.float32, shape=[None, None], name='labels')

    data = {
        'features': features_tensor,
        'mask': mask_tensor,
        'labels': labels_tensor
    }

    feed_dict = {
            features_tensor: features,
            mask_tensor: mask,
            labels_tensor: labels
        }

    model = models.DoubleSigmoidFreeAGOLet7Model(num_pre_features, num_post_features, num_mirs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess, data, feed_dict, maxiter)

        print('True weight1 diff: {}'.format(np.sum(np.abs(model.vars_evals['coefs_pre_sigmoid'] - true_weights1))))
        print('True weight2 diff: {}'.format(np.sum(np.abs(model.vars_evals['coefs_post_sigmoid'] - true_weights2))))
        print('True freeAgo diff: {}'.format(np.sum(np.abs(model.vars_evals['freeAgo'] - true_freeAgo))))
        print('True freeAgo_let7 diff: {}'.format(np.abs(model.vars_evals['let7_freeago_init'] - true_freeAgolet7)))
        print('True decay diff: {}'.format(np.abs(model.vars_evals['decay'] - true_decay)))
        print('True bias diff: {}'.format(np.abs(model.vars_evals['bias'] - true_bias)))

        print('Label r2: {}'.format(model.r2))

def test_original_model(num_genes, num_mirs, num_max_sites, maxiter):

    # generate random data
    np.random.seed(0)
    utr_lengths = (np.random.randint(5000, size=num_genes) / 2000).reshape([-1, 1])

    # get a random number of sites per mRNA/miRNA interaction
    features = np.zeros([num_genes, num_mirs, num_max_sites])
    for i in range(num_genes):
        for j in range(num_mirs):
            nsites = np.random.choice(num_max_sites)
            features[i,j,:nsites] = np.random.rand(nsites)

    mask = (features != 0).astype(int)

    true_freeAgo = np.random.random(num_mirs).reshape([1, -1, 1])
    true_decay = 1.5
    true_utr_coef = 0.1

    occ = sigmoid(features + true_freeAgo)
    nbound = true_decay * np.sum(occ * mask, axis=2)
    nbound_endog = true_utr_coef * utr_lengths

    pred_endog = np.log1p(nbound_endog)
    pred_transfect = np.log1p(nbound_endog + nbound)

    labels = -1 * (pred_transfect - pred_endog)

    tf.reset_default_graph()

    features_tensor = tf.placeholder(tf.float32, shape=[None, None, None], name='features')
    mask_tensor = tf.placeholder(tf.float32, shape=[None, None, None], name='mask')
    utrlen_tensor = tf.placeholder(tf.float32, shape=[None, 1], name='utr_len')
    labels_tensor = tf.placeholder(tf.float32, shape=[None, None], name='labels')

    data = {
        'ka_vals': features_tensor,
        'mask': mask_tensor,
        'utr_len': utrlen_tensor,
        'labels': labels_tensor
    }

    feed_dict = {
            features_tensor: features,
            mask_tensor: mask,
            utrlen_tensor: utr_lengths,
            labels_tensor: labels
        }

    model = models.OriginalModel(num_mirs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess, data, feed_dict, maxiter)

        print('True freeAgo diff: {}'.format(np.sum(np.abs(model.vars_evals['freeAgo'] - true_freeAgo))))
        print('True decay diff: {}'.format(np.abs(np.exp(model.vars_evals['log_decay']) - true_decay)))
        print('True utr_coef diff: {}'.format(np.abs(np.exp(model.vars_evals['log_utr_coef']) - true_utr_coef)))

        print('Label r2: {}'.format(model.r2))

def test_originallet7_model(num_genes, num_mirs, num_max_sites, maxiter):

    # generate random data
    np.random.seed(0)
    utr_lengths = (np.random.randint(5000, size=num_genes) / 2000).reshape([-1, 1])

    # get a random number of sites per mRNA/miRNA interaction
    features = np.zeros([num_genes, num_mirs, num_max_sites])
    for i in range(num_genes):
        for j in range(num_mirs):
            nsites = np.random.choice(num_max_sites)
            features[i,j,:nsites] = np.random.rand(nsites)

    mask = (features != 0).astype(int)

    true_freeAgo = np.random.random(num_mirs).reshape([1, -1, 1])
    true_freeAgolet7 = true_freeAgo[0,-1,0] - 1
    true_decay = 1.5
    true_utr_coef = 0.1

    occ = sigmoid(features + true_freeAgo)
    nbound = true_decay * np.sum(occ * mask, axis=2)
    nbound_endog = true_utr_coef * utr_lengths

    pred_endog = np.log1p(nbound_endog)
    pred_transfect = np.log1p(nbound_endog + nbound)

    labels = -1 * (pred_transfect - pred_endog)

    occ_let7 = sigmoid(features[:, -1, :] + true_freeAgolet7)
    nbound_let7 = true_decay * np.sum(occ_let7 * mask[:, -1, :], axis=1)
    labels2 = -1 * (np.log1p(nbound_let7 + nbound_endog[:, -1]) - pred_endog[:, -1])

    print(labels[:, -1].shape)

    labels[:, -1] -= labels2

    tf.reset_default_graph()

    features_tensor = tf.placeholder(tf.float32, shape=[None, None, None], name='features')
    mask_tensor = tf.placeholder(tf.float32, shape=[None, None, None], name='mask')
    utrlen_tensor = tf.placeholder(tf.float32, shape=[None, 1], name='utr_len')
    labels_tensor = tf.placeholder(tf.float32, shape=[None, None], name='labels')

    data = {
        'ka_vals': features_tensor,
        'mask': mask_tensor,
        'utr_len': utrlen_tensor,
        'labels': labels_tensor
    }

    feed_dict = {
            features_tensor: features,
            mask_tensor: mask,
            utrlen_tensor: utr_lengths,
            labels_tensor: labels
        }

    model = models.OriginalModelLet7(num_mirs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess, data, feed_dict, maxiter)

        print('True freeAgo diff: {}'.format(np.sum(np.abs(model.vars_evals['freeAgo'] - true_freeAgo))))
        print('True freeAgo_let7 diff: {}'.format(np.abs(model.vars_evals['freeAgo_init_let7'] - true_freeAgolet7)))
        print('True decay diff: {}'.format(np.abs(np.exp(model.vars_evals['log_decay']) - true_decay)))
        print('True utr_coef diff: {}'.format(np.abs(np.exp(model.vars_evals['log_utr_coef']) - true_utr_coef)))

        print('Label r2: {}'.format(model.r2))

    

# test_linear_model(5000,17,50,24,200)
# test_linear_model(100,17,10,10,200)
# test_boundedlinear_model(100,17,10,10,200)
# test_sigmoid_model(100, 5, 12, 5, 5, 2000)
# test_sigmoid_model(5000, 5, 50, 5, 5, 2000)
# test_doublesigmoid_model(100, 5, 12, 5, 5, 2000)
# test_doublesigmoid_model(5000, 5, 50, 5, 5, 2000)
# test_sigmoidfreeago_model(100, 5, 12, 5, 5, 2000)
# test_sigmoidfreeago_model(5000, 5, 50, 5, 5, 2000)
# test_doublesigmoidfreeago_model(100, 5, 12, 5, 5, 2000)
# test_doublesigmoidfreeago_model(5000, 5, 50, 5, 5, 2000)
# test_doublesigmoidfreeagolet7_model(100, 5, 12, 5, 5, 2000)
# test_doublesigmoidfreeagolet7_model(5000, 5, 50, 5, 5, 2000)
# test_original_model(100, 5, 12, 2000) 
test_originallet7_model(100, 5, 12, 2000)  

