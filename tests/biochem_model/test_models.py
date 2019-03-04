import numpy as np
import pandas as pd
import tensorflow as tf

import models
import utils


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

    feed_dict = {
            features_tensor: features,
            mask_tensor: mask,
            labels_tensor: labels
        }

    model = models.LinearModel(num_features)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess, features_tensor, mask_tensor, labels_tensor, feed_dict, maxiter)

        print('True weight diff: {}'.format(np.sum(np.abs(model.vars_evals['coefs'] - true_weights))))

        print('Label r2: {}'.format(model.r2))

# test_linear_model(5000,17,50,24,200)
# test_linear_model(100,17,10,10,200)


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

    feed_dict = {
            features_tensor: features_plus_bounds,
            mask_tensor: mask,
            labels_tensor: labels
        }

    model = models.BoundedLinearModel(num_features)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess, features_tensor, mask_tensor, labels_tensor, feed_dict, maxiter)

        print('True weight diff: {}'.format(np.sum(np.abs(model.vars_evals['coefs'] - true_weights))))
        print('Label r2: {}'.format(model.r2))

        bounded_pred = model.predict(sess, features_tensor, mask_tensor, feed_dict)
        print(utils.calc_r2(labels_bounded, bounded_pred))

# test_boundedlinear_model(100,17,10,10,200)

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
    true_decay = -1.5

    weighted1 = true_decay * utils.sigmoid(np.sum(np.multiply(features[:, :, :, :num_pre_features], true_weights1), axis=3) + true_bias1)
    weighted2 = np.sum(np.multiply(features[:, :, :, num_pre_features:], true_weights2), axis=3)

    weighted = weighted1 + weighted2
    labels = np.sum(np.multiply(weighted, mask), axis=2)


    print(features.shape)
    print(mask.shape)
    print(labels.shape)

    tf.reset_default_graph()

    features_tensor = tf.placeholder(tf.float32, shape=[None, None, None, num_features], name='features')
    mask_tensor = tf.placeholder(tf.float32, shape=[None, None, None], name='nsites')
    labels_tensor = tf.placeholder(tf.float32, shape=[None, None], name='labels')

    feed_dict = {
            features_tensor: features,
            mask_tensor: mask,
            labels_tensor: labels
        }

    model = models.SigmoidModel(num_pre_features, num_post_features, num_mirs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess, features_tensor, mask_tensor, labels_tensor, feed_dict, maxiter)

        print('True weight1 diff: {}'.format(np.sum(np.abs(model.vars_evals['coefs_pre_sigmoid'] - true_weights1))))
        print('True weight2 diff: {}'.format(np.sum(np.abs(model.vars_evals['coefs_post_sigmoid'] - true_weights2))))
        print('True bias1 diff: {}'.format(np.abs(model.vars_evals['bias1'] - true_bias1)))
        print('True decay diff: {}'.format(np.abs(model.vars_evals['decay'] - true_decay)))

        print('Label r2: {}'.format(model.r2))

test_sigmoid_model(100, 5, 12, 5, 5, 2000)
# test_sigmoid_model(5000, 5, 50, 5, 5, 2000)


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

    weighted1 = true_decay * utils.sigmoid(np.sum(np.multiply(features[:, :, :, :num_pre_features], true_weights1), axis=3) + true_bias1)
    weighted2 = utils.sigmoid(np.sum(np.multiply(features[:, :, :, num_pre_features:], true_weights2), axis=3) + true_bias2)

    weighted = np.multiply(weighted1, weighted2)
    labels = np.sum(np.multiply(weighted, mask), axis=2)

    print(features.shape)
    print(mask.shape)
    print(labels.shape)

    tf.reset_default_graph()

    features_tensor = tf.placeholder(tf.float32, shape=[None, None, None, num_features], name='features')
    mask_tensor = tf.placeholder(tf.float32, shape=[None, None, None], name='nsites')
    labels_tensor = tf.placeholder(tf.float32, shape=[None, None], name='labels')

    feed_dict = {
            features_tensor: features,
            mask_tensor: mask,
            labels_tensor: labels
        }

    model = models.DoubleSigmoidModel(num_pre_features, num_post_features, num_mirs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess, features_tensor, mask_tensor, labels_tensor, feed_dict, maxiter)

        print('True weight1 diff: {}'.format(np.sum(np.abs(model.vars_evals['coefs_pre_sigmoid'] - true_weights1))))
        print('True weight2 diff: {}'.format(np.sum(np.abs(model.vars_evals['coefs_post_sigmoid'] - true_weights2))))
        print('True decay diff: {}'.format(np.abs(model.vars_evals['decay'] - true_decay)))
        print('True bias1 diff: {}'.format(np.abs(model.vars_evals['bias1'] - true_bias1)))
        print('True bias2 diff: {}'.format(np.abs(model.vars_evals['bias2'] - true_bias2)))

        print('Label r2: {}'.format(model.r2))

test_doublesigmoid_model(100, 5, 12, 5, 5, 2000)
# test_doublesigmoid_model(5000, 5, 50, 5, 5, 2000)


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
    true_decay = -1.5
    true_weight = 1.2

    weighted1 = true_decay * utils.sigmoid(np.sum(np.multiply(features[:, :, :, :num_pre_features], true_weights1), axis=3) + true_freeAgo)
    weighted2 = np.sum(np.multiply(features[:, :, :, num_pre_features:], true_weights2), axis=3)

    weighted = weighted1 + weighted2
    labels = np.sum(np.multiply(weighted, mask), axis=2)


    print(features.shape)
    print(mask.shape)
    print(labels.shape)

    tf.reset_default_graph()

    features_tensor = tf.placeholder(tf.float32, shape=[None, None, None, num_features], name='features')
    mask_tensor = tf.placeholder(tf.float32, shape=[None, None, None], name='nsites')
    labels_tensor = tf.placeholder(tf.float32, shape=[None, None], name='labels')

    feed_dict = {
            features_tensor: features,
            mask_tensor: mask,
            labels_tensor: labels
        }

    model = models.SigmoidFreeAGOModel(num_pre_features, num_post_features, num_mirs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess, features_tensor, mask_tensor, labels_tensor, feed_dict, maxiter)

        print('True weight1 diff: {}'.format(np.sum(np.abs(model.vars_evals['coefs_pre_sigmoid'] - true_weights1))))
        print('True weight2 diff: {}'.format(np.sum(np.abs(model.vars_evals['coefs_post_sigmoid'] - true_weights2))))
        print('True freeAgo diff: {}'.format(np.sum(np.abs(model.vars_evals['freeAgo'] - true_freeAgo))))
        print('True decay diff: {}'.format(np.abs(model.vars_evals['decay'] - true_decay)))

        print('Label r2: {}'.format(model.r2))

test_sigmoidfreeago_model(100, 5, 12, 5, 5, 2000)
# test_sigmoidfreeago_model(5000, 5, 50, 5, 5, 2000)


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
    true_decay = -1.5
    true_bias = -0.4

    weighted1 = true_decay * utils.sigmoid(np.sum(np.multiply(features[:, :, :, :num_pre_features], true_weights1), axis=3) + true_freeAgo)
    weighted2 = utils.sigmoid(np.sum(np.multiply(features[:, :, :, num_pre_features:], true_weights2), axis=3) + true_bias)

    weighted = np.multiply(weighted1, weighted2)
    labels = np.sum(np.multiply(weighted, mask), axis=2)

    print(features.shape)
    print(mask.shape)
    print(labels.shape)

    tf.reset_default_graph()

    features_tensor = tf.placeholder(tf.float32, shape=[None, None, None, num_features], name='features')
    mask_tensor = tf.placeholder(tf.float32, shape=[None, None, None], name='nsites')
    labels_tensor = tf.placeholder(tf.float32, shape=[None, None], name='labels')

    feed_dict = {
            features_tensor: features,
            mask_tensor: mask,
            labels_tensor: labels
        }

    model = models.DoubleSigmoidFreeAGOModel(num_pre_features, num_post_features, num_mirs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess, features_tensor, mask_tensor, labels_tensor, feed_dict, maxiter)

        print('True weight1 diff: {}'.format(np.sum(np.abs(model.vars_evals['coefs_pre_sigmoid'] - true_weights1))))
        print('True weight2 diff: {}'.format(np.sum(np.abs(model.vars_evals['coefs_post_sigmoid'] - true_weights2))))
        print('True freeAgo diff: {}'.format(np.sum(np.abs(model.vars_evals['freeAgo'] - true_freeAgo))))
        print('True decay diff: {}'.format(np.abs(model.vars_evals['decay'] - true_decay)))
        print('True bias diff: {}'.format(np.abs(model.vars_evals['bias'] - true_bias)))

        print('Label r2: {}'.format(model.r2))

test_doublesigmoidfreeago_model(100, 5, 12, 5, 5, 2000)
# test_doublesigmoidfreeago_model(5000, 5, 50, 5, 5, 2000)

