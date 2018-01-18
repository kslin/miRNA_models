import numpy as np
import pandas as pd
import tensorflow as tf

import helpers
import model
import train_model

params = {
                'MIRLEN': 20,
                'SEQLEN': 12,
                'IN_NODES': 1,
                'OUT_NODES': 1,
                'HIDDEN1': 4,
                # 'HIDDEN2': 8,
                'HIDDEN3': 32,
                'ERROR_MODEL': 'l2',
                'MAX_EPOCH': 10,
                'BATCH_SIZE': 200,
                'LOGFC_BATCH_SIZE': 200,
                'REPORT_INT': 100,
                'KEEP_PROB_TRAIN': 0.5,
                'TEST_SIZE': 5000,
                'RESTORE_FROM': None,
                'LAMBDA': 0.1,
                'LOG_SCALE': False,
                'NCOLS': 1,
                'NUM_CLASSES': 5,
                'STARTING_LEARNING_RATE_L2': 0.001,
                'STARTING_LEARNING_RATE_DISC': 0.001,
        }

# create placeholders for data
x = tf.placeholder(tf.float32, shape=[None, 16 * params['MIRLEN'] * params['SEQLEN']], name='x')
y = tf.placeholder(tf.float32, shape=[None, params['OUT_NODES']], name='y')
classes = tf.placeholder(tf.int32, shape=[None, params['NUM_CLASSES']], name='s')
x_image = tf.reshape(x, [-1, 4*params['MIRLEN'], 4*params['SEQLEN'], 1])
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
phase_train = tf.placeholder(tf.bool, name='phase_train')

model.inference(x_image, y, classes, keep_prob, phase_train, params)