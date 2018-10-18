import numpy as np
import pandas as pd
import tensorflow as tf

import config, helpers, data_objects, model_objects

HIDDEN1, HIDDEN2, HIDDEN3 = 4, 16, 32
LOGDIR = None
NUM_TRAIN = 15
baseline_df = pd.DataFrame(None)


# reset and build the neural network
tf.reset_default_graph()

# start session
with tf.Session() as sess:

    # set up model
    ConvNet = model_objects.Model(config.MIRLEN, config.SEQLEN, NUM_TRAIN, LOGDIR, baseline_df, sess)
    ConvNet.build_model(HIDDEN1, HIDDEN2, HIDDEN3, config.NORM_RATIO)

    freeAGO_init = []
    for _ in range(NUM_TRAIN):
        freeAGO_init += [-5, -7]

    freeAGO_init = np.array(freeAGO_init).reshape([1, NUM_TRAIN*2, 1])
    # print(freeAGO_init)
    ConvNet.add_repression_layers_mean_offset(config.BATCH_SIZE_REPRESSION, config.BATCH_SIZE_BIOCHEM, freeAGO_init)

