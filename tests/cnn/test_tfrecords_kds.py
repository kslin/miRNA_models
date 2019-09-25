import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import parse_data_utils

TRAIN_MIRS_KDS = ['mir1','mir124','mir155','mir7','lsy6','let7']
TRAIN_KDS_FILES = np.array(['../data/cnn/tfrecords/log_kds_balanced_{}.tfrecord'.format(mir) for mir in TRAIN_MIRS_KDS])
print("Loading training KD data from")
print(TRAIN_KDS_FILES)

kd_train_dataset = parse_data_utils._load_multiple_tfrecords(TRAIN_KDS_FILES)

# shuffle, batch, and map datasets
kd_train_dataset = kd_train_dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1000))
kd_train_dataset = kd_train_dataset.map(parse_data_utils._parse_log_kd_function, num_parallel_calls=16)

# re-balance KD data towards high-affinity sites
kd_train_dataset = kd_train_dataset.filter(parse_data_utils._filter_kds)
kd_train_dataset = kd_train_dataset.batch(1000, drop_remainder=True)

# make feedable iterators
kd_train_iterator = kd_train_dataset.make_initializable_iterator()

next_kd_batch_mirs, next_kd_batch_images, next_kd_batch_labels, next_kd_batch_keep_probs, next_kd_batch_stypes = kd_train_iterator.get_next()


with tf.Session() as sess:

    # initialize varibles
    sess.run(tf.global_variables_initializer())
    sess.run(kd_train_iterator.initializer)

    vals = sess.run(next_kd_batch_labels)

    fig = plt.figure(figsize=(7, 7))
    plt.hist(vals)
    plt.savefig('ka_hist.png')
    plt.close()