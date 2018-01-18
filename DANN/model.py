import tensorflow as tf

import helpers

def inference(image, keep_prob, phase_train, params):
    """Build the model up to where it may be used for inference.
    Args:
        inputs: input data.
        label: output data,
        keep_prob: tensor for dropout probability
        params: dictionary of parameters for the model
    Returns:
        Output tensor.
        Accuracy.
        Loss.
        Predictions.
    """

    var_dict = {}

    ## LAYER 1 ##

    # convolution layer for each 4x4 box representing two paired nucleotides
    in_channels = params['IN_NODES']
    out_channels = params['HIDDEN1']
    layer1, vars1 = helpers.make_convolution_layer(image, 4, 4, in_channels, out_channels,
                                            4, 4, 'conv4x4', phase_train, padding='VALID', act=tf.nn.relu)

    print(layer1)
    var_dict.update(vars1)


    # with tf.name_scope('dropout1'):
    #     dropout1 = tf.nn.dropout(layer1, keep_prob)


    # ## LAYER 2 ##

    # convolution layer for each 3x3 paired nucleotides
    in_channels = out_channels
    out_channels = params['HIDDEN2']
    layer2, vars2 = helpers.make_convolution_layer(layer1, 2, 2, in_channels, out_channels,
                                              1, 1, 'convlayer1_1', phase_train, padding='VALID', act=tf.nn.relu)

    # print(layer1_1)
    # var_dict.update(vars1_1)

    # # convolution layer for each 3x3 paired nucleotides
    # in_channels = out_channels
    # out_channels = params['HIDDEN2']
    # layer1_2, vars1_2 = helpers.make_convolution_layer(layer1_1, 3, 3, in_channels, out_channels,
    #                                           1, 1, 'convlayer1_2', phase_train, padding='VALID', act=tf.nn.relu)

    # print(layer1_2)
    # var_dict.update(vars1_2)

    # convolution layer for each 3x3 paired nucleotides
    # in_channels = out_channels
    # out_channels = params['HIDDEN2']
    # layer2, vars2 = helpers.make_convolution_layer(layer1_1, 3, 3, in_channels, out_channels,
    #                                           1, 1, 'convlayer2', phase_train, padding='VALID', act=tf.nn.relu)

    layer2_dims = layer2.get_shape().as_list()
    new_dim1, new_dim2 = layer2_dims[1], layer2_dims[2]

    # # with tf.name_scope('dropout2'):
    # #     dropout2 = tf.nn.dropout(layer2, keep_prob)

    print(layer2)
    var_dict.update(vars2)

    in_channels = out_channels
    out_channels = params['HIDDEN3']
    layer3, vars3 = helpers.make_convolution_layer(layer2, new_dim1, new_dim2, in_channels, out_channels,
                                                   new_dim1, new_dim2, 'convlayer3', phase_train, padding='SAME', act=tf.nn.relu)

    print(layer3)
    var_dict.update(vars3)

    # add dropout
    with tf.name_scope('dropout'):
        dropout = tf.nn.dropout(layer3, keep_prob)

    # reshape to 1D tensor
    layer_flat_dim = dropout.get_shape().as_list()
    layer_flat_dim = layer_flat_dim[1] * layer_flat_dim[2] * layer_flat_dim[3]
    layer_flat = tf.reshape(dropout, [-1, layer_flat_dim])

    # print(layer_flat)
    # print(label)

    # add last layer
    in_channels = layer_flat_dim
    out_channels = params['OUT_NODES']
    layer4, vars4 = helpers.make_fullyconnected_layer(layer_flat, in_channels, out_channels,
                                                    'fullyconnected', act=tf.identity)

    var_dict.update(vars4)
    # print(layer4)
    all_vars = tf.trainable_variables()

    # layer_flat_plus_label = tf.concat([layer_flat, layer4], 1)

    # print(layer_flat_plus_label.shape)

    # disc_layer1, varsD1 = helpers.make_fullyconnected_layer(layer_flat, in_channels, params['DISC1'],
    #                                                          'disc_layer1', act=tf.nn.relu)

    # discriminator, varsD2 = helpers.make_fullyconnected_layer(disc_layer1, params['DISC1'], params['NUM_CLASSES'],
    #                                                          'discriminator', act=tf.identity)

    discriminator, varsD2 = helpers.make_fullyconnected_layer(layer_flat, in_channels, params['NUM_CLASSES'],
                                                             'discriminator', act=tf.identity)

    # var_dict.update(varsD1)
    var_dict.update(varsD2)
    # print(discriminator)
    D_vars = list(varsD2.values())# + list(varsD2.values())

    return layer_flat, layer4, discriminator, var_dict, all_vars, D_vars

    

    # with tf.name_scope('loss'):
    #     loss_disc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=image_class, logits=discriminator))
    #     correct_prediction = tf.equal(tf.argmax(image_class,1), tf.argmax(discriminator,1))
    #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #     # loss_l2 = tf.nn.l2_loss(tf.multiply(tf.subtract(layer4, label), mask)) / params['BATCH_SIZE']
    #     loss_l2 = tf.reduce_mean(tf.multiply(tf.subtract(layer4, label)**2, mask))
    #     loss = loss_l2 - tf.multiply(loss_disc, params['LAMBDA'])


    # # with tf.name_scope('train_all'):
    # #     optimizer = tf.train.AdamOptimizer(params['STARTING_LEARNING_RATE_L2'])
    # #     grads = tf.gradients(loss, all_vars)
    # #     D_grads = tf.gradients(loss_disc, D_vars)
    # #     train_step_all = optimizer.apply_gradients(zip(grads+D_grads, all_vars+D_vars))

    # with tf.name_scope('train_all'):
    #     extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #     with tf.control_dependencies(extra_update_ops):
    #         optimizer = tf.train.AdamOptimizer(params['STARTING_LEARNING_RATE_L2'])
    #         grads = tf.gradients(loss, all_vars)
    #         train_step_all = optimizer.apply_gradients(zip(grads, all_vars))

    # with tf.name_scope('train_D'):
    #     D_optimizer = tf.train.AdamOptimizer(params['STARTING_LEARNING_RATE_DISC'])
    #     D_grads = tf.gradients(loss_disc, D_vars)
    #     train_step_disc = D_optimizer.apply_gradients(zip(D_grads, D_vars))


    

    # return layer_flat, train_step_all, train_step_disc, loss_l2, loss_disc, layer4, accuracy, var_dict, all_vars, D_vars

