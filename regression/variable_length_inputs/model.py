import tensorflow as tf

import helpers

def inference_2layer(image, label, keep_prob, params):
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
                                            4, 4, 'conv4x4', padding='VALID', act=tf.nn.elu)

    print(layer1)
    var_dict.update(vars1)


    with tf.name_scope('dropout1'):
        dropout1 = tf.nn.dropout(layer1, keep_prob)

    in_channels = out_channels
    out_channels = params['HIDDEN3']
    layer3, vars3 = helpers.make_convolution_layer(dropout1, params['MIRLEN'], params['SEQLEN'], in_channels, out_channels,
                                                   params['MIRLEN'], params['SEQLEN'], 'convlayer3', padding='SAME', act=tf.nn.elu)

    print(layer3)
    var_dict.update(vars3)

    # add dropout
    with tf.name_scope('dropout'):
        dropout = tf.nn.dropout(layer3, keep_prob)

    # reshape to 1D tensor
    layer_flat_dim = dropout.get_shape().as_list()
    layer_flat_dim = layer_flat_dim[1] * layer_flat_dim[2] * layer_flat_dim[3]
    layer_flat = tf.reshape(dropout, [-1, layer_flat_dim])

    # add last layer
    in_channels = layer_flat_dim
    out_channels = params['OUT_NODES']
    layer4, vars4 = helpers.make_fullyconnected_layer(layer_flat, in_channels, out_channels,
                                                    'fullyconnected', act=tf.identity)

    var_dict.update(vars4)

    # layer4_logfc, vars4_logfc = helpers.make_fullyconnected_layer(layer_flat, in_channels, out_channels,
    #                                                 'layer_logfc', act=tf.nn.elu)

    # var_dict.update(vars4_logfc)

    b_initial = tf.constant(0.0, shape=[1])
    b_final = tf.Variable(b_initial, name='bias_final')

    w_initial = tf.constant(1.0, shape=[1,1])
    w_final = tf.Variable(w_initial, name='weight_final')

    var_dict['logfc_weight'] = w_final
    var_dict['logfc_bias'] = b_final

    with tf.name_scope('layer_logfc'):
        layer4_logfc = tf.nn.relu(tf.matmul(layer4, w_final) + b_final, name='activation')

    print(layer4)
    print(layer4_logfc)

    weight_regularize = tf.mul(tf.nn.l2_loss(var_dict['conv4x4_weight']) \
                        + tf.nn.l2_loss(var_dict['convlayer3_weight']) \
                        + tf.nn.l2_loss(var_dict['fullyconnected_weight']), params['LAMBDA'])

    # prepare training steps and log writers
    train_step, accuracy, loss = helpers.make_train_step_regression(params['ERROR_MODEL'],
                                                                    layer4,
                                                                    label,
                                                                    params['STARTING_LEARNING_RATE'],
                                                                    weight_regularize)


    train_step_logfc, accuracy_logfc, loss_logfc = helpers.make_train_step_regression(params['ERROR_MODEL'],
                                                                                      layer4_logfc,
                                                                                      label,
                                                                                      params['STARTING_LEARNING_RATE_LOGFC'],
                                                                                      weight_regularize)

    # outfile.close()

    return train_step, loss, layer4, var_dict, train_step_logfc, loss_logfc, layer4_logfc


def inference_3layer(image, label, keep_prob, params):
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
                                            4, 4, 'conv4x4', padding='VALID', act=tf.nn.elu)

    print(layer1)
    var_dict.update(vars1)


    with tf.name_scope('dropout1'):
        dropout1 = tf.nn.dropout(layer1, keep_prob)

    ## LAYER 2 ##

    # convolution layer for each 3x3 paired nucleotides
    in_channels = out_channels
    out_channels = params['HIDDEN2']
    layer2, vars2 = helpers.make_convolution_layer(dropout1, 2, 2, in_channels, out_channels,
                                              1, 1, 'convlayer2', padding='SAME', act=tf.nn.elu)

    with tf.name_scope('dropout2'):
        dropout2 = tf.nn.dropout(layer2, keep_prob)

    print(layer2)
    var_dict.update(vars2)

    ## LAYER 3 ##
    in_channels = out_channels
    out_channels = params['HIDDEN3']
    layer3, vars3 = helpers.make_convolution_layer(dropout2, params['MIRLEN'], params['SEQLEN'], in_channels, out_channels,
                                                   params['MIRLEN'], params['SEQLEN'], 'convlayer3', padding='SAME', act=tf.nn.elu)

    print(layer3)
    var_dict.update(vars3)

    # add dropout
    with tf.name_scope('dropout3'):
        dropout3 = tf.nn.dropout(layer3, keep_prob)

    # reshape to 1D tensor
    layer_flat_dim = dropout3.get_shape().as_list()
    layer_flat_dim = layer_flat_dim[1] * layer_flat_dim[2] * layer_flat_dim[3]
    layer_flat = tf.reshape(dropout3, [-1, layer_flat_dim])

    # add last layer
    in_channels = layer_flat_dim
    out_channels = params['OUT_NODES']
    layer4, vars4 = helpers.make_fullyconnected_layer(layer_flat, in_channels, out_channels,
                                                    'fullyconnected', act=tf.identity)

    var_dict.update(vars4)

    # # add logfc layer
    # b_initial = tf.constant(0.0, shape=[1])
    # b_final = tf.Variable(b_initial, name='bias_final')

    # w_initial = tf.constant(1.0, shape=[1,1])
    # w_final = tf.Variable(w_initial, name='weight_final')

    # var_dict['logfc_weight'] = w_final
    # var_dict['logfc_bias'] = b_final

    # with tf.name_scope('layer_logfc'):
    #     layer4_logfc = tf.nn.elu(tf.matmul(layer4, w_final) + b_final, name='activation')

    layer4_logfc, vars4_logfc = helpers.make_fullyconnected_layer(layer_flat, in_channels, out_channels,
                                                    'layer_logfc', act=tf.nn.elu)

    var_dict.update(vars4_logfc)


    print(layer4)
    print(layer4_logfc)

    weight_regularize = tf.mul(tf.nn.l2_loss(var_dict['conv4x4_weight']) \
                        + tf.nn.l2_loss(var_dict['convlayer2_weight']) \
                        + tf.nn.l2_loss(var_dict['convlayer3_weight']) \
                        + tf.nn.l2_loss(var_dict['fullyconnected_weight']), params['LAMBDA'])

    # prepare training steps and log writers
    train_step, accuracy, loss = helpers.make_train_step_regression(params['ERROR_MODEL'],
                                                                    layer4,
                                                                    label,
                                                                    params['STARTING_LEARNING_RATE'],
                                                                    weight_regularize)


    train_step_logfc, accuracy_logfc, loss_logfc = helpers.make_train_step_regression(params['ERROR_MODEL'],
                                                                                      layer4_logfc,
                                                                                      label,
                                                                                      params['STARTING_LEARNING_RATE_LOGFC'],
                                                                                      weight_regularize)

    return train_step, loss, layer4, var_dict, train_step_logfc, loss_logfc, layer4_logfc
