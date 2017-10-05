import os
import tensorflow as tf

import config
import helpers

def inference(image, label, baseline, input_channels, output_channels, baseline_channels,
                hidden1, hidden2, hidden3, keep_prob, starting_rate, error_model, logdir, data_weights=None):
    """Build the model up to where it may be used for inference.
    Args:
        inputs: input data.
        input_channels: number of channels in input data.
        output_channels: number of channels in label.
        hidden1: Size of the first hidden layer.
        hidden2: Size of the second hidden layer.
        hidden3: Size of the second hidden layer.
        starting_rate: starting learning rate.
        error_model: either 'l2' or 'poisson'
    Returns:
        Output tensor.
        Accuracy.
        Loss.
        Predictions.
    """

    var_dict = {}
    # outfile = open(os.path.join(logdir, 'layers.txt'), 'w')

    ## LAYER 1 ##

    # convolution layer for each 4x4 box representing two paired nucleotides
    in_channels = input_channels
    out_channels = hidden1
    layer1, vars1 = helpers.make_convolution_layer(image, 4, 4, in_channels, out_channels,
                                            4, 4, 'conv4x4', padding='VALID', act=tf.nn.relu)

    print(layer1)

    # outfile.write(layer1)
    # outfile.write('\n')

    ## LAYER 2 ##

    # convolution layer for each 4x4 box
    in_channels = out_channels
    out_channels = hidden2
    layer2, vars2 = helpers.make_convolution_layer(layer1, 2, 2, in_channels, out_channels,
                                              1, 1, 'convlayer2', padding='SAME', act=tf.nn.relu)

    print(layer2)
    # outfile.write(layer2)
    # outfile.write('\n')

    # reshape to 1D tensor
    # layer_flat_dim = layer2.get_shape().as_list()
    # layer_flat_dim = layer_flat_dim[1] * layer_flat_dim[2] * layer_flat_dim[3]
    # layer_flat = tf.reshape(layer2, [-1, layer_flat_dim])

    # print(layer_flat)

    ## LAYER 3 ##

    # add fully connected layer
    # in_channels = layer_flat_dim
    # out_channels = hidden3
    # layer3 = helpers.make_fullyconnected_layer(layer_flat, in_channels, out_channels, 'fullyconnected1', act=tf.identity)

    in_channels = out_channels
    out_channels = hidden3
    layer3, vars3 = helpers.make_convolution_layer(layer2, config.params['MIRLEN'], config.params['SEQLEN'], in_channels, out_channels,
                                                   config.params['MIRLEN'], 1, 'convlayer3', padding='SAME', act=tf.nn.relu)

    print(layer3)
    # outfile.write(layer3)
    # outfile.write('\n')

    var_dict.update(vars1)
    var_dict.update(vars2)
    var_dict.update(vars3)

    # # add dropout
    # with tf.name_scope('dropout'):
    #     tf.summary.scalar('dropout_keep_probability', keep_prob)
    #     dropout = tf.nn.dropout(layer3, keep_prob)

    # with tf.name_scope('pool_layer'):
    #     layer3_pool = tf.nn.max_pool(dropout, [1,1,12,1], [1,1,12,1], padding='SAME')

    # print(layer3_pool)
    # outfile.write(layer3_pool)
    # outfile.write('\n')

    # reshape to 1D tensor
    layer_flat_dim = layer3.get_shape().as_list()
    layer_flat_dim = layer_flat_dim[1] * layer_flat_dim[2] * layer_flat_dim[3]
    layer_flat = tf.reshape(layer3, [-1, layer_flat_dim])


    if baseline_channels > 0:
        layer_flat_plus_baseline = tf.concat(1, [layer_flat, baseline])
        print(layer_flat_plus_baseline)
        ## LAYER 4 ##

        # add last layer
        in_channels = layer_flat_dim + baseline_channels
        out_channels = output_channels
        with tf.name_scope('final_layer'):
            layer4 = helpers.make_fullyconnected_layer(layer_flat_plus_baseline, in_channels, out_channels,
                                                        'fullyconnected', act=tf.identity)

    else:
        # add last layer
        in_channels = layer_flat_dim
        out_channels = output_channels
        with tf.name_scope('final_layer'):
            layer4 = helpers.make_fullyconnected_layer(layer_flat, in_channels, out_channels,
                                                        'fullyconnected', act=tf.identity)


    # outfile.write(layer4)
    # outfile.write('\n')
    # print(layer4_logfc)

    # prepare training steps and log writers
    train_step, accuracy, loss = helpers.make_train_step_regression(error_model, layer4, label, starting_rate, data_weights=data_weights)

    # outfile.close()

    return train_step, accuracy, loss, layer4, var_dict
