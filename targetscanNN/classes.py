import numpy as np
import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# create dataset object that holds features and labels and cycles through the data in batches
class Dataset(object):

    def __init__(self, features, labels):
        assert (len(features) == len(labels))
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.index = 0
        self.size = len(labels)
    
    def next_batch(self, batch_size):
        old_index = self.index
        new_index = self.index + batch_size
        self.index = new_index % self.size
        if new_index <= self.size:
            return (self.features[old_index: new_index], self.labels[old_index: new_index])
        else:
            subfeatures = np.concatenate([self.features[old_index:], self.features[:self.index]])
            sublabels = np.concatenate([self.labels[old_index:], self.labels[:self.index]])
            return (subfeatures, sublabels)
    
    def reset_index(self):
        self.index = 0


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

# make 2D neural network object
class NeuralNet2D(object):
    
    def __init__(self, sess, dim1, dim2, label_size):
        """Initiate object with placeholders for data and layers"""
        self.sess = sess
        self.x = tf.placeholder(tf.float32, shape=[None, dim1*dim2])
        self.y_ = tf.placeholder(tf.float32, shape=[None, label_size])
        
        self.x_image = tf.summary.image('images', tf.reshape(self.x, [-1, dim1, dim2, 1]))
        
        self.layers = [tf.reshape(self.x, [-1,dim1,dim2,1])]
        self.layer_index = 0
    
    def add_layer(self, input_tensor, weight_dim, output_dim, layer_name, preact, act):
        """Reusable code for making a simple neural net layer.

        It does a matrix multiply, bias add, and then uses an activation function to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # add a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # create variables for weights and biases
            with tf.name_scope('weights'):
                weights = weight_variable(weight_dim)
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = preact(input_tensor, weights, biases)
                tf.summary.histogram('pre_activations', preactivate)
            
            out_layer = act(preactivate, name='activation')
            tf.summary.histogram('activations', out_layer)
            
            return out_layer
    
    def add_convolution(self, layer_name, dim1, dim2, stride1, stride2,
                        output_channels, padding='SAME', act=tf.nn.relu):
        """Create a convolution layer. 
        Dim1 and dim2 specify the size of the box for the convolution
        """
        # get the current layer and determine how many output channels it has
        current_layer = self.layers[self.layer_index]

        # this becomes the new input channel size
        input_channels = current_layer.get_shape().as_list()[-1]
        
        # specify the function that creates a new tensor
        preact = lambda tensor, weights, biases: (tf.nn.conv2d(tensor,
                                                               weights,
                                                               strides=[1, stride1, stride2, 1],
                                                               padding=padding) + biases)

        # create the new tensor
        new_layer = self.add_layer(current_layer,
                                   [dim1, dim2, input_channels, output_channels],
                                   output_channels, layer_name, preact, act)

        self.layers.append(new_layer)
        self.layer_index += 1

    def add_max_pool(self, dim1, dim2, stride1, stride2, padding='SAME'):
        """Adds a max pooling layer."""
        self.layers.append(tf.nn.max_pool(self.layers[self.layer_index],
                                            ksize=[1, dim1, dim2, 1],
                                            strides=[1, stride1, stride2, 1], padding=padding))
        
        self.layer_index += 1

    def add_fully_connected(self, layer_name, output_channels, act=tf.nn.relu):
        """Adds a fully connected layer."""

        current_layer = self.layers[self.layer_index]
        dim = current_layer.get_shape().as_list()
        dim = dim[1] * dim[2] * dim[3]
        
        preact = lambda tensor, weights, biases: tf.matmul(tensor, weights) + biases
    
        new_layer = self.add_layer(tf.reshape(current_layer, [-1, dim]),
                                   [dim, output_channels],
                                   output_channels, layer_name, preact, act)
            
        self.layers.append(new_layer)
        self.layer_index += 1
    
    def add_dropout(self, layer_name, num_nodes):
        """Adds a layer for dropout nodes in order to reduce overfitting"""
        current_layer = self.layers[self.layer_index]
        dim = current_layer.get_shape().as_list()
        self.keep_prob = tf.placeholder(tf.float32)
        
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = weight_variable([dim[-1], num_nodes])
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_variable([num_nodes])
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                out_layer = tf.matmul(tf.nn.dropout(current_layer, self.keep_prob), weights) + biases

            tf.summary.histogram('activations', out_layer)

        self.layers.append(out_layer)
        self.layer_index += 1
    
    def make_train_step(self, problem_type, logdir):
        current_layer = self.layers[self.layer_index]
        if problem_type == 'classification':
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(current_layer,
                                                                                   self.y_))
            with tf.name_scope('train'):
                self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            
            correct_prediction = tf.equal(tf.argmax(current_layer,1), tf.argmax(self.y_,1))
            
            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
        elif problem_type == 'regression':
            SS_err = tf.reduce_sum(tf.square(tf.sub(current_layer, self.y_)))
            SS_tot = tf.reduce_sum(tf.square(tf.sub(self.y_, tf.reduce_mean(self.y_))))
            R_2 = tf.sub(tf.cast(1.0, tf.float32), tf.div(SS_err, SS_tot))

            with tf.name_scope('train'):
                self.train_step = tf.train.AdamOptimizer(1e-4).minimize(SS_err)
            with tf.name_scope('accuracy'):
                self.accuracy = R_2
        
        else:
            print('problem_type must be \'classification\' or \'regression\'')
            
        tf.summary.scalar('accuracy', self.accuracy)
        
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(logdir + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(logdir + '/test')
    
    def train_model(self, train, test, num_epoch=20000, batch_size=50, 
                    report_int=1000, keep_prob_train=0.5):

        # initialize variables
        self.sess.run(tf.global_variables_initializer())
        

        # train epochs
        for i in range(num_epoch):
            batch = train.next_batch(batch_size)
            if i%report_int == 0:
                
                acc, summary = self.sess.run([self.accuracy, self.merged], feed_dict={self.x: test.features,
                                                                                 self.y_: test.labels,
                                                                                 self.keep_prob: 1.0})
                self.test_writer.add_summary(summary, i)
                print('Accuracy at step %s: %s' % (i, acc))
            
            else:
                _, summary = self.sess.run([self.train_step, self.merged], feed_dict={self.x: batch[0],
                                                                                 self.y_: batch[1],
                                                                                 self.keep_prob: keep_prob_train})
                self.train_writer.add_summary(summary, i)

        print("test accuracy %g"%self.accuracy.eval(feed_dict={self.x: test.features,
                                                               self.y_: test.labels,
                                                               self.keep_prob: 1.0}))
