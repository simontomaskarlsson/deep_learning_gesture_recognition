import tensorflow as tf
import cv2

def alexnet(x):
    pass

def twolayers(x, nrOfClasses):
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    in_filters = 1
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 64, 64, in_filters])


    with tf.name_scope('conv1'):
        out_filters = 32
        W_conv1 = weight_variable([5, 5, in_filters, out_filters])
        b_conv1 = bias_variable([out_filters])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        in_filters = out_filters

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        out_filters = 32
        W_conv2 = weight_variable([5, 5, in_filters, out_filters])
        b_conv2 = bias_variable([out_filters])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        in_filters = out_filters

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 64x64 image
    # is down to 16x16x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        out_filters = 1024
        W_fc1 = weight_variable([16 * 16 * in_filters, out_filters])
        b_fc1 = bias_variable([out_filters])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 16 * in_filters])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        in_filters = out_filters

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 13 classes, one for each angle
    with tf.name_scope('fc2'):
        out_filters = nrOfClasses # Changed to two classes for humanvsnonhuman
        W_fc2 = weight_variable([in_filters, out_filters])
        b_fc2 = bias_variable([out_filters])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob

a = True

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
