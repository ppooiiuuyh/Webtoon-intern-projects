import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np
from utils import *

#weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_init = tf.initializers.he_normal()

"""
pad = (k-1) // 2
size = (I-k+1+2p) // s
"""
weight_regularizer = None



def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)


    return w_norm




def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=True, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias :
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)



        return x

def deconv(x, channels, kernel=4, stride=2, use_bias=True, sn=True, scope='deconv_0'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()
        output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]
        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init, regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')

            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                                           strides=stride, padding='SAME', use_bias=use_bias)

        return x



def resblock(x_init, channels,  scope='resblock_0'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = tf.pad(x_init, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=3, kernel_initializer=weight_init, strides=1)
            x = batch_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=3, kernel_initializer=weight_init, strides=1)
            x = batch_norm(x)

        return x + x_init

def resNext(x_init, channels, num_card=32, scope='resblock_0'):
    # x = conv(x, channel/2/cardinality, kernel=4, stride=2, pad=1, scope='conv_' + str(i))
    with tf.variable_scope(scope):
        groups = []
        for c in range(num_card):
            with tf.variable_scope("group_{}".format(c)):
                with tf.variable_scope("first"):
                    r = tf.layers.conv2d(x_init, channels / 2 / num_card, kernel_size=1, strides=1, padding="SAME",  kernel_initializer=weight_init)
                with tf.variable_scope("second"):
                    r = tf.layers.conv2d(r, channels / 2 / num_card, kernel_size=3, strides=1, padding="SAME",  kernel_initializer=weight_init)
                    groups.append(r)

        with tf.variable_scope("merge"):
            resnext_block = tf.concat(groups,axis=-1)

        with tf.variable_scope("out"):
            x = tf.layers.conv2d(resnext_block, channels, kernel_size=1, strides=1, padding="SAME",kernel_initializer=weight_init) + x_init
    return x


def resNext_CWD(x_init, channels, num_use,idx ,num_card=32, scope='resblock_0'):
    merge_list = combination(num_card, num_use)[idx]

    # x = conv(x, channel/2/cardinality, kernel=4, stride=2, pad=1, scope='conv_' + str(i))
    with tf.variable_scope(scope):
        groups = []
        for c in range(num_card):
            with tf.variable_scope("group_{}".format(c)):
                with tf.variable_scope("first"):
                    r = tf.layers.conv2d(x_init,channels//2//num_card, kernel_size=1, strides=1, padding="SAME",  kernel_initializer=weight_init)
                with tf.variable_scope("second"):
                    r = tf.layers.conv2d(r,channels//2//num_card, kernel_size=3, strides=1, padding="SAME",  kernel_initializer=weight_init)
                    if c in merge_list:
                        groups.append(r)

        with tf.variable_scope("merge"):
            resnext_block = tf.concat( groups,axis=-1)

        with tf.variable_scope("out"):
            x = tf.layers.conv2d(resnext_block, channels, kernel_size=1, strides=1, padding="SAME",kernel_initializer=weight_init) + x_init
    return x





def flatten(x) :
    return tf.layers.flatten(x)



def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def sigmoid(x):
    return tf.sigmoid(x)


def tanh(x):
    return tf.tanh(x)


def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-05, center=True, scale=True, updates_collections=None, is_training=is_training, scope=scope)


def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss



def L2_loss(x, y):
    loss = tf.reduce_mean(tf.square(x - y))

    return loss

def TV_loss(images):
    return tf.reduce_mean(tf.image.total_variation(images))

def discriminator_loss(real, fake):

    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    loss = real_loss + fake_loss

    return loss


def generator_loss(fake):

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    return loss

def discriminator_loss_wasserstein(real,fake):
    loss = tf.reduce_mean(fake) - tf.reduce_mean(real)
    return loss

def generator_loss_wasserstein(fake):
    loss = -tf.reduce_mean(fake)
    return loss

