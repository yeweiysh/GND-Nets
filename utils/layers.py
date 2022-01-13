import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
conv1d = tf.layers.conv1d


def NeuralDiffSLP(inputs, output_dim, norm_mat, activation, in_drop=0.0):
    with tf.name_scope('neural_diff_slp'):
        if in_drop != 0.0:
            inputs = tf.nn.dropout(inputs, 1.0 - in_drop)

        inputs = tf.expand_dims(inputs, axis=0)
        seq_fts = tf.layers.conv1d(inputs, output_dim, 1, use_bias=False)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        hops = 20
        seq_fts = tf.squeeze(seq_fts, axis=0)
        n, m = seq_fts.get_shape()
        n = int(n)
        
        H1 = tf.sparse_tensor_dense_matmul(norm_mat, seq_fts)
        subspace = list()
        subspace.append(H1)

        for i in range(hops-1):
            Hi = tf.sparse_tensor_dense_matmul(norm_mat, subspace[i])
            subspace.append(Hi)

        #aggregation
        H = tf.stack(subspace, axis=0)
        H = tf.reshape(H, [hops, n*output_dim])
        H = tf.transpose(H)       
        H = tf.expand_dims(H, axis=0)
        H = tf.layers.conv1d(H, 1, 1, use_bias=False, name="conv")
        H = tf.transpose(H)
        vals = tf.reshape(H, [1, n, output_dim])
        
        #ret = tf.contrib.layers.bias_add(vals)
        bias = tf.get_variable(shape=[output_dim,],initializer=tf.initializers.zeros,name="bias0")
        ret = tf.nn.bias_add(vals,bias)

        return activation(ret)  # activation


def NeuralDiffMLP(inputs, output_dim, norm_mat, activation, in_drop=0.0):
    with tf.name_scope('neural_diff_mlp'):
        if in_drop != 0.0:
            inputs = tf.nn.dropout(inputs, 1.0 - in_drop)

        inputs = tf.expand_dims(inputs, axis=0)
        seq_fts = tf.layers.conv1d(inputs, output_dim, 1, use_bias=False)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        hops = 20
        seq_fts = tf.squeeze(seq_fts, axis=0)
        n, m = seq_fts.get_shape()
        n = int(n)

        H1 = tf.sparse_tensor_dense_matmul(norm_mat, seq_fts)
        subspace = list()
        subspace.append(H1)

        for i in range(hops-1):
            Hi = tf.sparse_tensor_dense_matmul(norm_mat, subspace[i])
            subspace.append(Hi)

        #aggregation
        H = tf.stack(subspace, axis=0)
        H = tf.reshape(H, [hops, n*output_dim])
        H = tf.transpose(H)
        H = tf.expand_dims(H, axis=0)
        H = tf.layers.conv1d(H, 32, 1, use_bias=False, activation="relu")
        #H = tf.layers.conv1d(H, 16, 1, use_bias=False, activation="relu")
        H = tf.layers.conv1d(H, 1, 1, use_bias=False)
        H = tf.transpose(H)
        vals = tf.reshape(H, [1, n, output_dim])
        
        #ret = tf.contrib.layers.bias_add(vals)
        bias = tf.get_variable(shape=[output_dim,],initializer=tf.initializers.zeros,name="bias")
        ret = tf.nn.bias_add(vals,bias)

        return activation(ret)  # activation
    

def NeuralDiffDS(inputs, output_dim, norm_mat, activation, in_drop=0.0):
    with tf.name_scope('neural_diff_ds'):
        if in_drop != 0.0:
            inputs = tf.nn.dropout(inputs, 1.0 - in_drop)

        inputs = tf.expand_dims(inputs, axis=0)
        seq_fts = tf.layers.conv1d(inputs, output_dim, 1, use_bias=False)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        hops = 2
        H0 = tf.squeeze(seq_fts, axis=0)
        subspace = list()
        subspace.append(H0)

        for i in range(hops-1):
            H = subspace[-1]
            H1 = tf.expand_dims(H, axis=0)
            H1 = tf.layers.conv1d(H1, output_dim, 1, use_bias=False, activation="relu")
            H1 = tf.layers.conv1d(H1, output_dim, 1, use_bias=False, activation="relu")
            H1 = tf.squeeze(H1, axis=0)
            for j in range(10):
                H1 = tf.sparse_tensor_dense_matmul(norm_mat, H1)
            Hi = H - H1
            subspace.append(Hi)

        vals = subspace[-1]
        vals = tf.expand_dims(vals, axis=0)

        bias = tf.get_variable(shape=[output_dim,],initializer=tf.initializers.zeros,name="bias")
        ret = tf.nn.bias_add(vals,bias)

        return activation(ret)  # activation


def linear_layer(inputs, output_dim, activation, in_drop=0.0):
    with tf.name_scope('dense'):
        if in_drop != 0.0:
            inputs = tf.nn.dropout(inputs, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(inputs, output_dim, 1, use_bias=False)

        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = seq_fts
        #ret = tf.contrib.layers.bias_add(vals)
        bias = tf.get_variable(shape=[output_dim,],initializer=tf.initializers.zeros,name="bias2")
        ret = tf.nn.bias_add(vals,bias)

        return activation(ret)  # activation

