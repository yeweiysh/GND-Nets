import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import types
import numpy as np
#import tensorflow as tf
#if type(tf.contrib) != types.ModuleType:  # if it is LazyLoader
#    tf.contrib._warning = None
#if type(tf.contrib) != type(tf): tf.contrib._warning = None
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from models import *
from utils import process

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

seed = 123
tf.set_random_seed(seed)

# training params
nb_epochs = 1000
patience = 50
lr = 0.005  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = 64 # numbers of hidden units of \theta for GNDC-MLP, set to 16 for GNDC-SLP.
nonlinearity = tf.nn.elu
model = GNDC_MLP #GNDC_SLP

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.') 
flags.DEFINE_string('checkpt_file', 'pre_trained/cora/mod_cora.ckpt', 'Path to save model.')
flags.DEFINE_integer('train_size', 2, 'The size of training data.')
flags.DEFINE_integer('validation_size', 500, 'The size of validation data.')
flags.DEFINE_float('dropout', 0.6, 'Dropout rate (1 - keep probability).') # For densely-labeled graphs (e.g., 60% vertices in each class have labels), the dropout value is recommended to set to 0.5.
flags.DEFINE_integer('seed', 3, 'Random seed.')

np.random.seed(FLAGS.seed)
checkpt_file = FLAGS.checkpt_file
dataset = FLAGS.dataset


# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)

#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
#process.load_data_split(FLAGS.dataset,FLAGS.train_size,FLAGS.validation_size,shuffle=True) # 'cora', 'citeseer', 'pubmed'

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
process.load_new_data(FLAGS.dataset,FLAGS.train_size,FLAGS.validation_size,shuffle=True) # 'chameleon', 'film', 'squirrel'

#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
#process.load_data_sparse_graph(FLAGS.dataset,FLAGS.train_size,FLAGS.validation_size,shuffle=True) # 'cora_ml', 'ms academic','amazon computers','amazon photo'

features, spars = process.preprocess_features(features)
nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]
norm_mat = process.preprocess_adj_rw(adj)

with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(nb_nodes, ft_size))
        nmat_in = tf.sparse_placeholder(dtype=tf.float32)
        lbl_in = tf.placeholder(dtype=tf.int32)
        msk_in = tf.placeholder(dtype=tf.int32)
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    logits = model.inference(ftr_in, nb_classes, is_train,
                                ffd_drop,
                                norm_mat=nmat_in,
                                output_dim=hid_units,
                                activation=nonlinearity)
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        # t=time.time()
        for epoch in range(nb_epochs):
   
            tr_size = features.shape[0]

            _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                feed_dict={
                    ftr_in: features,
                    nmat_in: norm_mat,
                    lbl_in: y_train,
                    msk_in: train_mask,
                    is_train: True,
                    ffd_drop: FLAGS.dropout})
            train_loss_avg += loss_value_tr
            train_acc_avg += acc_tr

            vl_size = features.shape[0]

            loss_value_vl, acc_vl = sess.run([loss, accuracy],
                feed_dict={
                    ftr_in: features,
                    nmat_in: norm_mat,
                    lbl_in: y_val,
                    msk_in: val_mask,
                    is_train: False,
                    ffd_drop: 0.0})
            val_loss_avg += loss_value_vl
            val_acc_avg += acc_vl

            # print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
            #         (train_loss_avg, train_acc_avg,
            #         val_loss_avg, val_acc_avg))

            if val_acc_avg >= vacc_mx or val_loss_avg <= vlss_mn:
                if val_acc_avg >= vacc_mx and val_loss_avg <= vlss_mn:
                    vacc_early_model = val_acc_avg
                    vlss_early_model = val_loss_avg
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg, vacc_mx))
                vlss_mn = np.min((val_loss_avg, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    # print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                    # print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        # print(time.time()-t)
        saver.restore(sess, checkpt_file)

        ts_loss = 0.0
        ts_acc = 0.0

        loss_value_ts, acc_ts = sess.run([loss, accuracy],
            feed_dict={
                ftr_in: features,
                nmat_in: norm_mat,
                lbl_in: y_test,
                msk_in: test_mask,
                is_train: False,
                ffd_drop: 0.0})
        ts_loss += loss_value_ts
        ts_acc += acc_ts

        # tvars = tf.trainable_variables()
        # tvars_vals = sess.run(tvars)
        # for var, val in zip(tvars, tvars_vals):
        #     print(var.name, val)  # Prints the name of the variable alongside its value.
        
        # print('Test loss:', ts_loss, '; Test accuracy:', ts_acc)
        print(ts_acc)

        sess.close()
