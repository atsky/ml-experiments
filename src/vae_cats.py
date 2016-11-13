#!/usr/bin/env python
import random

import numpy as np
import theano
import theano.tensor as T
import lasagne
import time
import os

from PIL import Image

from util.data import load_cats_data
from util.logs import get_result_directory_path

HIDDEN_VARS_NUMBER = 200
num_epochs = 100


def build_encoder(batch_size, input_var):
    l_in = lasagne.layers.InputLayer(shape=(batch_size, 3, 64, 64),
                                     input_var=input_var)

    l_conv1 = lasagne.layers.Conv2DLayer(
        l_in,
        num_filters=64,
        filter_size=(5, 5),
        stride=2,
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    l_conv2 = lasagne.layers.Conv2DLayer(
        l_conv1,
        num_filters=128,
        filter_size=(5, 5),
        stride=2,
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    l_conv3 = lasagne.layers.Conv2DLayer(
        l_conv2,
        num_filters=256,
        filter_size=(5, 5),
        stride=2,
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    l_hid1_mean = lasagne.layers.DenseLayer(
        l_conv3, num_units=1024,
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    l_hid2_mean = lasagne.layers.DenseLayer(
        l_hid1_mean, num_units=1024,
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    l_mean = lasagne.layers.DenseLayer(
        l_hid2_mean, num_units=HIDDEN_VARS_NUMBER,
        nonlinearity=None)

    l_hid1_sd = lasagne.layers.DenseLayer(
        l_conv3, num_units=1024,
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    l_hid2_sd = lasagne.layers.DenseLayer(
        l_hid1_sd, num_units=1024,
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    l_sd = lasagne.layers.DenseLayer(
        l_hid2_sd, num_units=HIDDEN_VARS_NUMBER,
        nonlinearity=None)

    return l_mean, l_sd


def build_decoder(batch_size, z):
    l_in = lasagne.layers.InputLayer(shape=(None, HIDDEN_VARS_NUMBER), input_var=z)

    l_hid1 = lasagne.layers.DenseLayer(
        l_in, num_units=2048,
        W=lasagne.init.Normal(0.02),
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    l_hid2 = lasagne.layers.DenseLayer(
        l_hid1, num_units=1024 * 4 * 4,
        W=lasagne.init.Normal(0.02),
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    l_reshaped = lasagne.layers.ReshapeLayer(l_hid2, shape=(batch_size, 1024, 4, 4))

    l_deconv1 = lasagne.layers.TransposedConv2DLayer(
        l_reshaped, 512, filter_size=(6, 6), stride=(2, 2), crop=2,
        W=lasagne.init.Normal(0.02),
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    l_deconv2 = lasagne.layers.TransposedConv2DLayer(
        l_deconv1, 256, filter_size=(6, 6), stride=(2, 2), crop=2,
        W=lasagne.init.Normal(0.02),
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    l_deconv3 = lasagne.layers.TransposedConv2DLayer(
        l_deconv2, 128, filter_size=(6, 6), stride=(2, 2), crop=2,
        W=lasagne.init.Normal(0.02),
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    l_deconv4 = lasagne.layers.TransposedConv2DLayer(
        l_deconv3, 3, filter_size=(6, 6), stride=(2, 2), crop=2,
        W=lasagne.init.Normal(0.02),
        nonlinearity=lasagne.nonlinearities.sigmoid)

    l_hid1_sd = lasagne.layers.DenseLayer(
        l_in, num_units=1024,
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    l_sd2 = lasagne.layers.DenseLayer(
        l_hid1_sd, num_units=1,
        W=lasagne.init.Normal(0.001),
        b=lasagne.init.Constant(2.),
        nonlinearity=None)

    return l_deconv4, l_sd2


def show_image(decode_fn, data, name):
    image_data = np.zeros(
        (64 * 10, 64 * 10, 3),
        dtype='uint8')

    index = 0

    mean, log_sd2 = decode_fn(0)
    for y in range(10):
        for x in range(5):
            x_o = 64 * x * 2
            y_o = 64 * y
            image_data[y_o:y_o + 64, x_o:x_o + 64, :] = np.transpose(data[index], (1, 2, 0)) * 255

            x_o = 64 * (x * 2 + 1)
            image_data[y_o:y_o + 64, x_o:x_o + 64, :] = np.transpose(mean[index], (1, 2, 0)) * 255
            index += 1

    image = Image.fromarray(image_data)
    image.save(name)


def main():
    train_x = load_cats_data()

    train_size = len(train_x)

    train_data = theano.shared(
        train_x.astype(theano.config.floatX),
        borrow=True)

    batch_size = 100

    index = T.iscalar("index")

    data_batch = train_data[index:index + batch_size, :]

    random_streams = theano.tensor.shared_randomstreams.RandomStreams()
    l_batch_mean_z, l_sd2_z = build_encoder(batch_size, data_batch)

    batch_mean_z = lasagne.layers.get_output(l_batch_mean_z)
    batch_log_sd2_z = lasagne.layers.get_output(l_sd2_z)
    batch_log_sd2_z = T.clip(batch_log_sd2_z, -10, 10)

    eps = random_streams.normal((batch_size, HIDDEN_VARS_NUMBER))

    z = batch_mean_z + eps * np.exp(batch_log_sd2_z / 2)
    l_data_mean, l_data_log_sd2 = build_decoder(batch_size, z)

    data_mean = lasagne.layers.get_output(l_data_mean)
    data_log_sd2 = lasagne.layers.get_output(l_data_log_sd2)
    data_log_sd2 = T.clip(data_log_sd2, -10, 10)

    kl = (batch_log_sd2_z - batch_mean_z ** 2 - T.exp(batch_log_sd2_z)) / 2.0

    diff_sq = T.sum(-(data_mean - data_batch) ** 2, axis=[1, 2, 3])
    pixel_loss = (diff_sq * T.exp(-data_log_sd2) - (64 * 64 * 3) * data_log_sd2) / 2.0
    lower_bound = T.sum(kl, axis=1) + pixel_loss
    loss = -lower_bound.mean()

    params = lasagne.layers.get_all_params([l_data_mean, l_data_log_sd2, l_batch_mean_z, l_sd2_z], trainable=True)

    grads = theano.grad(loss, params)

    GRAD_CLIP = 100

    scaled_grads = lasagne.updates.total_norm_constraint(grads, GRAD_CLIP)

    updates = lasagne.updates.adam(scaled_grads, params, learning_rate=0.0001)

    train_fn = theano.function([index], loss, updates=updates)

    decode_fn = theano.function([index], [data_mean, data_log_sd2])

    base_path = get_result_directory_path("vae_cats")
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    print("Starting training...")
    for epoch in range(num_epochs):
        indexes = list(range(train_size))
        random.shuffle(indexes)

        train_err = 0
        train_batches = 0
        start_time = time.time()
        for i in range(0, train_size - batch_size + 1, batch_size):
            loss = train_fn(i)
            print(" {:3}e, loss:{}".format(epoch + 1, loss))
            train_err += loss
            train_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

        show_image(decode_fn,
                   train_x,
                   os.path.join(base_path, 'samples_{}.png'.format(epoch + 1)))


if __name__ == '__main__':
    # theano.config.optimizer = "None"
    main()
