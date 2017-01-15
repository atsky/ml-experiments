#!/usr/bin/env python
import random

import numpy as np
import theano
import theano.tensor as T
import lasagne
import time
import os

from PIL import Image

from util.data import load_data, add_img
from util.logs import FileLogger
from util.logs import get_result_directory_path

HIDDEN_VARS_NUMBER = 50
num_epochs = 100


def build_encoder(batch_size, input_var):
    l_in = lasagne.layers.InputLayer(shape=(batch_size, 28 * 28),
                                     input_var=input_var)

    l_hid1 = lasagne.layers.DenseLayer(
        l_in, num_units=800,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_hid2 = lasagne.layers.DenseLayer(
        l_hid1, num_units=1000,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_hid3_mean = lasagne.layers.DenseLayer(
        l_hid2, num_units=1600,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_mean = lasagne.layers.DenseLayer(
        l_hid3_mean, num_units=HIDDEN_VARS_NUMBER,
        nonlinearity=None)

    l_hid2_sd = lasagne.layers.DenseLayer(
        l_hid1, num_units=1000,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_hid3_sd = lasagne.layers.DenseLayer(
        l_hid2_sd, num_units=1600,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_sd2 = lasagne.layers.DenseLayer(
        l_hid3_sd, num_units=HIDDEN_VARS_NUMBER,
        nonlinearity=None)

    return l_mean, l_sd2


def build_decoder(z):
    l_in = lasagne.layers.InputLayer(shape=(None, HIDDEN_VARS_NUMBER), input_var=z)

    l_hid1 = lasagne.layers.DenseLayer(
        l_in, num_units=1000,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_hid2 = lasagne.layers.DenseLayer(
        l_hid1, num_units=1500,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_hid3_mean = lasagne.layers.DenseLayer(
        l_hid2, num_units=2500,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_mean = lasagne.layers.DenseLayer(
        l_hid3_mean, num_units=4 * 14 * 14,
        nonlinearity=None)

    l_hid2_sd = lasagne.layers.DenseLayer(
        l_hid1, num_units=1000,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_hid3_sd = lasagne.layers.DenseLayer(
        l_hid2_sd, num_units=500,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_sd2 = lasagne.layers.DenseLayer(
        l_hid3_sd, num_units=4 * 14 * 14,
        W=lasagne.init.Normal(0.001),
        b=lasagne.init.Constant(2.),
        nonlinearity=None)

    return l_mean, l_sd2


def show_image(decode_fn, decode2, data, name):
    image_data = np.zeros(
        (28 * 19, 28 * 20, 3),
        dtype='uint8'
    )

    for x in range(20):
        index = x
        add_img(image_data, data[index, :], x, 0)

    for y in range(1, 10):
        mean, log_sd2 = decode_fn(0)
        if y == 1:
            sd = np.exp(log_sd2 / 2)
            print("sd - min {}, max {}".format(np.min(sd), np.max(sd)))

        for x in range(20):
            index = x
            sd = np.exp(log_sd2[index, :] / 2)

            data = mean[index, :] + np.random.normal(size=(4 * 14 * 14)) * sd
            add_img(image_data, np.clip(decode2(mean[index, :]), 0, 1), x, y * 2 - 1)
            add_img(image_data, np.clip(decode2(data.astype(theano.config.floatX), ), 0, 1), x, y * 2)

    image = Image.fromarray(image_data)
    image.save(name)


def main():
    batch_size = 1000

    normal_matrix_data = np.asarray(
        [[1., 0., 0., 0.],
         [0., 1., 0., 0.],
         [0., 0., 1., 0.],
         [0., 0., 0., 1.]])

    identity = theano.shared(
        np.identity(4).astype(theano.config.floatX),
        borrow=True)

    vector1 = theano.shared(
        np.random.normal(size=4).astype(theano.config.floatX),
        borrow=True)

    vector2 = theano.shared(
        np.random.normal(size=4).astype(theano.config.floatX),
        borrow=True)

    #vector3 = theano.shared(
    #    np.random.normal(size=4).astype(theano.config.floatX),
    #    borrow=True)

    #vector4 = theano.shared(
    #    np.random.normal(size=4).astype(theano.config.floatX),
    #    borrow=True)

    #def householder(v):
    #    return identity - 2 * T.outer(v, v) / T.sum(v ** 2)

    #normal_matrix = householder(vector1).dot(householder(vector2)).dot(householder(vector3)).dot(householder(vector4))

    normal_matrix = theano.shared(
        normal_matrix_data.astype(theano.config.floatX),
        borrow=True)

    normal_matrix = normal_matrix.reshape((4, 1, 2, 2))

    def get_decode_fn():
        input = T.vector("in")

        l_in = lasagne.layers.InputLayer(shape=(4 * 14 * 14,), input_var=input)
        l_in = lasagne.layers.ReshapeLayer(l_in, shape=(1, 4, 14, 14))

        out_conv = lasagne.layers.TransposedConv2DLayer(
            l_in,
            num_filters=1,
            filter_size=(2, 2),
            stride=2,
            W=normal_matrix,
            b=None,
            nonlinearity=None)

        out_conv = lasagne.layers.ReshapeLayer(out_conv, shape=(28 * 28,))

        out = lasagne.layers.get_output(out_conv)

        return theano.function([input], out)

    train_x, test_x = load_data()

    train_size = len(train_x)

    train_data = theano.shared(
        train_x.astype(theano.config.floatX),
        borrow=True)

    index = T.iscalar("index")

    data_batch = train_data[index:index + batch_size, :]

    l_in = lasagne.layers.ReshapeLayer(lasagne.layers.InputLayer(shape=(batch_size, 28 * 28), input_var=data_batch),
                                       shape=(batch_size, 1, 28, 28))

    l_conv1 = lasagne.layers.Conv2DLayer(
        l_in,
        num_filters=4,
        filter_size=(2, 2),
        stride=2,
        W=normal_matrix,
        b=None,
        nonlinearity=None)

    print(l_conv1.output_shape)

    l_tranformed = lasagne.layers.reshape(l_conv1, shape=(batch_size, 4 * 14 * 14))

    tranformed_batch = lasagne.layers.get_output(l_tranformed)

    random_streams = theano.tensor.shared_randomstreams.RandomStreams()
    l_batch_mean_z, l_sd2_z = build_encoder(batch_size, data_batch)

    batch_mean_z = lasagne.layers.get_output(l_batch_mean_z)
    batch_log_sd2_z = lasagne.layers.get_output(l_sd2_z)
    batch_log_sd2_z = T.clip(batch_log_sd2_z, -10, 10)

    eps = random_streams.normal((batch_size, HIDDEN_VARS_NUMBER))

    z = batch_mean_z + eps * np.exp(batch_log_sd2_z / 2)
    l_data_mean, l_data_log_sd2 = build_decoder(z)

    data_mean = lasagne.layers.get_output(l_data_mean)
    data_log_sd2 = lasagne.layers.get_output(l_data_log_sd2)
    data_log_sd2 = T.clip(data_log_sd2, -10, 10)

    kl = (batch_log_sd2_z - batch_mean_z ** 2 - T.exp(batch_log_sd2_z)) / 2.0

    pixel_loss = (-(data_mean - tranformed_batch) ** 2 * T.exp(-data_log_sd2) - data_log_sd2) / 2.0
    lower_bound = T.sum(kl, axis=1) + T.sum(pixel_loss, axis=1)
    loss = -lower_bound.mean()

    params = lasagne.layers.get_all_params([l_data_mean, l_data_log_sd2, l_batch_mean_z, l_sd2_z], trainable=True)

    #params += [vector1, vector2, vector3, vector4]

    grads = theano.grad(loss, params)

    GRAD_CLIP = 100

    scaled_grads = lasagne.updates.total_norm_constraint(grads, GRAD_CLIP)

    updates = lasagne.updates.adam(scaled_grads, params, learning_rate=0.001)

    train_fn = theano.function([index], [loss, vector1], updates=updates)

    decode_fn = theano.function([index], [data_mean, data_log_sd2])

    base_path = get_result_directory_path("conv_autoencoder")

    print("Starting training...")
    for epoch in range(num_epochs):
        indexes = list(range(train_size))
        random.shuffle(indexes)

        train_err = 0
        train_batches = 0
        start_time = time.time()
        for i in range(0, train_size, batch_size):
            loss, vector1 = train_fn(i)
            print("loss:{}".format(loss))
            print(vector1)
            train_err += loss
            train_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

        show_image(decode_fn,
                   get_decode_fn(),
                   train_x,
                   os.path.join(base_path, 'samples_{}.png'.format(epoch + 1)))


if __name__ == '__main__':
    theano.config.optimizer = "None"
    main()
