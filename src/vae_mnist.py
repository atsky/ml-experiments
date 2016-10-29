#!/usr/bin/env python
import random

import numpy as np
import theano
import theano.tensor as T
import lasagne
import time
import os

from PIL import Image

from util import load_data, add_img

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
        l_hid3_mean, num_units=28 * 28,
        nonlinearity=None)

    l_hid2_sd = lasagne.layers.DenseLayer(
        l_hid1, num_units=1000,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_hid3_sd = lasagne.layers.DenseLayer(
        l_hid2_sd, num_units=500,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_sd2 = lasagne.layers.DenseLayer(
        l_hid3_sd, num_units=1,
        W=lasagne.init.Normal(0.001),
        b=lasagne.init.Constant(2.),
        nonlinearity=None)

    return l_mean, l_sd2


def show_image(decode_fn, data, name):
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

            data = mean[index, :] + np.random.normal(size=(28 * 28)) * sd
            add_img(image_data, np.clip(mean[index, :], 0, 1), x, y * 2 - 1)
            add_img(image_data, np.clip(data, 0, 1), x, y * 2)

    image = Image.fromarray(image_data)
    image.save(name)


def main():
    train_x, test_x = load_data()

    train_size = len(train_x)

    train_data = theano.shared(
        train_x.astype(theano.config.floatX),
        borrow=True)

    batch_size = 1000

    index = T.iscalar("index")

    data_batch = train_data[index:index + batch_size, :]

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

    pixel_loss = (-(data_mean - data_batch) ** 2 * T.exp(-data_log_sd2) - data_log_sd2) / 2.0
    lower_bound = T.sum(kl, axis=1) + T.sum(pixel_loss, axis=1)
    loss = -lower_bound.mean()

    params = lasagne.layers.get_all_params([l_data_mean, l_data_log_sd2, l_batch_mean_z, l_sd2_z], trainable=True)

    grads = theano.grad(loss, params)

    GRAD_CLIP = 100

    scaled_grads = lasagne.updates.total_norm_constraint(grads, GRAD_CLIP)

    grad_sum = None
    for g in scaled_grads:
        if grad_sum:
            grad_sum = grad_sum + (g ** 2).mean()
        else:
            grad_sum = (g ** 2).mean()

    updates = lasagne.updates.adam(scaled_grads, params, learning_rate=0.001)

    train_fn = theano.function([index], [loss, grad_sum], updates=updates)

    decode_fn = theano.function([index], [data_mean, data_log_sd2])

    print("Starting training...")
    for epoch in range(num_epochs):
        indexes = list(range(train_size))
        random.shuffle(indexes)

        train_err = 0
        train_batches = 0
        start_time = time.time()
        grads = []
        for i in range(0, train_size, batch_size):
            loss, grad = train_fn(i)
            print("loss:{}".format(loss))
            train_err += loss
            grads.append(grad)
            train_batches += 1

        # print("grads:{}".format(grads))

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

        base_img_path = "../img/vae_mnist/"
        if not os.path.exists(base_img_path):
            os.makedirs(base_img_path)

        show_image(decode_fn,
                   train_x,
                   os.path.join(base_img_path, 'samples_{}.png'.format(epoch + 1)))


if __name__ == '__main__':
    # theano.config.optimizer = "None"
    main()
