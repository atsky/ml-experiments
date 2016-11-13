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
from util.logs import get_result_directory_path, FileLogger

HIDDEN_VARS_NUMBER = 50
num_epochs = 1000


class DiscriminatorParams():
    def __init__(self):
        initW = lasagne.init.GlorotUniform()
        inttB = lasagne.init.Constant(0.)

        beta = lasagne.init.Constant(0)
        gamma = lasagne.init.Constant(1)

        def shared(x):
            return theano.shared(x.astype(theano.config.floatX), borrow=True)

        self.num_filter1 = 20
        self.W1 = shared(initW.sample(shape=(self.num_filter1, 1, 5, 5)))
        self.beta1 = shared(beta.sample(shape=self.num_filter1))
        self.gamma1 = shared(gamma.sample(shape=self.num_filter1))

        self.num_filter2 = 40
        self.W2 = shared(initW.sample(shape=(self.num_filter2, self.num_filter1, 5, 5)))
        self.beta2 = shared(beta.sample(shape=self.num_filter2))
        self.gamma2 = shared(gamma.sample(shape=self.num_filter2))

        self.h3_size = 800
        self.W3 = shared(initW.sample(shape=(40 * 4 * 4, self.h3_size)))
        self.beta3 = shared(beta.sample(shape=self.h3_size))
        self.gamma3 = shared(gamma.sample(shape=self.h3_size))

        self.W_out = shared(initW.sample(shape=(self.h3_size, 2)))
        self.b_out = shared(inttB.sample(2))

    def get_list(self):
        return [self.W1, self.beta1, self.gamma1,
                self.W2, self.beta2, self.gamma2,
                self.W3, self.beta3, self.gamma3,
                self.W_out, self.b_out]


def build_discriminator(batch_size, input_var, params):
    l_in = lasagne.layers.InputLayer(shape=(batch_size, 28 * 28),
                                     input_var=input_var)

    l_in = lasagne.layers.ReshapeLayer(l_in, shape=(batch_size, 1, 28, 28))

    l_conv1 = lasagne.layers.Conv2DLayer(
        l_in,
        num_filters=params.num_filter1,
        filter_size=(5, 5),
        stride=2,
        W=params.W1,
        b=None,
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    l_conv1 = lasagne.layers.batch_norm(l_conv1,
                                        beta=params.beta1,
                                        gamma=params.gamma1)

    print("l_conv1.output_shape {}".format(l_conv1.output_shape))

    l_conv2 = lasagne.layers.Conv2DLayer(
        l_conv1,
        num_filters=params.num_filter2,
        filter_size=(5, 5),
        stride=2,
        W=params.W2,
        b=None, nonlinearity=lasagne.nonlinearities.leaky_rectify)

    l_conv2 = lasagne.layers.batch_norm(
        l_conv2,
        beta=params.beta2,
        gamma=params.gamma2)

    l_hid3 = lasagne.layers.DenseLayer(
        l_conv2, num_units=params.h3_size,
        W=params.W3, b=None,
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    l_hid3 = lasagne.layers.batch_norm(
        l_hid3,
        beta=params.beta3,
        gamma=params.gamma3)

    l_out = lasagne.layers.DenseLayer(
        l_hid3, num_units=2,
        W=params.W_out,
        b=params.b_out,
        nonlinearity=None)

    return l_out


def build_generator(batch_size, z):
    l_in = lasagne.layers.InputLayer(shape=(None, HIDDEN_VARS_NUMBER), input_var=z)

    l_hid1 = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
        l_in, num_units=1000,
        W=lasagne.init.Normal(0.1),
        nonlinearity=lasagne.nonlinearities.leaky_rectify))

    l_hid2 = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
        l_hid1, num_units=128 * 5 * 5,
        W=lasagne.init.Normal(0.1),
        nonlinearity=lasagne.nonlinearities.leaky_rectify))

    l_reshaped = lasagne.layers.ReshapeLayer(l_hid2, shape=(batch_size, 128, 5, 5))

    l_deconv1 = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(
        l_reshaped, 128, filter_size=(4, 4), stride=2, crop=2,
        nonlinearity=lasagne.nonlinearities.leaky_rectify))

    l_deconv2 = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(
        l_deconv1, 64, filter_size=(4, 4), stride=2, crop=2,
        nonlinearity=lasagne.nonlinearities.leaky_rectify))

    l_deconv3 = lasagne.layers.TransposedConv2DLayer(
        l_deconv2, 1, filter_size=(4, 4), stride=2, crop=1,
        nonlinearity=lasagne.nonlinearities.sigmoid)

    return lasagne.layers.ReshapeLayer(l_deconv3, shape=(batch_size, 28 * 28))


def show_image(data, name):
    image_data = np.zeros(
        (28 * 10, 28 * 10, 3),
        dtype='uint8')

    index = 0

    for x in range(10):
        for y in range(10):
            add_img(image_data, np.clip(data[index, :], 0, 1), x, y)
            index += 1

    image = Image.fromarray(image_data)
    image.save(name)


def main():
    train_x, test_x = load_data()

    train_size = len(train_x)

    train_data = theano.shared(
        train_x.astype(theano.config.floatX),
        borrow=True)

    batch_size = 100

    index = T.iscalar("index")

    data_batch = train_data[index:index + batch_size, :]

    random_streams = theano.tensor.shared_randomstreams.RandomStreams()

    z = random_streams.normal((batch_size, HIDDEN_VARS_NUMBER))

    l_x_generated = build_generator(batch_size, z)
    x_generated = lasagne.layers.get_output(l_x_generated)

    params = DiscriminatorParams()

    l_p_data = build_discriminator(batch_size, data_batch, params)
    l_p_gen = build_discriminator(batch_size, x_generated, params)

    log_p_data = T.nnet.logsoftmax(lasagne.layers.get_output(l_p_data))
    log_p_gen = T.nnet.logsoftmax(lasagne.layers.get_output(l_p_gen))

    loss_data = -log_p_data[:, 1].mean()
    loss_gen = -log_p_gen[:, 0].mean()
    loss1 = loss_data + loss_gen

    params1 = params.get_list()

    updates = lasagne.updates.adam(loss1, params1, learning_rate=0.002)

    train_discrim_fn = theano.function([index], [loss_data, loss_gen], updates=updates)

    x_gen_fn = theano.function([], x_generated)

    loss2 = -log_p_gen[:, 1].mean()

    params2 = lasagne.layers.get_all_params(l_x_generated, trainable=True)

    updates = lasagne.updates.adam(loss2, params2, learning_rate=0.0005)

    train_gen_fn = theano.function([], loss2, updates=updates)

    base_path = get_result_directory_path("dcgan_minst")
    logger = FileLogger(base_path, "main")

    logger.log("Starting training...")
    for epoch in range(num_epochs):
        indexes = list(range(train_size))
        random.shuffle(indexes)

        start_time = time.time()
        for offset in range(0, train_size, batch_size):
            loss_data, loss_gen = train_discrim_fn(offset)
            loss2 = train_gen_fn()
            logger.log("loss_data {:.5f} loss_gen {:.5f}, loss2 {:.5f} "
                       .format(float(loss_data), float(loss_gen), float(loss2)))

        logger.log("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))

        show_image(x_gen_fn(),
                   os.path.join(base_path, "samples_{}.png".format(epoch + 1)))

    logger.close()


if __name__ == '__main__':
    # theano.config.optimizer = "None"
    main()
