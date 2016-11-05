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

        def shared(x):
            return theano.shared(x.astype(theano.config.floatX), borrow=True)

        self.h1_size = 600
        self.W1 = shared(initW.sample(shape=(28 * 28, self.h1_size)))
        self.b1 = shared(inttB.sample(self.h1_size))

        self.h2_size = 400
        self.W2 = shared(initW.sample(shape=(self.h1_size, self.h2_size)))
        self.b2 = shared(inttB.sample(self.h2_size))

        self.W_out = shared(initW.sample(shape=(self.h2_size, 1)))
        self.b_out = shared(inttB.sample(1))

    def get_list(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W_out, self.b_out]


def build_discriminator(batch_size, input_var, params):
    l_in = lasagne.layers.InputLayer(shape=(batch_size, 28 * 28),
                                     input_var=input_var)

    l_in = lasagne.layers.DropoutLayer(l_in, p=0.1)

    l_hid1 = lasagne.layers.DenseLayer(
        l_in, num_units=params.h1_size,
        W=params.W1,
        b=params.b1,
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    l_hid1 = lasagne.layers.DropoutLayer(l_hid1, p=0.4)

    l_hid2 = lasagne.layers.DenseLayer(
        l_hid1, num_units=params.h2_size,
        W=params.W2,
        b=params.b2,
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    l_hid2 = lasagne.layers.DropoutLayer(l_hid2, p=0.4)

    l_out = lasagne.layers.DenseLayer(
        l_hid2, num_units=1,
        W=params.W_out,
        b=params.b_out,
        nonlinearity=lasagne.nonlinearities.sigmoid)

    return l_out


def build_generator(z):
    l_in = lasagne.layers.InputLayer(shape=(None, HIDDEN_VARS_NUMBER), input_var=z)

    l_hid1 = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
        l_in, num_units=1000,
        W=lasagne.init.Normal(0.1),
        nonlinearity=lasagne.nonlinearities.leaky_rectify))

    l_hid2 = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
        l_hid1, num_units=1500,
        W=lasagne.init.Normal(0.1),
        nonlinearity=lasagne.nonlinearities.leaky_rectify))

    l_hid3 = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
        l_hid2, num_units=2500,
        W=lasagne.init.Normal(0.1),
        nonlinearity=lasagne.nonlinearities.leaky_rectify))

    l_mean = lasagne.layers.DenseLayer(
        l_hid3, num_units=28 * 28,
        W=lasagne.init.Normal(0.1),
        nonlinearity=lasagne.nonlinearities.sigmoid)

    return l_mean


def show_image(data, name):
    image_data = np.zeros(
        (28 * 20, 28 * 20, 3),
        dtype='uint8')

    index = 0

    for x in range(20):
        for y in range(20):
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

    batch_size = 1000

    index = T.iscalar("index")

    data_batch = train_data[index:index + batch_size, :]

    random_streams = theano.tensor.shared_randomstreams.RandomStreams()

    z = random_streams.normal((batch_size, HIDDEN_VARS_NUMBER))

    l_x_generated = build_generator(z)
    x_generated = lasagne.layers.get_output(l_x_generated)

    params = DiscriminatorParams()

    l_p_data = build_discriminator(batch_size, data_batch, params)
    l_p_gen = build_discriminator(batch_size, x_generated, params)

    p_data = lasagne.layers.get_output(l_p_data)
    p_gen = lasagne.layers.get_output(l_p_gen)

    loss_data = -T.log(p_data).mean()
    loss_gen = - T.log1p(-p_gen).mean()
    loss1 = loss_data + loss_gen

    params1 = params.get_list()

    updates = lasagne.updates.nesterov_momentum(loss1, params1, learning_rate=0.002, momentum=0.9)

    train_discrim_fn = theano.function([index], [loss_data, loss_gen], updates=updates)

    x_gen_fn = theano.function([], x_generated)

    loss2 = -T.log(p_gen).mean()

    params2 = lasagne.layers.get_all_params(l_x_generated, trainable=True)

    updates = lasagne.updates.nesterov_momentum(loss2, params2, learning_rate=0.001, momentum=0.9)

    train_gen_fn = theano.function([], loss2, updates=updates)

    base_path = get_result_directory_path("gan_minst")
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

        logger.log("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))

        show_image(x_gen_fn(),
                   os.path.join(base_path, "samples_{}_final.png".format(epoch + 1)))

    logger.close()


if __name__ == '__main__':
    # theano.config.optimizer = "None"
    main()
