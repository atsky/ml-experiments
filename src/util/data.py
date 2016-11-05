import gzip
import os
import urllib.request
import numpy as np

import pickle


def load_data():
    #############
    # LOAD DATA #
    #############

    data_dir, data_file = "../data", "mnist.pkl.gz"
    dataset = os.path.join(data_dir, data_file)

    # Download the MNIST dataset if it is not present
    if not os.path.isfile(dataset):
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = np.concatenate([train_set_x, valid_set_x], axis=0)
    return train_set_x, test_set_x


def add_img(image_data, img, x, y):
    x_o = 28 * x
    y_o = 28 * y
    image_data[y_o:y_o + 28, x_o:x_o + 28, 0] = 255 - img.reshape((28, 28)) * 255
    image_data[y_o:y_o + 28, x_o:x_o + 28, 1] = 255 - img.reshape((28, 28)) * 255
    image_data[y_o:y_o + 28, x_o:x_o + 28, 2] = 255 - img.reshape((28, 28)) * 255