import gzip
import os
import random
import urllib.request
import numpy as np

import pickle

from PIL import Image


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

def load_cats_data():
    imgs_path = os.path.join("..", "data", "out_aug_64x64")
    file_names = [name for name in os.listdir(imgs_path) if name.endswith(".jpg")]
    images_number = len(file_names)
    random.shuffle(file_names)
    data = np.zeros((images_number, 3, 64, 64))
    print("images_number {}".format(images_number))
    for i, file_name in enumerate(file_names):
        img_path = os.path.join(imgs_path, file_name)
        img = np.asarray(Image.open(img_path)).astype(float) / 255
        data[i] = np.transpose(img, (2, 0, 1))
    print("loading done")
    return data

def add_img(image_data, img, x, y):
    x_o = 28 * x
    y_o = 28 * y
    image_data[y_o:y_o + 28, x_o:x_o + 28, 0] = 255 - img.reshape((28, 28)) * 255
    image_data[y_o:y_o + 28, x_o:x_o + 28, 1] = 255 - img.reshape((28, 28)) * 255
    image_data[y_o:y_o + 28, x_o:x_o + 28, 2] = 255 - img.reshape((28, 28)) * 255