
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage


def load_dataset():
    train_dataset = h5py.File('./datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T #(209,64,64,3) converted to the shape (12288,209)
    train_set_x = train_set_x_flatten / 255.
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T #(50,64,64,3) converted to the shape (12288,50)
    test_set_x = test_set_x_flatten / 255.

    return train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, classes

train_x,train_y,test_x,test_y,class_list = load_dataset()

