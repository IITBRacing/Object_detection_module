
import numpy as np
import h5py
import matplotlib.pyplot as plt

train_dataset = h5py.File('./datasets/train_catvnoncat.h5', "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

test_dataset = h5py.File('./datasets/test_catvnoncat.h5', "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

train_classes = np.array(train_dataset["list_classes"][:])  # the list of classes same as in the test dataset classes
classes = np.array(test_dataset["list_classes"][:])  # the list of classes

print("There are two class",classes)
print("b -- represents binary")
print("length of 'non-cat': " ,len(classes[0]))
print("these numbers are the encoding for the characters 'non-cat'")
for i in range(len(classes[0])):
    print(classes[0][i])
print("length of 'cat': " ,len(classes[1]))
print("these numbers are the encoding for the characters 'cat' ")
for i in range(len(classes[1])):
    print(classes[1][i])

print("c - 99 , a - 97 , t -116 ")

print()
print(train_set_x_orig.shape,"This has 209 images each of size 64x 64 and 3 reprensents RGB")
print()
print(train_set_y_orig.shape,"This has information of each image as cat or non cat --> 0 means non cat image and 1 means cat image")
print()
print(test_set_x_orig.shape,"This has 50 images each of size 64x 64 and 3 reprensents RGB")
print()
print(test_set_y_orig.shape,"This has information of each image as cat or non cat  --> 0 means non cat image and 1 means cat image")


# Example of a picture
index = 10
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y_orig[index]) + ", it's a '" + classes[np.squeeze(train_set_y_orig[index])].decode("utf-8") +  "' picture.")