# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 17:40:11 2019

@author: Shenghai
"""

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape

print("train_labels' lengtg is {}".format(len(train_labels)))

train_labels

print("test_images has {} images, {} x {} pixels".format(test_images.shape[0], \
                                                         test_images.shape[1], \
                                                         test_images.shape[2]))

print("test_labels' length is {}".format(len(test_labels)))

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure()
plt.imshow(train_images[3427])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0
test_images  = test_images  / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])