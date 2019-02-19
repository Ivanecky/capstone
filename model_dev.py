from __future__ import absolute_import, division, print_function
from time import time

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import LeakyReLU
from keras.utils import plot_model
from keras.callbacks import TensorBoard

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import MaxAbsScaler

# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------
# Read in numpy arrays from csv file
test_data = genfromtxt('test_data.csv', delimiter=',')
train_data = genfromtxt('train_data.csv', delimiter=',')
test_labels = genfromtxt('test_labels.csv', delimiter=',')
train_labels = genfromtxt('train_labels.csv', delimiter=',')

# ----------------------------------------------------------------------
# PREPROCESSING DATA
# ----------------------------------------------------------------------
# Define scaler to be used
scaler = MaxAbsScaler()

# Scale training data
train_data = scaler.fit_transform(train_data)

# Scale testing data
test_data = scaler.fit_transform(test_data)

# Normalize the data - use l1 for sparsity
train_data = preprocessing.normalize(train_data, norm='l2')
test_data = preprocessing.normalize(test_data, norm='l2')

# ----------------------------------------------------------------------
# KERAS MODEL BUILD & TRAINING
# ----------------------------------------------------------------------
# Define model
model = Sequential()
# Define input shape
input_shape = train_data[0].shape
# Add layers to model
model.add(Dense(100, activation='tanh', input_shape=input_shape))
model.add(Dense(53, activation='softmax'))
# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=[keras.metrics.categorical_accuracy])

# Configure tensorboard
tbCallBack = keras.callbacks.TensorBoard(
    log_dir='Graph', histogram_freq=0, write_graph=True, write_images=True)

# Fit model with tensorboard callback
model.fit(train_data, train_labels, epochs=25,
          batch_size=32, callbacks=[tbCallBack], verbose=1)

# Evaluate the model
score = model.evaluate(test_data, test_labels, batch_size=32)
