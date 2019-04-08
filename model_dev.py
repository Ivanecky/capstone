from __future__ import absolute_import, division, print_function

from time import time

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import Activation, Dense, Dropout, Flatten, LeakyReLU
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import plot_model
from numpy import genfromtxt
from tensorflow import keras

import matplotlib
import matplotlib.pyplot as plt
import pylab
import seaborn as sn
from sklearn import metrics, preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, MaxAbsScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import plotly.plotly as py
import plotly.graph_objs as go

# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------
# Read in numpy arrays from csv file
test_data = genfromtxt(
    'C:\\Users\\samiv\\Desktop\\senior_project\\test_data.csv', delimiter=',', dtype='float')
train_data = genfromtxt(
    'C:\\Users\\samiv\\Desktop\\senior_project\\train_data.csv', delimiter=',', dtype='float')
test_labels = genfromtxt(
    'C:\\Users\\samiv\\Desktop\\senior_project\\test_labels.csv', delimiter=',')
train_labels = genfromtxt(
    'C:\\Users\\samiv\\Desktop\\senior_project\\train_labels.csv', delimiter=',')

# ----------------------------------------------------------------------
# PREPROCESSING DATA
# ----------------------------------------------------------------------
# Define scaler to be used
scaler = MaxAbsScaler()

# Scale training data
train_data = scaler.fit_transform(train_data)

# Scale testing data
test_data = scaler.fit_transform(test_data)

# Normalize the data - use l2 for sparsity
train_data = preprocessing.normalize(train_data, norm='l2')
test_data = preprocessing.normalize(test_data, norm='l2')

# ----------------------------------------------------------------------
# PRINCIPAL COMPONENT ANALYSIS
# ----------------------------------------------------------------------
# Model instance of only PCA with 70% variance explained
pca = PCA(.70)

# Fit PCA on training data
pca.fit(train_data)
pca.fit(test_data)

# Apply mapping to both data sets - condenses dimensions
# from 56202 to 1657
train_data_pca = pca.transform(train_data)
test_data_pca = pca.transform(test_data)

# ----------------------------------------------------------------------
# PCA MODEL BUILD & TRAINING
# ----------------------------------------------------------------------
# Define model
model = Sequential()
# Setup early stopping
earlyStop = keras.callbacks.EarlyStopping(
    monitor='val_acc', patience=3, verbose=1)
# Define input shape
input_shape = train_data_pca[0].shape
# Add layers to model
model.add(Dense(100, activation='tanh', input_shape=input_shape))
model.add(Dense(53, activation='softmax'))
# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=[keras.metrics.categorical_accuracy, 'accuracy'])

# Configure tensorboard
tbCallBack = keras.callbacks.TensorBoard(
    log_dir='Graph', histogram_freq=0, write_graph=True, write_images=True)

# Fit model with tensorboard callback
model.fit(train_data_pca, train_labels, epochs=50,
          batch_size=32, verbose=1, validation_split=0.3,
          callbacks=[earlyStop, tbCallBack])

# Model evaluation
score = model.evaluate(test_data_pca, test_labels, batch_size=32, verbose=1)

# Get loss and accuracy from test evaluation
test_loss = score[0]
test_acc = score[1]

# ----------------------------------------------------------------------
# FULL MODEL BUILD & TRAINING
# ----------------------------------------------------------------------
# Define model
model = Sequential()
# Define input shape
input_shape = train_data[0].shape
# Add layers to model
model.add(Dense(100, activation='sigmoid', input_shape=input_shape))
model.add(Dense(53, activation='softmax'))
# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=[keras.metrics.categorical_accuracy, 'accuracy'])

# Configure tensorboard
tbCallBack = keras.callbacks.TensorBoard(
    log_dir='Graph', histogram_freq=0, write_graph=True, write_images=True)

# Fit model with tensorboard callback
model.fit(train_data, train_labels, epochs=50,
          batch_size=32, verbose=1, validation_split=0.3,
          callbacks=[earlyStop, tbCallBack])


# Model evaluation
score = model.evaluate(test_data, test_labels, batch_size=64, verbose=1)

# Get loss and accuracy from test evaluation
test_loss = score[0]
test_acc = score[1]

# Make predictions
predictions = model.predict_classes(test_data, batch_size=32)

# Generate confusion matrix
matrix = metrics.confusion_matrix(test_labels.argmax(axis=1), predictions)

# Create heatmap using seaborn
plt.clf()
plt.figure()
hmap1 = sn.heatmap(matrix, annot=True, square=True,
                   xticklabels=True, yticklabels=True)
hmap1.figure.set_size_inches(24.5, 18.5)
hmap1.figure.savefig('hmap_sigmoid.png')

# Second heatmap using matplotlib
plt.clf()
plt.figure()
hmap2 = plt.imshow(matrix, cmap='hot')
hmap2.figure.set_size_inches(18.5, 10.5)
hmap2.figure.savefig('hmap2.png')

# Third heatmap with no annotations
plt.clf()
plt.figure()
hmap3 = sn.heatmap(matrix)
hmap3.figure.set_size_inches(24.5, 18.5)
hmap3.figure.savefig('hmap3.png')
