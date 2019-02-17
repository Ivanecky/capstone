from __future__ import absolute_import, division, print_function

import pandas as pd
import tensorflow as tf
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import plot_model
from tensorflow import keras

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer

# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------
# parse gct file into dataframe
datafile = "test_data_1.gct"

# read in dataframe
df = pd.read_csv(datafile, header=2, sep='\t')

# read in gene summary data
gs = pd.read_csv("gene_data_summary.txt", sep='\t')

# subset gene summmary to first and last columns
gs = gs.iloc[:, [0, 34]]

# ----------------------------------------------------------------------
# DATA SPLITTING - TRAIN & TEST
# ----------------------------------------------------------------------
# Transpose entire dataset
df_t = df.T

# Shuffle data
np.random.shuffle(df_t)

# Split into training and testing
train = df_t.iloc[2:2002, :].copy()
test = df_t.iloc[2500:3000, :].copy()

# ----------------------------------------------------------------------
# DATA FORMATTING
# ----------------------------------------------------------------------

# create column to hold tissue type
train["tissue"] = ""
test["tissue"] = ""

# create vectors to hold variables
tissues = np.array(gs.iloc[:, 1])
source_names = np.array(gs.iloc[:, 0])

tr_tissue_ids = np.array(list(train.index))
tr_names = np.array(train.iloc[:, len(train.columns) - 1])

ts_tissue_ids = np.array(list(test.index))
ts_names = np.array(test.iloc[:, len(test.columns) - 1])

# get matching tissue names
# Train
for i in range(len(tr_tissue_ids)):
    for j in range(len(source_names)):
        if tr_tissue_ids[i] == source_names[j]:
            tr_names[i] = tissues[j]
            break
# Test
for i in range(len(ts_tissue_ids)):
    for j in range(len(source_names)):
        if ts_tissue_ids[i] == source_names[j]:
            ts_names[i] = tissues[j]
            break

# Assign tissue names
train.iloc[:, len(train.columns) - 1] = tr_names
test.iloc[:, len(test.columns) - 1] = ts_names

# Convert data to array format
tr_gmat = train.iloc[:, 0:(len(train.columns)-1)]
tr_gmat = tr_gmat.to_numpy()

ts_gmat = test.iloc[:, 0:(len(test.columns)-1)]
ts_gmat = ts_gmat.to_numpy()

# Convert array from objects to floats
tr_gmat.astype(float)
ts_gmat.astype(float)

# Normalize the data
tr_gmat = preprocessing.normalize(tr_gmat)
ts_gmat = preprocessing.normalize(ts_gmat)

# One hot encode the tissue names
encoder = LabelBinarizer()
tr_gmat_labels = encoder.fit_transform(train.iloc[:, len(train.columns) - 1])
ts_gmat_labels = encoder.fit_transform(test.iloc[:, len(test.columns)-1])

# ----------------------------------------------------------------------
# MODEL BUILD & TRAINING
# ----------------------------------------------------------------------
# Define model
model = Sequential()
# Define input shape
input_shape = tr_gmat[0].shape
# Add layers to model
model.add(Dense(100, activation='relu', input_shape=input_shape))
model.add(Dense(75, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(53, activation='softmax'))
# Optimizer function
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Compile model
model.compile(optimizer=sgd, loss='categorical_crossentropy',
              metrics=['accuracy'])

# Try to make a model
model.fit(tr_gmat, tr_gmat_labels, epochs=10, batch_size=75)
