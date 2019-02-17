from __future__ import absolute_import, division, print_function

import pandas as pd
import tensorflow as tf
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import LeakyReLU
from keras.utils import plot_model
from tensorflow import keras

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

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

# Remove first two rows with gene names and description
df_t = df_t.iloc[2:]

# Take random sample
rs = df_t.sample(frac=0.20, replace=False, random_state=1)

# ----------------------------------------------------------------------
# DATA FORMATTING
# ----------------------------------------------------------------------
# create column to hold tissue type
rs["tissue"] = ""

# create vectors to hold variables
tissues = np.array(gs.iloc[:, 1])
source_names = np.array(gs.iloc[:, 0])

tissue_ids = np.array(list(rs.index))
names = np.array(rs.iloc[:, len(rs.columns) - 1])

# get matching tissue names
# Train
for i in range(len(tissue_ids)):
    for j in range(len(source_names)):
        if tissue_ids[i] == source_names[j]:
            names[i] = tissues[j]
            break

# Assign tissue names
rs.iloc[:, len(rs.columns) - 1] = names

# Convert data to array format
gmat = rs.iloc[:, 0:(len(rs.columns))]
gmat = gmat.to_numpy()

# Randomize data
np.random.shuffle(gmat)

# Split into train & test data
train, test = gmat[:2000, :], gmat[2000:, :]

# One hot encode the tissue names
encoder = LabelBinarizer()
train_labels = encoder.fit_transform(train[:, -1])
test_labels = encoder.fit_transform(test[:, -1])

# Convert array from objects to floats
train_data = train[:, :-1].astype(float)
test_data = test[:, :-1].astype(float)

# ----------------------------------------------------------------------
# CHECK FOR MIN COLUMNS
# ----------------------------------------------------------------------
# Get 90th pct value for each column
pcts = np.percentile(train_data, 0.9, axis=0)
for i in range(np.size(train_data, 1)):
    if np.percentile(train_data[:, i], 0.9) == 0:
        train_data = np.delete(train_data, i, axis=1)

# ----------------------------------------------------------------------
# DECISION TREE
# ----------------------------------------------------------------------
# Create decision tree
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(train_data, train[:, -1])

# Predict the response for test dataset
y_pred = clf.predict(test_data)

# Test
print("Accuracy:", metrics.accuracy_score(test[:, -1], y_pred))

# ----------------------------------------------------------------------
# NORMALIZE DATA
# ----------------------------------------------------------------------
# Normalize the data
train_data = preprocessing.normalize(train_data)
test_data = preprocessing.normalize(test_data)

# ----------------------------------------------------------------------
# MODEL BUILD & TRAINING
# ----------------------------------------------------------------------
# Define model
model = Sequential()
# Define input shape
input_shape = train_data[0].shape
# Add layers to model
model.add(Dense(1000, activation='tanh', input_shape=input_shape))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(1000, activation='relu'))
#model.add(Dense(200, activation='sigmoid'))
model.add(Dense(53, activation='softmax'))
# Optimizer function
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Compile model
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Try to make a model
model.fit(train_data, train_labels, epochs=10, batch_size=32)

# Evaluate the model
score = model.evaluate(test_data, test_labels, batch_size=32)
