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
import sklearn as sk
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import MaxAbsScaler

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
rs = df_t.sample(frac=0.5, replace=False, random_state=1)

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

# Calculate sparsity of matrix
sp = 1 - (np.count_nonzero(gmat) / np.size(gmat))

# Split into train & test data at 80:20 ratio
train, test = gmat[:(np.size(gmat, 0)*0.8),
                   :], gmat[(np.size(gmat, 0)*0.8):, :]

# One hot encode the tissue names
encoder = LabelBinarizer()
train_labels = encoder.fit_transform(train[:, -1])
test_labels = encoder.fit_transform(test[:, -1])

# Convert array from objects to floats
train_data = train[:, :-1].astype(float)
test_data = test[:, :-1].astype(float)

# ----------------------------------------------------------------------
# SAVE TO FILES
# ----------------------------------------------------------------------
# Save data to csv files for loading convenience
np.savetxt('train_data.csv', train_data, fmt='%f', delimiter=',')
np.savetxt('test_data.csv', test_data, fmt='%f', delimiter=',')
np.savetxt('train_labels.csv', train_labels, fmt='%f', delimiter=',')
np.savetxt('test_labels.csv', test_labels, fmt='%f', delimiter=',')
