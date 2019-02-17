from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
# DATA SPLITTING
# ----------------------------------------------------------------------

# generate random sample of large data
rs = df.sample(frac=0.25, replace=False, random_state=1)

# transpose random sample
rs_t = rs.T

# convert first row to column names
rs_t.columns = rs_t.iloc[0]

# remove rows 1 and 2
rs_t = rs_t.iloc[2:]

# create column to hold tissue type
rs_t["tissue"] = ""

# create vectors to hold variables
tissues = np.array(gs.iloc[:, 1])
source_names = np.array(gs.iloc[:, 0])
tissue_ids = np.array(list(rs_t.index))
names = np.array(rs_t.iloc[:, len(rs_t.columns)-1])

# get matching tissue names
for i in range(len(tissue_ids)):
    for j in range(len(source_names)):
        if tissue_ids[i] == source_names[j]:
            names[i] = tissues[j]
            break

# assign to tissue variable in rs_t
rs_t.iloc[:, len(rs_t.columns) - 1] = names

# Convert data to array format
gmat = rs_t.iloc[:, 0:(len(rs_t.columns)-1)]
gmat = gmat.to_numpy()

# Convert array from objects to floats
gmat.astype(float)

# One hot encode the tissue names
encoder = LabelBinarizer()
gmat_labels = encoder.fit_transform(rs_t.iloc[:, len(rs_t.columns)-1])

# Attempt to build model using keras
# Define model
model = Sequential()
# Define input shape
input_shape = gmat[0].shape
# Add layers to model
model.add(Dense(128, activation='sigmoid', input_shape=input_shape))
model.add(Dense(84, activation='sigmoid'))
model.add(Dense(54, activation='softmax'))
# Optimizer function
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Compile model
model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])

# Try to make a model
model.fit(gmat, gmat_labels, epochs=10, batch_size=200)
