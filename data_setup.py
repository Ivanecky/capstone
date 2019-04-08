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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import MaxAbsScaler

# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------
# Paths to data files
datafile1 = "C:\\Users\\samiv\\Desktop\\senior_project\\test_data_1.gct"
datafile2 = "C:\\Users\\samiv\\Desktop\\senior_project\\gene_data_summary.txt"

# read in dataframe
df = pd.read_csv(datafile1, header=2, sep='\t')

# read in gene summary data
gs = pd.read_csv(datafile2, sep='\t')

# subset gene summmary to first and last columns
gs = gs.iloc[:, [0, 34]]

# ----------------------------------------------------------------------
# DATA TRANSPOSING
# ----------------------------------------------------------------------
# Save column headers
colheads = list(df.columns.values)

# Transpose entire dataset
df_t = df.T

# Remove first two rows with gene names and description
df_t = df_t.iloc[2:]

# Take random sample
rs = df_t.sample(frac=0.5, replace=False, random_state=3)

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
rs = rs[rs.tissue != '']

# Convert data to array format
gmat = rs.iloc[:, 0:(len(rs.columns))]
gmat = gmat.to_numpy()

# Randomize data
np.random.shuffle(gmat)

# Calculate sparsity of matrix
sp = 1 - (np.count_nonzero(gmat) / np.size(gmat))

# Generate split index
sp_index = round(np.size(gmat, 0)*(0.7))

# TEST
gmat = pd.DataFrame(gmat)

y = gmat.iloc[:, 56202]

gmat = gmat.iloc[:, :56201]

X_train, X_test, y_train, y_test = train_test_split(
    gmat, y, test_size=0.3, random_state=42, stratify=y)
                   
# create vectors to hold variables
x_train = np.array(X_train.iloc[:, :56201])
x_test = np.array(X_test.iloc[:, :56201])

y_train = np.array(y_train)
y_test = np.array(y_test)

# One hot encode the tissue names
encoder = LabelBinarizer()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

# Convert array from objects to floats
x_train = x_train.astype(float)
x_test = x_test.astype(float)

# ----------------------------------------------------------------------
# SAVE TO FILES
# ----------------------------------------------------------------------
# Save data to csv files for loading convenience
np.savetxt('train_data.csv', x_train, fmt='%f', delimiter=',')
np.savetxt('test_data.csv', x_test, fmt='%f', delimiter=',')
np.savetxt('train_labels.csv', y_train, fmt='%f', delimiter=',')
np.savetxt('test_labels.csv', y_test, fmt='%f', delimiter=',')
