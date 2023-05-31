"""
Build a neuron network for handwritten digit recognition using existing labeled data sets.
Data source: www.kaggle.com/c/digit-recognizer
Data explanation: www.wikiwand.com/en/mnist_database

## Data
Source data is a set of 28x28 pixels images of handwritten digits (784 pixel values)
CSV files rows: digits tata (0, 1, 2, 3 ... 9), 42 000 digit examples
CSV file columns: label (digit), pixel 0, pixel 1, ... pixel 783

## Process
3 parts of training:
    - forward propagation
    - backward propagation
    - update parameters
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data_file = "C:/Users/kko8/OneDrive/projects/master/CIS579/assignments/data/train.csv"
data = pd.read_csv(data_file)
# print(data.head())

data = np.array(data)
rows, columns = data.shape
# Shuffle before splitting into dev and training sets
np.random.shuffle(data)

# Split all data into 2 sets, training and testing
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:columns]
X_dev = X_dev / 255.

data_train = data[1000:rows].T
Y_train = data_train[0]  # 784 items
X_train = data_train[1:columns]
X_train = X_train / 255.

_, m_train = X_train.shape

# print(Y_train)
print(X_train[:, 0].shape)
