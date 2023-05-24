"""
Build a neuron network for handwritten digit recognition using existing labeled data sets.
Data source: www.kaggle.com/c/digit-recognizer
Data explanation: www.wikiwand.com/en/mnist_database

## Data
Source data is a set of 28x28 pixels images of handwritten digits (784 pixel values)
CSV files rows: digits tata (0, 1, 2, 3 ... 9), 42 000 digit examples
CSV file columns: label (digit), pixel 0, pixel 1, ... pixel 783


"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data_file = "C:/Users/kko8/OneDrive/projects/master/CIS579/assignments/data/train.csv"
data = pd.read_csv(data_file)
print(data)