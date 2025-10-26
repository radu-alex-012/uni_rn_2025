import pickle
import os
import pandas as pd
import numpy as np

train_file = "extended_mnist_train.pkl"
test_file = "extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train = pickle.load(fp)

with open(test_file, "rb") as fp:
    test = pickle.load(fp)
