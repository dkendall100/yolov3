import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import explained_variance_score, \
    mean_absolute_error, \
    median_absolute_error
from sklearn.model_selection import train_test_split

df = pd.DataFrame('pred_data.csv')

dataset = df.values

X = dataset[:,0:10]
