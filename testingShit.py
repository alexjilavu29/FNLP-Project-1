
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    mean_absolute_error
)

from scipy.stats import spearmanr, kendalltau
from sklearn.preprocessing import LabelEncoder

##############################
# 1. Configuration
##############################
# NOTE: You can modify 'nrows_to_use' to limit the number of rows read from the CSV.
# For example, set nrows_to_use = 1000 to only read 1000 rows from the file.
# nrows_to_use = 50000  # Currently set to None -> read ALL rows

##############################
# 2. Load and Inspect Data
##############################

df = pd.read_csv('data-adjusted.csv', sep = ';')  # Adjust path as needed
df = df.sample(10000,random_state=17)

print("Data shape:", df.shape)
print(df.head())
print(df["topic"].describe())