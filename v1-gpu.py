#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
nrows_to_use = 50000  # Currently set to None -> read ALL rows

##############################
# 2. Load and Inspect Data
##############################

df = pd.read_csv('data-adjusted.csv', nrows=nrows_to_use, sep = ';')  # Adjust path as needed

print("Data shape:", df.shape)
print(df.head())

##############################
# 3. Basic Text Cleaning
##############################

def clean_text(text):
    """
    Basic text cleaning:
    - Lowercase
    - Remove typical URL patterns
    - Remove extra non-alphanumeric characters
    - Convert multiple spaces to single space
    """
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', ' ', text)  # remove URLs
    text = re.sub(r'[^a-z0-9\s.,!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text_clean'] = df['text'].astype(str).apply(clean_text)

##############################
# 4. Encode Targets
##############################

# 4.1 Encode gender: male -> 0, female -> 1
df['gender_encoded'] = df['gender'].map({'male': 0, 'female': 1})

# 4.2 Age stays numeric (for regression)
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# 4.3 Encode topic as multi-class
topic_encoder = LabelEncoder()
df['topic_encoded'] = topic_encoder.fit_transform(df['topic'].astype(str))

##############################
# 5. Split Train/Test
##############################

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

print("Train size:", train_df.shape)
print("Test size:", test_df.shape)

##############################
# 6. Feature Extraction (Tfidf)
##############################

tfidf = TfidfVectorizer(
    max_features=5000,     # can tune
    ngram_range=(1, 2),    # can tune
    stop_words='english'   # can remove or change
)
tfidf.fit(train_df['text_clean'])

X_train = tfidf.transform(train_df['text_clean'])
X_test = tfidf.transform(test_df['text_clean'])

##############################
# 7. Define Models
##############################

# For classification tasks (gender, topic):
classification_models = {
    # RandomForest on CPU
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    # XGBoost on GPU
    'XGB': XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        #tree_method='gpu_hist',       # GPU usage
        #predictor='gpu_predictor'
    ),
}

# For regression tasks (age):
regression_models = {
    # RandomForest on CPU
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    # XGBoost on GPU
    'XGB': XGBRegressor(
        random_state=42,
        #tree_method='gpu_hist',       # GPU usage
        #predictor='gpu_predictor'
    ),
}

##############################
# 8. Metrics
##############################

def evaluate_classification(y_true, y_pred):
    """
    Return a dict with: Accuracy, Spearman, Kendall, MAE, MSE
    """
    acc = accuracy_score(y_true, y_pred)
    spear = spearmanr(y_true, y_pred).correlation
    kend = kendalltau(y_true, y_pred).correlation
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        'Accuracy': acc,
        'Spearman': spear,
        'Kendall': kend,
        'MAE': mae,
        'MSE': mse
    }

def evaluate_regression(y_true, y_pred):
    """
    Return a dict with: Accuracy (NaN), Spearman, Kendall, MAE, MSE
    """
    spear = spearmanr(y_true, y_pred).correlation
    kend = kendalltau(y_true, y_pred).correlation
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        'Accuracy': np.nan,  # Not relevant for regression
        'Spearman': spear,
        'Kendall': kend,
        'MAE': mae,
        'MSE': mse
    }

##############################
# 9. Train & Evaluate
##############################

results = []

# --- 9A. Gender (binary classification) ---
y_train_gender = train_df['gender_encoded'].values
y_test_gender = test_df['gender_encoded'].values

for model_name, model in classification_models.items():
    model.fit(X_train, y_train_gender)
    y_pred_gender = model.predict(X_test)
    metrics = evaluate_classification(y_test_gender, y_pred_gender)
    results.append({
        'Characteristic': 'Gender',
        'Model': model_name,
        **metrics
    })

# --- 9B. Age (regression) ---
y_train_age = train_df['age'].values
y_test_age = test_df['age'].values

for model_name, model in regression_models.items():
    model.fit(X_train, y_train_age)
    y_pred_age = model.predict(X_test)
    metrics = evaluate_regression(y_test_age, y_pred_age)
    results.append({
        'Characteristic': 'Age',
        'Model': model_name,
        **metrics
    })

# --- 9C. Topic (multi-class classification) ---
y_train_topic = train_df['topic_encoded'].values
y_test_topic = test_df['topic_encoded'].values

for model_name, model in classification_models.items():
    model.fit(X_train, y_train_topic)
    y_pred_topic = model.predict(X_test)
    metrics = evaluate_classification(y_test_topic, y_pred_topic)
    results.append({
        'Characteristic': 'Topic',
        'Model': model_name,
        **metrics
    })

##############################
# 10. Results Table
##############################

results_df = pd.DataFrame(
    results,
    columns=['Characteristic', 'Model', 'Accuracy', 'Spearman', 'Kendall', 'MAE', 'MSE']
)

print("\n===== Final Results =====")
print(results_df)

# Optionally, save to CSV
results_df.to_csv('model_evaluation_results.csv', index=False)

print("\nDone!")