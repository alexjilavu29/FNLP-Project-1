#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script demonstrates how to:
1. Load and clean text data.
2. Encode targets (gender, age, topic).
3. Split train/test.
4. Generate TF-IDF or BERT embeddings.
5. Train multiple models (RandomForest, XGB, BERT-based classifier) on each characteristic.
6. Evaluate and compare results.
"""

import pandas as pd
import numpy as np
import re

# Sklearn
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

# Hugging Face Transformers
import torch
from transformers import BertTokenizer, BertModel

##############################
# 1. Configuration & Flags
##############################

# NOTE: You can modify 'nrows_to_use' to limit the number of rows read from the CSV.
# For example, set nrows_to_use = 1000 to only read 1000 rows from the file.
# If set to None, it reads ALL rows.
nrows_to_use = 10000

# Model usage flags - set any to False to skip that model
USE_RANDOM_FOREST = False
USE_XGB = True
USE_BERT = False   # Will do BERT embeddings for topic classification

# CSV reading parameters (adjust delimiter as needed)
CSV_PATH = 'data.csv'
CSV_SEP = ';'   # e.g., ';' if semicolon-delimited, ',' if comma-delimited

# For demonstration, we'll use BERT only for the "Topic" classification.
# You could also apply BERT embeddings to "Gender" or "Age" tasks if you wish.

##############################
# 2. Load and Inspect Data
##############################

df = pd.read_csv(CSV_PATH, sep=CSV_SEP)
if nrows_to_use is not None:
    # randomly sample 'nrows_to_use' rows for training, if you want
    df = df.sample(n=nrows_to_use, random_state=42)

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

# 4.2 Convert age to numeric (for regression)
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
# 6. Feature Extraction
##############################

# 6A. TF-IDF (used for RandomForest / XGB by default)
tfidf = TfidfVectorizer(
    max_features=5000,     # can tune
    ngram_range=(1, 2),    # can tune
    stop_words='english'   # can remove or change
)
tfidf.fit(train_df['text_clean'])

X_train_tfidf = tfidf.transform(train_df['text_clean'])
X_test_tfidf = tfidf.transform(test_df['text_clean'])

# 6B. BERT Embeddings (for topic classification, or any classification you choose)
#    We'll define a helper function to embed a list of texts with a pretrained BERT model.

device = torch.device("mps" if torch.mps.is_available() else "cpu")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

def get_bert_embeddings(texts, batch_size=16):
    """
    Convert a list of texts into BERT embeddings by:
    1. Tokenizing and encoding with bert_tokenizer
    2. Passing through BertModel
    3. Average pooling the last hidden states
    Return a NumPy array of shape [len(texts), 768] for 'bert-base-uncased'.
    """
    all_embeddings = []
    # Process in batches to avoid out-of-memory
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = bert_tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        )
        # Move to GPU if available
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = bert_model(input_ids, attention_mask=attention_mask)
            # outputs.last_hidden_state shape: [batch_size, seq_len, hidden_dim=768]
            # We'll do a simple mean pooling across seq_len
            last_hidden = outputs.last_hidden_state
            mean_pooled = torch.mean(last_hidden, dim=1)  # shape: [batch_size, 768]

        # Convert to CPU numpy
        all_embeddings.append(mean_pooled.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings

# We'll only compute BERT embeddings if USE_BERT is True
X_train_bert = None
X_test_bert = None

if USE_BERT:
    print("Generating BERT embeddings for train set (topic)...")
    X_train_bert = get_bert_embeddings(train_df['text_clean'].tolist(), batch_size=16)
    print("Train BERT shape:", X_train_bert.shape)

    print("Generating BERT embeddings for test set (topic)...")
    X_test_bert = get_bert_embeddings(test_df['text_clean'].tolist(), batch_size=16)
    print("Test BERT shape:", X_test_bert.shape)

##############################
# 7. Define Models
##############################

# We create dictionaries for classification and regression.
# We'll fill them conditionally based on the flags.

classification_models = {}
regression_models = {}

# --- Classification (Gender, Topic) ---
if USE_RANDOM_FOREST:
    classification_models['RandomForest'] = RandomForestClassifier(n_estimators=100, random_state=42)

if USE_XGB:
    classification_models['XGB'] = XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        n_estimators=700,
        max_depth=8,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=3,
        # Uncomment if you have GPU build of XGBoost
        # tree_method='gpu_hist',
        # predictor='gpu_predictor'
    )

# We'll add a "BERTTopicClassifier" (simple RandomForest or XGB on BERT embeddings) just for topic
if USE_BERT:
    # For demonstration, let's do a RandomForest on top of BERT embeddings for topic classification
    classification_models['BERTTopic_RF'] = RandomForestClassifier(n_estimators=100, random_state=42)
    # Or you can add more (XGB on top of BERT, etc.)

# --- Regression (Age) ---
if USE_RANDOM_FOREST:
    regression_models['RandomForest'] = RandomForestRegressor(n_estimators=100, random_state=42)

if USE_XGB:
    regression_models['XGB'] = XGBRegressor(
        random_state=42,
        n_estimators=700,
        max_depth=8,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=3,
        # Uncomment if you have GPU build of XGBoost
        # tree_method='gpu_hist',
        # predictor='gpu_predictor'
    )

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

# Prepare target arrays
y_train_gender = train_df['gender_encoded'].values
y_test_gender = test_df['gender_encoded'].values

y_train_age = train_df['age'].values
y_test_age = test_df['age'].values

y_train_topic = train_df['topic_encoded'].values
y_test_topic = test_df['topic_encoded'].values

##############################
# 9A. Gender (binary classification)
##############################
# We'll train each classification model on TF-IDF features for gender
for model_name, model in classification_models.items():
    # If it's our BERTTopic_RF, skip it for gender
    if model_name == 'BERTTopic_RF':
        continue

    print(f"\nTraining {model_name} for Gender classification...")
    model.fit(X_train_tfidf, y_train_gender)
    y_pred_gender = model.predict(X_test_tfidf)
    metrics = evaluate_classification(y_test_gender, y_pred_gender)
    results.append({
        'Characteristic': 'Gender',
        'Model': model_name,
        **metrics
    })

##############################
# 9B. Age (regression)
##############################
# We'll train each regression model on TF-IDF features
for model_name, model in regression_models.items():
    print(f"\nTraining {model_name} for Age regression...")
    model.fit(X_train_tfidf, y_train_age)
    y_pred_age = model.predict(X_test_tfidf)
    metrics = evaluate_regression(y_test_age, y_pred_age)
    results.append({
        'Characteristic': 'Age',
        'Model': model_name,
        **metrics
    })

##############################
# 9C. Topic (multi-class classification)
##############################
# We'll train each classification model for Topic.
# - If it's 'BERTTopic_RF', we use the BERT embeddings (X_train_bert, X_test_bert).
# - Otherwise, we use TF-IDF.
for model_name, model in classification_models.items():
    print(f"\nTraining {model_name} for Topic classification...")

    if model_name == 'BERTTopic_RF' and USE_BERT:
        # Use BERT embeddings
        model.fit(X_train_bert, y_train_topic)
        y_pred_topic = model.predict(X_test_bert)
    else:
        # Use TF-IDF
        model.fit(X_train_tfidf, y_train_topic)
        y_pred_topic = model.predict(X_test_tfidf)

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
results_df.to_csv('model_evaluation_results-v2.csv', index=False)

print("\nDone!")