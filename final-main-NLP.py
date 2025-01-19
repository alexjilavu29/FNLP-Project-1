import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.metrics import f1_score
from scipy.stats import spearmanr, kendalltau
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import BertTokenizer, BertModel
from gensim.models import word2vec
import gensim.downloader
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC, SVR


#  Get Data
def get_data(samples_to_use, csv_file):
    df = pd.read_csv(csv_file, sep=";")
    if samples_to_use is not None:
        df = df.sample(n=samples_to_use, random_state=17)
    return df


def show_data(df):
    print("Shape: ")
    print(df.shape)
    print("Head: ")
    print(df.head())
    print("\n\n")


#  Preprocessing

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_data(df):
    df["gender_encoded"] = df["gender"].map({"male": 0, "female": 1})
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    topic_encoder = LabelEncoder()
    df["topic_encoded"] = topic_encoder.fit_transform(df["topic"].astype(str))
    return df


#  Extract features

def tokenise(tokeniser, text_column_train, text_column_test):
    match tokeniser:
        case "n_grams":
            return n_grams_with_tfidf(text_column_train, text_column_test)
        case "TF-IDF":
            return tokenise_tf_idf(text_column_train, text_column_test)
        case "BERT":
            return tokenise_bert(text_column_train, text_column_test)
        case "W2V":
            return tokenise_w2v(text_column_train, text_column_test)
        case _:
            return tokenise_w2v(text_column_train, text_column_test)


def n_grams_with_tfidf(text_column_train, text_column_test):
    print(f"Tokenising data with n_grams")
    word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer="word")
    character_vectorizer = CountVectorizer(ngram_range=(3, 5), analyzer="char")
    feature_union = FeatureUnion([('tfidf', word_vectorizer), ('charvect', character_vectorizer)])
    feature_union.fit(text_column_train)
    return feature_union.transform(text_column_train), feature_union.transform(text_column_test)


def tokenise_tf_idf(text_column_train, text_column_test):
    print(f"Tokenising data with TF-IDF")
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    tfidf.fit(text_column_train)
    return tfidf.transform(text_column_train), tfidf.transform(text_column_test)


def tokenise_bert(text_column_train, text_column_test):
    print(f"Tokenising data with BERT")
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

    X_train_bert = get_bert_embeddings(bert_tokenizer, bert_model, device, text_column_train.tolist(), batch_size=16)
    X_test_bert = get_bert_embeddings(bert_tokenizer, bert_model, device, text_column_test.tolist(), batch_size=16)
    return X_train_bert, X_test_bert


def get_bert_embeddings(tokeniser, model, device, texts, batch_size=16):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        encoded = tokeniser(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state
            mean_pooled = torch.mean(last_hidden, dim=1)

        all_embeddings.append(mean_pooled.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings


def tokenise_w2v(text_column_train, text_column_test):
    print(f"Tokenising data with W2V")
    word2vec = gensim.downloader.load("word2vec-google-news-300")
    return compute_w2v(text_column_train, word2vec), compute_w2v(text_column_test, word2vec)


def compute_w2v(text_column, word2vec):
    train_emb = []
    for row in text_column:
        words = row.split(" ")
        words = filter(lambda x: x in word2vec, words)
        text_emb = [word2vec[word] for word in words]

        if len(text_emb) == 0:
            train_emb.append(np.zeros(300))
            continue

        doc_embedding = np.mean(text_emb, axis=0)
        train_emb.append(doc_embedding)
    return np.array(train_emb)


#  Apply models
def use_SVM_Models(X_train, y_train_all, X_test, y_test_all):
    print("Training for gender classification:")
    gender_results = SVM_classifier(X_train, y_train_all["gender_encoded"], X_test, y_test_all["gender_encoded"])
    print("Training for age regression:")
    age_results = SVM_regressor(X_train, y_train_all["age"], X_test, y_test_all["age"])
    print("Training for topic classification:")
    topic_results = SVM_classifier(X_train, y_train_all["topic_encoded"], X_test, y_test_all["topic_encoded"])
    return [gender_results, age_results, topic_results]

def SVM_classifier(X_train,y_train,X_test,y_test):
    svm = SVC(kernel="linear",random_state=17)
    y_pred = train_and_predict(svm,X_train,y_train,X_test)
    metrics = classification_results(y_test,y_pred)
    print(metrics)
    return metrics

def SVM_regressor(X_train,y_train,X_test,y_test):
    svm = SVR(kernel="linear")
    y_pred = train_and_predict(svm,X_train,y_train,X_test)
    metrics = classification_results(y_test,y_pred)
    print(metrics)
    return metrics

def use_XGB_Models(X_train, y_train_all, X_test, y_test_all):
    print("Training for gender classification:")
    gender_results = XGB_classifier(X_train, y_train_all["gender_encoded"], X_test, y_test_all["gender_encoded"])
    print("Training for age regression:")
    age_results = XGB_regressor(X_train, y_train_all["age"], X_test, y_test_all["age"])
    print("Training for topic classification:")
    topic_results = XGB_classifier(X_train, y_train_all["topic_encoded"], X_test, y_test_all["topic_encoded"])
    return [gender_results, age_results, topic_results]


def XGB_classifier(X_train, y_train, X_test, y_test):
    classifier = XGBClassifier(
        use_label_encoder=False,
        random_state=17,
        eval_metrics="logloss",
        n_estimators=1200,
        max_depth=5,
        min_child_weight=1,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=1.0,
        gamma=0,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )
    y_pred = train_and_predict(classifier, X_train, y_train, X_test)
    metrics = classification_results(y_test, y_pred)
    print("Classification results:")
    print(metrics)
    return metrics


def XGB_regressor(X_train, y_train, X_test, y_test):
    regressor = XGBRegressor(
        use_label_encoder=False,
        random_state=17,
        n_estimators=1200,
        max_depth=5,
        learning_rate=0.05,
        subsample=1,
        colsample_bytree=1,
        gamma=0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=1,
    )
    y_pred = train_and_predict(regressor, X_train, y_train, X_test)
    metrics = regression_results(y_test, y_pred)
    print("Regression results:")
    print(metrics)
    return metrics


def use_Random_Forest(X_train, y_train_all, X_test, y_test_all):
    print("Training for gender classification:")
    gender_results = RF_classifier(X_train, y_train_all["gender_encoded"], X_test, y_test_all["gender_encoded"])
    print("Training for age regression:")
    age_results = RF_regressor(X_train, y_train_all["age"], X_test, y_test_all["age"])
    print("Training for topic classification:")
    topic_results = RF_classifier(X_train, y_train_all["topic_encoded"], X_test, y_test_all["topic_encoded"])
    return [gender_results, age_results, topic_results]


def RF_classifier(X_train, y_train, X_test, y_test):
    classifier = RandomForestClassifier(n_estimators=100, random_state=17)
    y_pred = train_and_predict(classifier, X_train, y_train, X_test)
    metrics = classification_results(y_test, y_pred)
    print("Classification results:")
    print(metrics)
    return metrics


def RF_regressor(X_train, y_train, X_test, y_test):
    classifier = RandomForestRegressor(n_estimators=100, random_state=17)
    y_pred = train_and_predict(classifier, X_train, y_train, X_test)
    metrics = regression_results(y_test, y_pred)
    print("Regression results:")
    print(metrics)
    return metrics


def train_and_predict(model, X_train, y_train, X_test):
    y_train = y_train.values
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred


#  Results testing
def classification_results(y_true, y_pred):
    MAE = mean_absolute_error(y_true, y_pred)
    MSE = mean_squared_error(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    Spearman = spearmanr(y_true, y_pred).correlation
    Kendall = kendalltau(y_true, y_pred).correlation
    F1 = f1_score(y_true, y_pred, average="weighted")
    return {
        "MAE": MAE,
        "MSE": MSE,
        "Accuracy": accuracy,
        "Spearman": Spearman,
        "Kendall": Kendall,
        "F1-score": F1
    }


def regression_results(y_true, y_pred):
    MAE = mean_absolute_error(y_true, y_pred)
    MSE = mean_squared_error(y_true, y_pred)
    Spearman = spearmanr(y_true, y_pred).correlation
    Kendall = kendalltau(y_true, y_pred).correlation
    return {
        "MAE": MAE,
        "MSE": MSE,
        "Accuracy": np.nan,
        "Spearman": Spearman,
        "Kendall": Kendall,
        "F1-score": np.nan
    }


#  Main
def main():
    # Get data
    df = get_data(10000, "data.csv")
    show_data(df)
    # Clean data
    df["text_clean"] = df["text"].astype(str).apply(clean_text)
    df = encode_data(df)
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=17, shuffle=True)
    show_data(train_df)
    show_data(test_df)
    # Extract features
    TOKENISER = "TF-IDF"
    X_train_tokenised, X_test_tokenised = tokenise(TOKENISER, train_df["text_clean"], test_df["text_clean"])
    # Train and test
    y_train = train_df[["gender_encoded", "age", "topic_encoded"]]
    y_test = test_df[["gender_encoded", "age", "topic_encoded"]]
    print("XGB model:")
    results_xgb = use_XGB_Models(X_train_tokenised, y_train, X_test_tokenised, y_test)
    XGB_results_df = pd.DataFrame(results_xgb, columns=["MAE", "MSE", "Accuracy", "Spearman", "Kendall"],
                                  index=["XGB - Gender", "XGB - Age", "XGB - Topic"])
    print(XGB_results_df)
    print("RF model:")
    results_rf = use_Random_Forest(X_train_tokenised, y_train, X_test_tokenised, y_test)
    RF_results_df = pd.DataFrame(results_rf, columns=["MAE", "MSE", "Accuracy", "Spearman", "Kendall"],
                                 index=["RF - Gender", "RF - Age", "RF - Topic"])
    print(RF_results_df)

    print("Final results:")
    final_results = pd.concat([XGB_results_df, RF_results_df])
    print(final_results)
    final_results.to_csv("model_evaluation.csv", index=True)


main()
