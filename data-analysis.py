#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script:
1. Loads the 'data.csv' file (with columns: id, gender, age, topic, text).
2. Computes and prints relevant statistics on gender, age, and topic.
3. Visualizes these distributions with various plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # 1. Load Data
    data_path = "data.csv"  # Adjust if needed
    print("Loading dataset...")
    df = pd.read_csv(data_path, sep=";", encoding="utf-8",
                     on_bad_lines="skip")  # or sep=";" if your CSV is semicolon-delimited

    # Quick check of columns (uncomment if your file uses semicolons)
    # df = pd.read_csv(data_path, sep=";", encoding="utf-8", on_bad_lines="skip")

    print("Data shape:", df.shape)
    print(df.head())
    print("\nColumns:", df.columns.to_list())

    # 2. Basic Data Overview
    print("\nData Info:")
    print(df.info())

    print("\nData Describe (numeric columns):")
    print(df.describe())

    # 3. Gender Analysis
    # Check unique values in gender
    print("\nUnique gender values:", df['gender'].unique())

    # Drop rows if gender is missing or invalid, or just count them
    gender_counts = df['gender'].value_counts(dropna=False)
    total_entries = len(df)
    print(f"\nGender Counts:\n{gender_counts}")

    # Percentages
    gender_percentages = (gender_counts / total_entries) * 100
    print("\nGender Percentages:")
    print(gender_percentages)

    # Bar chart for gender distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='gender', data=df, order=gender_counts.index)
    plt.title("Gender Distribution")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # (Optional) Pie chart for gender
    plt.figure(figsize=(5, 5))
    gender_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140)
    plt.ylabel("")
    plt.title("Gender Distribution")
    plt.show()

    # 4. Age Analysis
    # Check if age is numeric or if there are missing values
    print("\nAge Info:")
    print(df['age'].describe())  # mean, std, min, max
    # Print the number of entries for each age found
    print("\nAge Value Counts:")
    print(df['age'].value_counts().sort_index())

    # Histogram of age
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='age', bins=30, kde=False)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Box plot of age (identifying outliers)
    plt.figure(figsize=(4, 6))
    sns.boxplot(y='age', data=df)
    plt.title("Age Box Plot")
    plt.ylabel("Age")
    plt.tight_layout()
    plt.show()

    # Age groups (e.g., 0–9, 10–19, 20–29, etc.)
    # If you expect only certain ages, adjust bins accordingly
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    df['age_group'] = pd.cut(df['age'], bins=age_bins)
    age_group_counts = df['age_group'].value_counts().sort_index()
    print("\nAge Group Distribution:")
    print(age_group_counts)

    # Bar chart of age groups
    plt.figure(figsize=(8, 5))
    age_group_counts.plot(kind='bar')
    plt.title("Age Group Distribution")
    plt.xlabel("Age Group")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # 5. Topic Analysis
    print("\nUnique Topics:", df['topic'].nunique(), "unique topics.")
    topic_counts = df['topic'].value_counts().head(20)  # top 20
    print("\nTop 20 Topics:\n", topic_counts)

    # Bar chart of top 20 topics
    plt.figure(figsize=(10, 6))
    sns.barplot(x=topic_counts.values, y=topic_counts.index)
    plt.title("Top 20 Topics Distribution")
    plt.xlabel("Count")
    plt.ylabel("Topic")
    plt.tight_layout()
    plt.show()

    # If you want to see distribution of all topics (could be large):
    # topic_counts_all = df['topic'].value_counts()
    # plt.figure(figsize=(10,20))
    # sns.barplot(x=topic_counts_all.values, y=topic_counts_all.index)
    # plt.title("All Topics Distribution")
    # plt.xlabel("Count")
    # plt.ylabel("Topic")
    # plt.tight_layout()
    # plt.show()

    # 6. Additional Analysis (Optional)
    # For example, mean age by gender:
    mean_age_by_gender = df.groupby('gender')['age'].mean()
    print("\nMean Age by Gender:")
    print(mean_age_by_gender)

    # Visualize mean age by gender:
    plt.figure(figsize=(6, 4))
    mean_age_by_gender.plot(kind='bar', color=['blue', 'pink'])
    plt.title("Mean Age by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Mean Age")
    plt.tight_layout()
    plt.show()

    print("\nDone with analysis!")


if __name__ == "__main__":
    main()