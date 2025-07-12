"""
naive_bayes_trainer.py

Trains and evaluates a Naive Bayes classifier on a CSV dataset.
Demonstrates classic supervised learning workflow including:
- Data loading and inspection
- Train-test split
- Model training
- Accuracy evaluation and classification report

Author: [Your Name]
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

def load_data(file_path):
    """
    Loads the CSV data into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)


def preprocess_data(df):
    """
    Splits features and target variable.

    Args:
        df (pd.DataFrame): Dataset.

    Returns:
        (pd.DataFrame, pd.Series): Features (X) and target (y).
    """
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Trains a Gaussian Naive Bayes model and evaluates it.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Testing target.

    Prints:
        Accuracy score and classification report.
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


def main():
    """
    Main workflow: load data, split, train, and evaluate.
    """
    data_path = '../data/Naive-Bayes-Classification-Data.csv'
    df = load_data(data_path)
    print("First 5 rows of the dataset:\n", df.head())

    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_and_evaluate(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
