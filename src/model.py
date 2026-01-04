"""
Kumparan's Model Interface

This is an interface file to implement your model.

You must implement `train` method and `predict` method.

`train` is a method to train your model. You can read
training data, preprocess and perform the training inside
this method.

`predict` is a method to run the prediction using your
trained model. This method depends on the task that you
are solving, please read the instruction that sent by
the Kumparan team for what is the input and the output
of the method.

In this interface, we implement `save` method to helps you
save your trained model. You may not edit this directly.

You can add more initialization parameter and define
new methods to the Model class.

Usage:
Install `kumparanian` first:

    pip install kumparanian

Run

    python model.py

It will run the training and save your trained model to
file `model.pickle`.
"""

from kumparanian import ds

# Import your libraries here
# Example:
# import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import re
import string
import time

# Helper function for text standadrization
def standardization(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = " ".join(text.split())
    return text

class Model:

    def __init__(self):
        """
        You can add more parameter here to initialize your model
        """
        self.pipeline = None

    def train(self):
        """
        NOTE: Implement your training procedure in this method.
        """

        # Examples; psuedocode
        # data = read_dataset("file.csv")
        # self.network = torch.RNN ...
        # self.network.train(data)
        print("Loading dataset...")
        try:
            df = pd.read_csv('data.csv')
        except FileNotFoundError:
            print("Error: data.csv not found. Please ensure it is in the same directory.")
            return
        print("Cleaning and Preprocessing data...")
        whitespace_mask = df["article_content"].str.strip() == ""
        df.loc[whitespace_mask, "article_content"] = np.nan
        df = df.dropna(subset=["article_content"])
        topic_counts = df.groupby("article_content")["article_topic"].nunique()
        conflicting_contents = topic_counts[topic_counts > 1].index
        df = df[~df["article_content"].isin(conflicting_contents)]
        df = df.drop_duplicates(subset=["article_content"], keep='first')
        df["word_count"] = df["article_content"].apply(lambda x: len(str(x).split()))
        df = df[df["word_count"] >= 20].copy()
        df["standardized_content"] = df["article_content"].apply(standardization)
        X = df["standardized_content"]
        y = df["article_topic"]
        print("Initializing Ensemble Model (LinearSVC + LogisticRegression)...")
        svc = LinearSVC(class_weight='balanced', random_state=777)
        calibrated_svc = CalibratedClassifierCV(svc, method='sigmoid', cv=5)
        lr = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=777)
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("voting", VotingClassifier(
                estimators=[
                    ("svc", calibrated_svc),
                    ("lr", lr)
                ],
                voting='soft'
            ))
        ])
        print("Training model...")
        start_time = time.time()
        self.pipeline.fit(X, y)
        end_time = time.time()
        print(f"Training finished in {end_time - start_time:.2f} seconds.")

    def predict(self, input):
        """
        NOTE: Implement your predict procedure in this method.
        """

        # Examples; psuedocode
        # processed_input = process_input(input)
        # output = self.network.forward(processed_input)
        # label = get_label(output)
        # return label
        if self.pipeline is None:
            raise Exception("Model has not been trained yet. Call model.train() first.")
        standardized_input = standardization(input)
        prediction = self.pipeline.predict([standardized_input])[0]
        return prediction

    def save(self):
        """
        Save trained model to model.pickle file.
        """
        ds.model.save(self, "model.pickle")


if __name__ == '__main__':
    # NOTE: Edit this if you add more initialization parameter
    model = Model()

    # Train your model
    model.train()

    # Save your trained model to model.pickle
    model.save()
