#!/usr/bin/env python3
"""
CMPSC 165 - Machine Learning
Homework 2, Problem 1: Voted Perceptron
"""

import numpy as np
import pandas as pd


def load_data(X_path: str, y_path: str = None):
    """Load features and labels from CSV files."""
    # TODO: Implement

    x = pd.read_csv(X_path) # input features
    y = pd.read_csv(y_path) # {+1, -1}
    return x, y


def preprocess_data(X_train, X_test):
    """Preprocess training and test data."""
    # columns with large scales: columns 56, 57 
    x_train_max55 = X_train.iloc[:, 55].max()
    x_train_max56 = X_train.iloc[:, 56].max()
    

    # min max scaling --> find max and divide by it
    X_train.iloc[:,55] = X_train.iloc[:,55] / x_train_max55
    X_train.iloc[:,56] = X_train.iloc[:,56] / x_train_max56

    X_test.iloc[:,55] = X_test.iloc[:,55] / x_train_max55
    X_test.iloc[:, 56] = X_test.iloc[:, 56] / x_train_max56

    return X_train, X_test

class VotedPerceptron:
    """Voted Perceptron Classifier."""

    def train(self, X, y):
        """Fit the classifier to training data."""
        epochs = 100
        weights = [np.zeros(X.shape[1])] # w_0
        survival = [0] # c_0
        for cur_epoch in range(epochs):
            cur_w = weights[-1] # start at most recent weight vector
            cnt = survival[-1]
            for i, row in X.iterrows():
                val = y[0][i] * (np.dot(cur_w, row)) # check if current weight preds correct
                if val <= 0:
                    survival.append(cnt)
                    cnt = 1
                    new_w = cur_w + (y[0][i] * row)
                    weights.append(new_w)
                    cur_w = new_w
                else:
                    cnt += 1
        survival.append(cnt)
        self.weights = weights
        self.survival = survival

    def predict(self, X):
        """Predict labels for input samples."""
        y_hat = []
        for index, row in X.iterrows():
            total = 0
            for i in range(len(self.weights)):
                val =  np.sign(np.dot(self.weights[i], row))
                if val == 0:
                    val = 1
                total += self.survival[i] * val
            y_hat.append(np.sign(total))
        return y_hat


def evaluate(y_true, y_pred):
    """Compute classification accuracy."""
    # TODO: Implement
    total_num = len(y_true)
    correct = 0
    for prediction in range(len(y_true)):
        if y_pred[prediction] == y_true[prediction]:
            correct += 1
    accuracy = correct / total_num
    return accuracy

def run(Xtrain_file: str, Ytrain_file: str, test_data_file: str, pred_file: str):
    """Main function called by autograder."""

    X_train = pd.read_csv(Xtrain_file)
    Y_train = pd.read_csv(Ytrain_file)
    x_test = pd.read_csv(test_data_file)
    y_test = pd.read_csv(pred_file)


    X_train, x_test = preprocess_data(X_train=X_train, X_test=x_test)


    model = VotedPerceptron()

    model.train(X_train, Y_train)
    y_hat = model.predict(x_test)

    accuracy = evaluate(y_true=y_test, y_pred=y_hat)
    pd.DataFrame(y_hat).to_csv(pred_file, header=False, index=False)
