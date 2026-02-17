#!/usr/bin/env python3
"""
CMPSC 165 - Machine Learning
Homework 2, Problem 2: Support Vector Machine (SVM)
"""

import numpy as np
import pandas as pd


def load_data(X_path: str, y_path: str = None):
    """Load features and labels from CSV files."""
    x = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    return x,y


def preprocess_data(X_train, X_test): # this code was taken from the perceptron code
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


class SVMClassifier:
    """Support Vector Machine Classifier."""

    def train(self, X, y):
        """Fit the classifier to training data."""
        epochs = 40
        weight = np.zeros(X.shape[1])
        alpha = 0.001 
        lambda_reg = 0.001

    
        for cur_epoch in range(epochs):
            for index, row in X.iterrows():
                val = y.iloc[index, 0] * (np.dot(weight, row))
                if val < 1:
                    weight = weight + alpha * (y.iloc[index, 0] * row - lambda_reg * weight) # SGD update rule 
                else:
                    weight = weight - alpha * lambda_reg * weight # regularization penalization


        self.weight = weight


    def predict(self, X):
        """Predict labels for input samples."""
        y_hat = []
        for index, row in X.iterrows():
            val = np.sign(np.dot(self.weight, row))
            if val == 0:
                val = 1
            y_hat.append(val)
        return y_hat


def evaluate(y_true, y_pred):
    """Compute classification accuracy."""
    total_num = len(y_true)
    cnt = 0
    for prediction in range(len(y_true)):
        if(y_pred[prediction] == y_true[prediction]):
            cnt += 1
    accuracy = cnt / total_num
    return accuracy


def run(Xtrain_file: str, Ytrain_file: str, test_data_file: str, pred_file: str):
    """Main function called by autograder."""
    
    
    X_train = pd.read_csv(Xtrain_file)
    Y_train = pd.read_csv(Ytrain_file)
    x_test = pd.read_csv(test_data_file)
   

    X_train, x_test = preprocess_data(X_train=X_train, X_test=x_test)


    model = SVMClassifier()

    model.train(X_train, Y_train)
    y_hat = model.predict(x_test)

    pd.DataFrame(y_hat).to_csv(pred_file, header=False, index=False)


