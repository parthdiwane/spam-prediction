#!/usr/bin/env python3
"""
CMPSC 165 - Machine Learning
Homework 2, Problem 1: Voted Perceptron
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(X_path: str, y_path: str = None):
    """Load features and labels from CSV files."""
    # TODO: Implement

    x = pd.read_csv(X_path) # input features
    y = pd.read_csv(y_path) # {+1, -1}
    return x, y


def preprocess_data(X_train, X_test):
    """Preprocess training and test data."""
    train_mean = X_train.mean(axis=0)                                                                                                                                                                      
    train_std = X_train.std(axis=0)

    train_std[train_std == 0] = 1                                                                                                                                                                          

    X_train = (X_train - train_mean) / train_std                                                                                                                                                           
    X_test = (X_test - train_mean) / train_std

    return X_train, X_test

class VotedPerceptron:
    """Voted Perceptron Classifier."""

    def train(self, X, y):
        """Fit the classifier to training data."""
        epochs = 40
        weights = [np.zeros(X.shape[1])] # w_0
        survival = [0] # c_0
        for cur_epoch in range(epochs):
            cur_w = weights[-1] # start at most recent weight vector
            cnt = survival[-1]
            for i, row in X.iterrows():
                val = y.iloc[i, 0]* (np.dot(cur_w, row)) # check if current weight preds correct
                if val <= 0:
                    survival.append(cnt)
                    cnt = 1
                    new_w = cur_w + (y.iloc[i, 0] * row)
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


    X_train, x_test = preprocess_data(X_train=X_train, X_test=x_test)


    model = VotedPerceptron()

    model.train(X_train, Y_train)
    y_hat = model.predict(x_test)

    pd.DataFrame(y_hat).to_csv(pred_file, header=False, index=False)

if __name__ == "__main__":
    x,y = load_data("/Users/parth/Desktop/spam-prediction/data/spam_X.csv", "/Users/parth/Desktop/spam-prediction/data/spam_y.csv")

    n = len(x)
    index = int(n * 0.9)

    X_train = x.iloc[:index].copy()
    Y_train = y.iloc[:index].copy() # 90 percent
    
    x_test = x.iloc[index:].copy() # 10 percent 
    y_test = y.iloc[index:].copy()

    percentage = [0.01, 0.02, 0.05, 0.1, 0.2, 1]
    accuracies = []

    for amount in percentage:
        index_new = int(amount * len(X_train))

        X_train_percent = X_train.iloc[:index_new].reset_index(drop=True)
        Y_train_percent= Y_train.iloc[:index_new].reset_index(drop=True)
        x_test_cpy = x_test.reset_index(drop=True)

        x_train_preprocessed, x_test_preprocessed = preprocess_data(X_train_percent.copy(), x_test_cpy.copy())

        model = VotedPerceptron()

        model.train(x_train_preprocessed, Y_train_percent)
        y_hat = model.predict(x_test_preprocessed)

        y_true = y_test.iloc[:,0].to_list()
        accuracy = evaluate(y_true, y_hat)

        accuracies.append(accuracy)
    

    pct = [f * 100 for f in percentage]
    plt.figure(figsize=(8, 5))
    plt.plot(pct, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel("Percent of Remaining Training Data")
    plt.ylabel("Accuracy")
    plt.title("Voted Perceptron: Accuracy vs Training Data Fraction")
    plt.xticks(pct, [f"{p:.0f}%" for p in pct])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()