#!/usr/bin/env python3
"""
CMPSC 165 - Machine Learning
Homework 2, Problem 2: Support Vector Machine (SVM)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(X_path: str, y_path: str = None):
    """Load features and labels from CSV files."""
    x = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    return x,y


def preprocess_data(X_train, X_test): # this code was taken from the perceptron code
    """Preprocess training and test data."""
    train_mean = X_train.mean(axis=0)                                                                                                                                                                      
    train_std = X_train.std(axis=0)

    train_std[train_std == 0] = 1                                                                                                                                                                          

    X_train = (X_train - train_mean) / train_std                                                                                                                                                           
    X_test = (X_test - train_mean) / train_std

    X_train['bias'] = 1                                                                                                                                                                                    
    X_test['bias'] = 1       

    return X_train, X_test


class SVMClassifier:
    """Support Vector Machine Classifier."""

    def train(self, X, y, lambda_reg):
        """Fit the classifier to training data."""
        epochs = 60
        weight = np.zeros(X.shape[1])
        alpha = 0.01 
        # lambda_reg = 0.001

    
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

    model.train(X_train, Y_train, lambda_reg=0.001)
    y_hat = model.predict(x_test)

    pd.DataFrame(y_hat).to_csv(pred_file, header=False, index=False)


# if __name__ == "__main__":
#     x,y = load_data("/Users/parth/Desktop/spam-prediction/data/spam_X.csv", "/Users/parth/Desktop/spam-prediction/data/spam_y.csv")

#     n = len(x)
#     index = int(n * 0.9)

#     X_train = x.iloc[:index].copy()
#     Y_train = y.iloc[:index].copy() # 90 percent

    
#     x_test = x.iloc[index:].copy() # 10 percent 
#     y_test = y.iloc[index:].copy()


#     X_train, x_test = preprocess_data(X_train, x_test)

#     lambda_set = [0.0001, 0.001, 0.01, 0.1, 1, 10]
#     accuracies = []

#     for cur_lambda in lambda_set:
#         model = SVMClassifier()

#         model.train(X_train, Y_train, cur_lambda)
#         y_hat = model.predict(x_test)
#         y_true = y_test.iloc[:, 0].to_list()
#         accuracy = evaluate(y_true, y_hat)
#         accuracies.append(accuracy)

#     pct = lambda_set
#     plt.figure(figsize=(8, 5))
#     plt.plot(pct, accuracies, 'bo-', linewidth=2, markersize=8)
#     plt.xlabel("Lambda")
#     plt.ylabel("Accuracy")
#     plt.title("SVM: Accuracy vs Lambda")
#     plt.xscale('log')
#     plt.xticks(pct, [str(l) for l in pct])
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig("svm_accuracy_plot.png", dpi=150)
#     plt.show()
    



