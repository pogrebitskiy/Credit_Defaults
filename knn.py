import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random as rnd
import numpy as np


def nfolds(df, n_folds):
    '''To split the data into a given number of folds'''
    # Find the size of each fold
    fold_size = len(df) // n_folds
    df = df.sample(frac=1).reset_index(drop=True)
    folds = []

    # Create each fold
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size
        if i == n_folds - 1:
            end = len(df)
        folds.append(df[start:end])

    return folds

def knn(train_set, test_set, k):
    '''To perform K-Nearest Neighbors on the train and test sets'''
    predictions = []
    for test_row in test_set:
        distances = []
        for train_row in train_set:
            dist = np.sqrt(np.sum((test_row[:-1] - train_row[:-1])**2))
            distances.append((dist, train_row[-1]))
        distances.sort()
        neighbors = distances[:k]
        output = [row[-1] for row in neighbors]
        prediction = max(set(output), key=output.count)
        predictions.append(prediction)
    return predictions

if __name__ == '__main__':
    df = pd.read_csv('balanced_credit.csv')

    folds = nfolds(df, 10)

