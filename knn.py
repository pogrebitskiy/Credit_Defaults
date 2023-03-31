import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random as rnd
import numpy as np
from utilities import utilities as util

def knn(train_set, test_set, k=10):
    '''To perform K-Nearest Neighbors on the train and test sets'''
    predictions = []
    for test_row in test_set.values:
        distances = []
        for train_row in train_set.values:
            dist = np.sqrt(np.sum((test_row[:-1] - train_row[:-1])**2))
            distances.append((dist, train_row[-1]))
        distances.sort()
        neighbors = distances[:k]
        output = [row[-1] for row in neighbors]
        prediction = max(set(output), key=output.count)
        predictions.append(prediction)

    metrics = util.metrics(test_set.iloc[:, -1].to_list(), predictions)

    print(metrics)

    return metrics

if __name__ == '__main__':
    df = pd.read_csv('balanced_credit.csv')

    folds = util.nfolds(df, 10)

    result = util.cross_validate(folds, knn)

    print(result)

