import pandas as pd
import math
import numpy as np
from utilities import utilities as util
import itertools
from tqdm import tqdm

def euclidean_dist(point1, point2):
    '''Calculate the euclidean distance between two points'''
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

def manhattan_dist(point1, point2):
    """Calculate the Manhattan distance between two points."""
    distance = 0
    for i in range(len(point1)):
        distance += abs(point1[i] - point2[i])
    return distance

def cosine_dist(vector1, vector2):
    """
    Calculate the cosine distance between two vectors.
    """
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    return 1 - (dot_product / (norm1 * norm2))

def knn(train_set, test_set, hyperparams):
    '''To perform K-Nearest Neighbors on the train and test sets'''
    predictions = []
    best_hyperparams = None
    best_acc = 0.0
    best_metrics = None

    for k, distance_model in tqdm(hyperparams):
        for test_row in test_set.values:
            distances = []
            for train_row in train_set.values:
                dist = distance_model(test_row[:-1], train_row[:-1])
                distances.append((dist, train_row[-1]))
            distances.sort()
            neighbors = distances[:k]
            output = [row[-1] for row in neighbors]
            prediction = max(set(output), key=output.count)
            predictions.append(prediction)

        metrics = util.metrics(test_set.iloc[:, -1].to_list(), predictions)
        if metrics[0] > best_acc:
            best_acc = metrics[0]
            best_hyperparams = [k, distance_model]
            best_metrics = metrics
        print(metrics)

    return best_hyperparams, best_metrics

if __name__ == '__main__':
    df = pd.read_csv('balanced_credit.csv')

    folds = util.nfolds(df, 10)

    k_vals = [5, 10, 15, 20]
    models = [euclidean_dist, cosine_dist, manhattan_dist]

    hyperparams = list(itertools.product(k_vals, models))

    result = util.cross_validate(folds, knn, hyperparams)

    print(result)

