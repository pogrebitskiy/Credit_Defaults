import pandas as pd
import math
import numpy as np
from utilities import utilities as util
import itertools
from tqdm import tqdm
import pickle

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
    k, distance_model = hyperparams
    predictions = []
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

    print(metrics)

    return metrics

if __name__ == '__main__':
    df = pd.read_csv('balanced_credit.csv')

    folds = util.nfolds(df, 10)

    k_vals = [5, 8, 10, 12, 15]
    models = [euclidean_dist, cosine_dist, manhattan_dist]

    hyperparam_lst = list(itertools.product(k_vals, models))
    print(hyperparam_lst)

    best_hyperparams = None
    best_acc = 0.0
    best_metrics = None

    results_df = pd.DataFrame(columns=['K', 'Model', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1_score'])

    for hyperparams in tqdm(hyperparam_lst):
        result = util.cross_validate(folds, knn, hyperparams)
        df_row = [hyperparams[0], hyperparams[1], result[0], result[1], result[2], result[3], result[4]]
        results_df.loc[len(results_df)] = df_row
        if result[0] > best_acc:
            best_metrics = result
            best_hyperparams = hyperparams

    results_df.to_csv('df.csv', index=False)
    print(results_df)




