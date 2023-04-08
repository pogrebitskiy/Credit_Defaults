import pandas as pd
import math
import numpy as np
from utilities import utilities as util
import itertools
from tqdm import tqdm
from collections import Counter
from sklearn.preprocessing import scale
import time

def euclidean_dist(point1, point2):
    '''Calculate the euclidean distance between two points'''
    squared_diff = np.subtract(point1, point2) ** 2
    distance = math.sqrt(np.sum(squared_diff))
    return distance

def manhattan_dist(point1, point2):
    """Calculate the Manhattan distance between two points."""
    distance = np.sum(np.abs(np.subtract(point1, point2)))
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
    k, distance_func = hyperparams
    predictions = []
    for test_row in test_set.values:
        distances = [distance_func(test_row[:-1], train_row[:-1]) for train_row in train_set.values]
        sorted_indices = np.argsort(distances)
        neighbors = sorted_indices[:k]
        output = [train_set.iloc[neighbor][-1] for neighbor in neighbors]
        prediction = Counter(output).most_common(1)[0][0]
        predictions.append(prediction)

    metrics = util.metrics(test_set.iloc[:, -1].to_list(), predictions)
    print(metrics)

    return metrics

if __name__ == '__main__':
    start_time = time.time()
    df = pd.read_csv('balanced_credit.csv')
    feature_cols = df.columns[:-1]
    df[feature_cols] = scale(df[feature_cols], with_mean=True, with_std=True)

    folds = util.nfolds(df, 10)

    k_vals = [5, 7, 11, 15, 19, 23]
    models = [euclidean_dist, cosine_dist, manhattan_dist]

    hyperparam_lst = list(itertools.product(k_vals, models))

    best_hyperparams = None
    best_metrics = None
    best_acc = 0.0

    results_df = pd.DataFrame(columns=['K', 'Model', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1_score'])

    for hyperparams in tqdm(hyperparam_lst):
        result = util.cross_validate(folds, knn, hyperparams)
        df_row = [hyperparams[0], hyperparams[1], result[0], result[1], result[2], result[3], result[4]]
        results_df.loc[len(results_df)] = df_row
        if result[0] > best_acc:
            best_metrics = result
            best_hyperparams = hyperparams
            best_acc = result[0]

    results_df.to_csv('knn_results.csv', index=False)
    print(results_df.to_string())
    print(best_metrics)
    print(best_hyperparams)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")




