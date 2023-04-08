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
    # Find the squared differences between the points
    squared_diff = np.subtract(point1, point2) ** 2
    # Get the square root of the sum of all distances
    distance = math.sqrt(np.sum(squared_diff))
    return distance

def manhattan_dist(point1, point2):
    """Calculate the Manhattan distance between two points."""
    # Calculate Manhattan distance
    distance = np.sum(np.abs(np.subtract(point1, point2)))
    return distance

def cosine_dist(point1, point2):
    """
    Calculate the cosine distance between two vectors.
    """
    # Get the dot product between the points
    dot_product = np.dot(point1, point2)
    # Calculate Euclidean norm of the two points
    norm1 = np.linalg.norm(point1)
    norm2 = np.linalg.norm(point2)
    # Calculate Cosine distance
    return 1 - (dot_product / (norm1 * norm2))


def knn(train_set, test_set, hyperparams):
    '''To perform K-Nearest Neighbors on the train and test sets'''
    # Get the hyperparams from the tuple
    k, distance_func = hyperparams
    predictions = []
    # Iterate over testing set
    for test_row in test_set.values:
        # Find the distance to each point in training set using the given model
        distances = [distance_func(test_row[:-1], train_row[:-1]) for train_row in train_set.values]
        # Sort the distances by ascending values
        sorted_indices = np.argsort(distances)
        # Select the k nearest neighbors
        neighbors = sorted_indices[:k]
        # Get the target variables of the nearest neighbors
        output = [train_set.iloc[neighbor][-1] for neighbor in neighbors]
        # Use the most frequent target variable as the prediction
        prediction = Counter(output).most_common(1)[0][0]
        predictions.append(prediction)

    # Calculate the metrics of the knn algorithm
    metrics = util.metrics(test_set.iloc[:, -1].to_list(), predictions)

    return metrics

if __name__ == '__main__':
    df = pd.read_csv('balanced_credit.csv')
    # Scale the feature columns by mean and std dev
    feature_cols = df.columns[:-1]
    df[feature_cols] = scale(df[feature_cols], with_mean=True, with_std=True)

    # Split the data in ten folds
    folds = util.nfolds(df, 10)

    # select hyperparameters
    k_vals = [5, 7, 11, 15, 19, 23]
    models = [euclidean_dist, cosine_dist, manhattan_dist]

    # Create all possible combinations of hyperparams
    hyperparam_lst = list(itertools.product(k_vals, models))

    best_hyperparams = None
    best_metrics = None
    best_acc = 0.0

    results_df = pd.DataFrame(columns=['K', 'Model', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1_score'])

    # Iterate over hyperparameter combinations
    for hyperparams in tqdm(hyperparam_lst):
        # Cross validate using knn algorithm
        result = util.cross_validate(folds, knn, hyperparams)
        # Store result in df
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




