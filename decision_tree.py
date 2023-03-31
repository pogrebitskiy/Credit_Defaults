"""
Decision Tree
"""
from collections import Counter
import numpy as np
import math
import pandas as pd
import pickle
from itertools import product
from tqdm import tqdm


def total(cnt):
    """Sum the values in a dict"""
    return sum(cnt.values())


def gini(cnt):
    """1 - sum(p^2)"""
    tot = total(cnt)
    return 1 - sum([(v / tot) ** 2 for v in cnt.values()])


def entropy(cnt):
    """Calculate entropy"""
    tot = total(cnt)
    return -sum([(v / tot) * math.log2(v / tot) for v in cnt.values()])


def get_class_counts(data):
    # Returns a dictionary with the counts of each class in the dataset
    return Counter(data.loc[:, 'default'])


def get_impurity(data, criterion):
    # Calculates the impurity of the data using the gGiven criterion
    counts = get_class_counts(data)
    return criterion(counts)


def wavg(cnt1, cnt2, measure):
    """Calculate the weighted average of the classes"""
    tot1, tot2 = total(cnt1), total(cnt2)
    tot = tot1 + tot2
    return (measure(cnt1) * tot1 + measure(cnt2) * tot2) / tot


def evaluate_split(df, class_col, split_col, feature_val, measure):
    """Evaluate the split location"""
    df1, df2 = df[df[split_col] < feature_val], df[df[split_col] >= feature_val]
    cnt1, cnt2 = Counter(df1[class_col]), Counter(df2[class_col])
    return wavg(cnt1, cnt2, measure)


def best_split_for_column(df, class_col, split_col, measure):
    best_v = ''
    best_meas = float('inf')
    for v in set(df[split_col]):
        meas = evaluate_split(df, class_col, split_col, v, measure)
        if meas < best_meas:
            best_v = v
            best_meas = meas
    return best_v, best_meas


def best_split(df, class_col, measure):
    best_col = None
    best_v = None
    best_meas = float("inf")

    # Iterate through columns to find where to split the tree
    for split_col in tqdm(df.columns):
        if split_col != class_col:
            v, meas = best_split_for_column(df, class_col, split_col, measure)
            if meas < best_meas:
                best_v = v
                best_meas = meas
                best_col = split_col

    return best_col, best_v, best_meas


def split(df, best_col, best_v):
    # Split the dataframe based on the split val in a column
    df1, df2 = df[df[best_col] < best_v], df[df[best_col] >= best_v]

    return df1, df2


def build_tree(data, depth, criterion, max_depth, min_instances, target_impurity):
    # Get impurity of node
    impurity_score = get_impurity(data, criterion)

    # Get majority class
    majority_class = get_class_counts(data).most_common(1)[0][0]

    # Case to terminate node if max depth or impurity target is reached
    if (max_depth is not None and depth >= max_depth) or (impurity_score <= target_impurity):
        return (None, None, len(data), majority_class, impurity_score, depth, None, None)
    # Stop splitting if we have too few instances
    elif len(data) <= min_instances:
        return (None, None, len(data), majority_class, impurity_score, depth, None, None)

    # Split the data
    best_col, best_value, best_meas = best_split(data, 'default', criterion)

    # Case to terminate node if there is no best_col
    if best_col is None:
        return (None, None, len(data), majority_class, impurity_score, depth, None, None)

    # Split into left and right data
    left_data, right_data = split(data, best_col, best_value)

    # check if no further split could be made
    if len(left_data) == 0 or len(right_data) == 0:
        return (None, None, len(data), majority_class, impurity_score, depth, None, None)

    # Recursively build the left and right subtrees
    return (
        best_col, best_value, len(data), majority_class, impurity_score, depth,
        build_tree(left_data, depth + 1, criterion, max_depth, min_instances, target_impurity),
        build_tree(right_data, depth + 1, criterion, max_depth, min_instances, target_impurity))


def dtree(train, criterion, max_depth=None, min_instances=2, target_impurity=0.0):
    """Build the decision tree recursively"""
    tree = build_tree(train, 0, criterion, max_depth, min_instances, target_impurity)
    return tree


def predict_one(model, song):
    """Recursively walk the tree to make a prediction about a specified song"""
    left_tree, right_tree = model[6], model[7]
    majority = model[3]
    col = model[0]
    val = model[1]
    # Base case when you reach a leaf node
    if left_tree == None and right_tree == None:
        return majority

    # Go left if less than val
    elif song[col] < val:
        return predict_one(left_tree, song)

    # Else go right
    else:
        return predict_one(right_tree, song)


def predict(model, data):
    """Predict the genre of every song in a dataframe by applying the predict_one function"""
    return data.apply(lambda x: predict_one(model, x), axis=1)


def accuracy(y, y_pred):
    """Calculate accuracy given y_true and y_pred"""
    correct = np.sum(y == y_pred)
    total = len(y)
    acc = correct / total
    return acc


def cross_validate(data, n_splits, lst_of_hyperparams):
    """Function to iterate through combinations of hyperparams and cross validate their accuracies"""
    # Shuffle the dataset
    df = data.sample(frac=1).reset_index(drop=True)

    # Define the number of folds
    k = n_splits
    # Determine the number of samples per fold
    fold_size = len(df) // k
    folds = [df.iloc[i:i + fold_size] for i in range(0, len(df), fold_size)]

    best_hyperparams = None
    best_accuracy = None

    # Iterate through each combination of hyperparams
    for comb in tqdm(lst_of_hyperparams):
        accuracies = []
        eval_func, max_d, min_e, target_imp = comb
        print(comb)
        for i in range(len(folds)):
            # Use all folds except the i-th fold as the training set
            train_folds = folds[:i] + folds[i + 1:]
            train_df = pd.concat(train_folds)

            test_df = folds[i]

            # Build tree using specified hyperparams
            model = dtree(train_df, eval_func, max_d, min_e, target_imp)

            # Make predictions and append accuracy
            y_pred = predict(model, test_df).to_list()
            accuracies.append(accuracy(test_df.loc[:, 'default'], y_pred))

        print(f'Combination Accuracy: {np.mean(accuracies)}')
        if not best_accuracy or np.mean(accuracies) > best_accuracy:
            best_accuracy = np.mean(accuracies)
            best_hyperparams = comb
        print()
        print(f'Best Params: {best_hyperparams}')
        print(f'Best Accuracy: {best_accuracy}')

    return best_hyperparams, best_accuracy