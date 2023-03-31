import pandas as pd

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


def cross_validate(folds, model):
    '''To cross validate a model using a given set of folds'''
    results = []

    for i in range(len(folds)):
        # Use all folds except the i-th fold as the training set
        train_folds = folds[:i] + folds[i + 1:]
        train_df = pd.concat(train_folds)
        # Use the i-th fold as the validation set
        val_df = folds[i].copy()
        metrics = model(train_df, val_df)
        results.append(metrics)

    avg_results = []
    # Find the mean of the model results across the folds
    for i in range(len(results[0])):
        sum_values = 0
        for sub_results in results:
            sum_values += sub_results[i]
        mean = sum_values / len(results)
        avg_results.append(mean)

    return avg_results



def metrics(y, y_pred):
    '''Find the accuracy, sensitivity, specificity, precision and f1 score for the model'''
    total = len(y)
    # Find the number of correct predictions and the accuracy
    correct = (y == y_pred).sum()
    accuracy = correct / total

    # Find the true positive, false negative and sensitivity
    true_positive = ((y == 1) & (y_pred == 1)).sum()
    false_negative = ((y == 1) & (y_pred == 0)).sum()
    sensitivity = true_positive / (true_positive + false_negative)

    # Find the true negative, false positive and specificity
    true_negative = ((y == 0) & (y_pred == 0)).sum()
    false_positive = ((y == 0) & (y_pred == 1)).sum()
    specificity = true_negative / (true_negative + false_positive)

    # Find the precision
    precision = true_positive / (true_positive + false_positive)

    # Calculate the f1 score
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

    return accuracy, sensitivity, specificity, precision, f1_score