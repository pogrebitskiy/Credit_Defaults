import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('balanced_credit.csv')

# features and target
X = df.drop('default', axis=1)
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# hyper param grid
param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan', 'cosine']}

# create class
knn = KNeighborsClassifier()

# grid search cving
grid = GridSearchCV(knn, param_grid, cv=5, verbose=1)
grid.fit(X_train, y_train)

#best hyperparameters and score
best_params = grid.best_params_
best_score = grid.best_score_
print(best_params)

# use best hyperparams
knn = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'])
knn.fit(X_train, y_train)

# report accuracy
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

results = pd.DataFrame(grid.cv_results_)
results.to_csv('knn_sklearn.csv')
