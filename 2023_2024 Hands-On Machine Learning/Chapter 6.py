#%% Exercise 7
# train and fine-tune a decision tree for the moons dataset
import os
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import optuna
from sklearn.metrics import accuracy_score
from optuna_dashboard import run_server

n_cpu = os.cpu_count()
print("Number of CPUs in the system:", n_cpu)

# use_make_moons() to generate a moons dataset
X, y = make_moons(n_samples=10000, noise=0.4)

# use train_test_split to split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# use grid search with cross-validation to find good hyperparameter values for a
# decision tree classifier
dt_clf = DecisionTreeClassifier(random_state=42)
param_distributions = {
    "max_leaf_nodes": optuna.distributions.IntDistribution(2, 100),
    "max_depth": optuna.distributions.IntDistribution(1, 7),
    "min_samples_split":  optuna.distributions.IntDistribution(2, 4)
}
storage = optuna.storages.InMemoryStorage()
cv_study = optuna.create_study(storage=storage, study_name="Ch6_moons_cv",
                               direction="maximize")
optuna_search = optuna.integration.OptunaSearchCV(
    dt_clf, param_distributions, cv=3, n_jobs=n_cpu-2, n_trials=50, random_state=42,
    refit=True, scoring='accuracy', study=cv_study, verbose=2)
optuna_search.fit(X_train, y_train)

run_server(storage)

print("Best trial:")
trial = optuna_search.study_.best_trial

print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# train the best model on the full training set and assess performance on the test set
# (you should get about 85% to 87% accuracy)
best_dt = optuna_search.best_estimator_
print("Best estimator:", best_dt)
print(best_dt.score(X_train, y_train))
print(best_dt.score(X_test, y_test))

#%% Exercise 8
# grow a forest
from sklearn.model_selection import ShuffleSplit
from sklearn.base import clone
import numpy as np
from scipy.stats import mode

# generate 1,000 subsets of the training set, each containing 100 instances selected
# randomly
num_trees = 1000
num_samples = 100
ss = ShuffleSplit(n_splits=num_trees, test_size=len(X_train) - num_samples,
                  random_state=42)
mini_train = []
for train_idx, test_idx in ss.split(X_train):
    mini_X_train = X_train[train_idx]
    mini_y_train = y_train[train_idx]
    mini_train.append((mini_X_train, mini_y_train))

# train a single decision tree on each subset, using the best hyperparameters found
# previously. Evaluate these 1,000 trees on the test set
forest = [clone(best_dt) for _ in range(num_trees)]

accuracy_scores = []
for tree, (mini_X_train, mini_y_train) in zip(forest, mini_train):
    tree.fit(mini_X_train, mini_y_train)

    y_pred = tree.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(test_accuracy)
print(np.mean(accuracy_scores))

# for each test set instance, generate the predictions of the 1,000 trees and keep only
# the most frequent prediction
Y_pred = np.empty([num_trees, len(X_test)], dtype=np.uint8)

for idx, tree in enumerate(forest):
    Y_pred[idx] = tree.predict(X_test)
majority_vote_preds = np.array(mode(Y_pred, axis=0))[0]

# evaluate these predictions on the test set
forest_accuracy = accuracy_score(y_test, majority_vote_preds.reshape([-1]))
print(forest_accuracy)
