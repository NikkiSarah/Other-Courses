#%% Exercise 8
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              VotingClassifier)
from sklearn.svm import SVC
import numpy as np

# load the MNIST dataset and split it into a training, validation and test set
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data / 255., mnist.target

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, stratify=y, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, stratify=y_train_val, test_size=10000, random_state=42)

# train a random forest, extra-trees and SVM classifier
n_cpu = os.cpu_count()
print("Number of CPUs in the system:", n_cpu)

rf_clf = RandomForestClassifier(n_jobs=n_cpu-2, random_state=42)
et_clf = ExtraTreesClassifier(n_jobs=n_cpu-2, random_state=42)
svm_clf = SVC(probability=True, random_state=42)

classifiers = [rf_clf, et_clf, svm_clf]
for clf in classifiers:
    clf.fit(X_train, y_train)
print([clf.score(X_val, y_val) for clf in classifiers])
# [0.9686, 0.9704, 0.9783]

# combine them into an ensemble classifier that outperforms every individual classifier on
# the validation set, using soft or hard voting
named_clf = [
    ("random_forest_clf", rf_clf),
    ("extra_trees_clf", et_clf),
    ("svm_clf", svm_clf),
    ]
# - hard voting
hard_voting_clf = VotingClassifier(named_clf, voting="hard", n_jobs=n_cpu-2,
                                   verbose=True)
hard_voting_clf.fit(X_train, y_train)
print(hard_voting_clf.score(X_val, y_val))
# 0.9736

y_val_enc = y_val.astype(np.int64)
print([clf.score(X_val, y_val_enc) for clf in hard_voting_clf.estimators_])
# [0.9686, 0.9704, 0.9783]

hard_voting_clf.set_params(random_forest_clf="drop")
rf_clf_trained = hard_voting_clf.named_estimators_.pop("random_forest_clf")
hard_voting_clf.estimators_.remove(rf_clf_trained)
print(hard_voting_clf.score(X_val, y_val))
# 0.9755

# - soft voting
soft_voting_clf = VotingClassifier(named_clf, voting="soft", n_jobs=n_cpu-2,
                                   verbose=True)
soft_voting_clf.fit(X_train, y_train)
print(soft_voting_clf.score(X_val, y_val))
# 0.9785

print([clf.score(X_val, y_val_enc) for clf in soft_voting_clf.estimators_])
# [0.9686, 0.9704, 0.9783]

soft_voting_clf.set_params(random_forest_clf="drop")
rf_clf_trained = soft_voting_clf.named_estimators_.pop("random_forest_clf")
soft_voting_clf.estimators_.remove(rf_clf_trained)
print(soft_voting_clf.score(X_val, y_val))
# 0.9789

# assess performance on the test set. How much better does it perform compared to the
# individual classifiers?
print([clf.score(X_test, y_test) for clf in classifiers])
# [0.9686, 0.9718, 0.977]
print(hard_voting_clf.score(X_test, y_test))
# 0.9729
print(soft_voting_clf.score(X_test, y_test))
# 0.9779

#%% Exercise 9
from sklearn.ensemble import StackingClassifier
import time
# run the individual classifiers to make predictions on the validation set and create a
# new training set with the predictions: each training instance is a vector containing the
# set of predictions from all the classifiers for an image, and the target is the image's
# class
X_val_preds = np.empty((len(X_val), len(classifiers)), dtype=object)
for idx, clf in enumerate(classifiers):
    X_val_preds[:, idx] = clf.predict(X_val)

# train a classifier on this new training set
svm_blender = SVC(probability=True, random_state=42)
svm_blender.fit(X_val_preds, y_val)
print(svm_blender.score(X_val_preds, y_val))
# 0.971

# evaluate the ensemble on the test set: for each image in the test set, make predictions
# with all the classifiers and feed them into the blender to get the ensemble's
# predictions. How does it compare to the voting classifier?
X_test_preds = np.empty((len(X_test), len(classifiers)), dtype=object)
for idx, clf in enumerate(classifiers):
    X_test_preds[:, idx] = clf.predict(X_test)
print(svm_blender.score(X_test_preds, y_test))
# 0.97

# try again using a StackingClassifer. Do you get better performance?
t0 = time.time()
stacking_clf = StackingClassifier(named_clf, final_estimator=svm_blender, cv=3,
                                  n_jobs=n_cpu-2, verbose=2)
stacking_clf.fit(X_train_val, y_train_val)
run_time = time.time() - t0
print(stacking_clf.score(X_test, y_test))
# 0.9797