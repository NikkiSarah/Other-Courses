#%% Exercise 1
# build a MNIST classifier that achieves over 97% accuracy on the test set
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os
import time
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
X, y = mnist.data, mnist.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7,
                                                    random_state=42, stratify=y)

param_grid = [
    {'n_neighbors': [1, 3, 5],
     'weights': ['uniform', 'distance']}
]

n_cpu = os.cpu_count()
print("Number of CPUs in the system:", n_cpu)

t0 = time.time()
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, scoring="accuracy",
                           n_jobs=n_cpu-2, cv=3)
grid_search.fit(X_train, y_train)
run_time = time.time() - t0

print(run_time)
print(grid_search.best_params_)

cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
grid_search_best_accuracy = grid_search.best_score_
print(grid_search_best_accuracy)

best_estimator = grid_search.best_estimator_
best_estimator.fit(X_train, y_train)
test_accuracy = best_estimator.score(X_test, y_test)
print(test_accuracy)

#%% Exercise 2
# create a function that can shift an MNIST image in any direction by 1 pixel
from scipy.ndimage import shift
import numpy as np

def shift_image(image, vertical_shift, horizontal_shift):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [vertical_shift, horizontal_shift])
    shifted_image = shifted_image.reshape([-1])
    return shifted_image

# create 4 shifted copies (1 for each direction) for each image in the training set and
# add them to the trainig set
X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]

for vertical_shift, horizontal_shift in ((1, 0), (0, 1), (-1, 0), (0, -1)):
    for image, label in zip(X_train_augmented, y_train_augmented):
        X_train_augmented.append(shift_image(image, vertical_shift, horizontal_shift))
        y_train_augmented.append(label)

# train the best model from the previous exercise on the augmented dataset
X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

clf = KNeighborsClassifier(n_neighbours=grid_search.best_params_["n_neighbors"],
                           weights=grid_search.best_params_["weights"])
clf.fit(X_train_augmented, y_train_augmented)

test_accuracy_augmented = clf.score(X_test, y_test)
print(test_accuracy_augmented)
