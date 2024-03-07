#%% Exercise 9
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier

# load the MNIST dataset and split it into a training, validation and test set
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data / 255., mnist.target

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, stratify=y, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, stratify=y_train_val, test_size=10000, random_state=42)

# train a random forest classifier, time how long it takes and evaluate performance on the
# test set
n_cpu = os.cpu_count()
print("Number of CPUs in the system:", n_cpu)

t0 = time.time()
rf_clf = RandomForestClassifier(n_jobs=n_cpu-2, random_state=42)
rf_clf.fit(X_train, y_train)
rf_time = time.time() - t0
print(rf_time)
# 11.77
print(rf_clf.score(X_test, y_test))
# 0.9677

# use PCA to reduce the dataset's dimensionality, with an explained variance ratio of 95%
pca = PCA(n_components=0.95, svd_solver='full', random_state=42)
X_train_reduced = pca.fit_transform(X_train, y_train)

# train a new random forest classifier on the reduced dataset, time how long it takes. How
# does performance compare to the previous classifier
t0 = time.time()
rf_clf2 = RandomForestClassifier(n_jobs=n_cpu-2, random_state=42)
rf_clf2.fit(X_train_reduced, y_train)
rf2_time = time.time() - t0
print(rf2_time)
# 47.15
X_test_reduced = pca.transform(X_test)
print(rf_clf2.score(X_test_reduced, y_test))
# 0.9424

# repeat the exercise with a SGDClassifier
t0 = time.time()
sgd_clf = SGDClassifier(n_jobs=n_cpu-2, random_state=42)
sgd_clf.fit(X_train, y_train)
sgd_time = time.time() - t0
print(sgd_time)
# 6.06
print(sgd_clf.score(X_test, y_test))
# 0.9045

pca = PCA(n_components=0.95, svd_solver='full', random_state=42)
X_train_reduced = pca.fit_transform(X_train, y_train)

t0 = time.time()
sgd_clf2 = SGDClassifier(n_jobs=n_cpu-2, random_state=42)
sgd_clf2.fit(X_train_reduced, y_train)
sgd2_time = time.time() - t0
print(sgd2_time)
# 1.82
X_test_reduced = pca.transform(X_test)
print(sgd_clf2.score(X_test_reduced, y_test))
# 0.9069

#%% Exercise 10
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS
import matplotlib.pyplot as plt
import numpy as np

# use t-sne to reduce the first 5,000 images from MNIST down to 2 dimensions and plot the
# result
X_sub = X[:5000]
y_sub = y[:5000]
tsne = TSNE(random_state=42, n_jobs=n_cpu-2)
X_tsne = tsne.fit_transform(X_sub)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sub.astype(np.int8), cmap="jet", alpha=0.5)
plt.axis("off")
plt.colorbar()

# repeat with other algorithms like PCA, LLE and MDS
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_sub)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_sub.astype(np.int8), cmap="jet", alpha=0.5)
plt.axis("off")
plt.colorbar()


lle = LocallyLinearEmbedding(random_state=42, n_jobs=n_cpu-2)
X_lle = lle.fit_transform(X_sub)

plt.scatter(X_lle[:, 0], X_lle[:, 1], c=y_sub.astype(np.int8), cmap="jet", alpha=0.5)
plt.axis("off")
plt.colorbar()


mds = MDS(n_jobs=n_cpu-2, random_state=42, normalized_stress='auto')
X_mds = mds.fit_transform(X_sub)

plt.scatter(X_mds[:, 0], X_mds[:, 1], c=y_sub.astype(np.int8), cmap="jet", alpha=0.5)
plt.axis("off")
plt.colorbar()
