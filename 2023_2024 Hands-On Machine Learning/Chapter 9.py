#%% Exercise 10
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
# load the olivetti faces dataset and split it into a training, validation and test set,
# using stratified sampling
faces, labels = fetch_olivetti_faces(return_X_y=True)

ss = StratifiedShuffleSplit(n_splits=1, test_size=40, random_state=42)
train_val_idx, test_idx = next(ss.split(faces, labels))
X_train_val = faces[train_val_idx]
y_train_val = labels[train_val_idx]
X_test = faces[test_idx]
y_test = labels[test_idx]

ss = StratifiedShuffleSplit(n_splits=1, test_size=80, random_state=43)
train_idx, val_idx = next(ss.split(X_train_val, y_train_val))
X_train = X_train_val[train_idx]
y_train = y_train_val[train_idx]
X_val = X_train_val[val_idx]
y_val = y_train_val[val_idx]

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

# cluster the images with k-means, ensuring there are a good number of clusters
pca = PCA(n_components=0.99, svd_solver='full', random_state=42)
X_train_reduced = pca.fit_transform(X_train)
X_val_reduced = pca.transform(X_val)
X_test_reduced = pca.transform(X_test)

k_range = range(5, 150, 5)
kmeans_per_k = []
for k in k_range:
    print(f"k={k}")
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(X_train_reduced)
    kmeans_per_k.append(kmeans)

silhouette_scores = [silhouette_score(X_train_reduced, model.labels_) for model in
                     kmeans_per_k]
best_idx = np.argmax(silhouette_scores)
best_k = k_range[best_idx]
print(best_k)
best_score = silhouette_scores[best_idx]
print(best_score)

plt.plot(k_range, silhouette_scores, "bo-")
plt.xlabel("$k$")
plt.ylabel("Silhouette score")
plt.plot(best_k, best_score, "rs")

inertias = [model.inertia_ for model in kmeans_per_k]
best_inertia = inertias[best_idx]
print(best_inertia)

plt.plot(k_range, inertias, "bo-")
plt.xlabel("$k$")
plt.ylabel("Inertia")
plt.plot(best_k, best_inertia, "rs")

# visualise the clusters
best_model = kmeans_per_k[best_idx]

def plot_faces(faces, labels, n_cols=5):
    faces = faces.reshape(-1, 64, 64)
    n_rows = (len(faces) - 1) // n_cols + 1
    for idx, (face, label) in enumerate(zip(faces, labels)):
        plt.subplot(n_rows, n_cols, idx+1)
        plt.imshow(face, cmap="grey")
        plt.axis("off")
        plt.title(label)

for clust_id in np.unique(best_model.labels_)[:5]:
    print("Cluster", clust_id)
    in_clust = best_model.labels_ == clust_id
    faces = X_train[in_clust]
    labels = X_train[in_clust]
    plot_faces(faces, labels)


#%% Exercise 11
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd
# train a classifier to predict which person is represented in each picture and evaluate
# it on the validation set
n_cpu = os.cpu_count()
print("Number of CPUs in the system:", n_cpu)

rf_clf = RandomForestClassifier(oob_score=True, n_jobs=n_cpu-2, random_state=True)
rf_clf.fit(X_train, y_train)
print(rf_clf.score(X_val, y_val))
# 0.95

# use k-means as a dimensionality reduction tool and train a new classifier. Search for
# the number of clusters that achieves the best performance
clf = make_pipeline(KMeans(n_init='auto', random_state=42),
                    RandomForestClassifier(oob_score=True, n_jobs=n_cpu-2,
                                           random_state=42))
param_distrib = {"kmeans__n_clusters": np.arange(5, 150, 5)}
grid_search = GridSearchCV(clf, param_distrib, cv=3)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.score(X_val, y_val))
# 0.85

# append the features from the reduced set to the original features and repeat the
# exercise
best_model = KMeans(n_clusters=115, n_init='auto', random_state=42)
X_train_reduced = best_model.fit_transform(X_train)
X_val_reduced = best_model.transform(X_val)
X_train_augmented = np.c_[X_train, X_train_reduced]
X_val_augmented = np.c_[X_val, X_val_reduced]

rf_clf = RandomForestClassifier(oob_score=True, n_jobs=n_cpu-2, random_state=True)
rf_clf.fit(X_train_augmented, y_train)
print(rf_clf.score(X_val_augmented, y_val))
# 0.95

#%% Exercise 12
from sklearn.mixture import GaussianMixture
# reduce the datasets dimensionality, preserving 99% of the variance, and then train a
# Gaussian mixture model
pca = PCA(n_components=0.99, svd_solver='full', random_state=42)
X_train_reduced = pca.fit_transform(X_train)
X_val_reduced = pca.transform(X_val)
X_test_reduced = pca.transform(X_test)

gmm = GaussianMixture(n_components=40, random_state=42, verbose=2)
gmm.fit(X_train_reduced, y_train)
y_pred = gmm.predict(X_train_reduced)

# use the model to generate some new faces and visualise them
n_gen_faces = 20
gen_faces_reduced, y_gen_faces = gmm.sample(n_samples=n_gen_faces)
gen_faces = pca.inverse_transform(gen_faces_reduced)

plot_faces(gen_faces, y_gen_faces)

# modify some images and see if the model can detect the anomalies
n_rotated = 4
rotated = np.transpose(X_train[:n_rotated].reshape(-1, 64, 64), axes=[0, 2, 1])
rotated = rotated.reshape(-1, 64*64)
y_rotated = y_train[:n_rotated]

n_flipped = 3
flipped = X_train[:n_flipped].reshape(-1, 64, 64)[:, ::-1]
flipped = flipped.reshape(-1, 64*64)
y_flipped = y_train[:n_flipped]

n_darkened = 3
darkened = X_train[:n_darkened].copy()
darkened[:, 1:-1] *= 0.3
y_darkened = y_train[:n_darkened]

X_mod_faces = np.r_[rotated, flipped, darkened]
y_mod = np.concatenate([y_rotated, y_flipped, y_darkened])

plot_faces(X_mod_faces, y_mod)

reduced_mod_faces = pca.transform(X_mod_faces)
print(gmm.score_samples(reduced_mod_faces))

print(gmm.score_samples(X_train_reduced[:10]))

#%% Exercise 13
from sklearn.metrics import mean_squared_error
# reduce the dataset's dimensionality, preserving 99% of the variance
pca = PCA(n_components=0.99, svd_solver='full', random_state=42)
X_train_reduced = pca.fit_transform(X_train)
X_val_reduced = pca.transform(X_val)
X_test_reduced = pca.transform(X_test)

# compute the reconstruction error of each image
X_train_recovered = pca.inverse_transform(X_train_reduced)
reconstruct_error = np.square(X_train_recovered - X_train).mean()
reconstruct_error = mean_squared_error(X_train, X_train_recovered)
print(reconstruct_error)

# take some of the modified images from the previous exercise and calculate their
# reconstruction error
X_mod_recovered = pca.inverse_transform(reduced_mod_faces)
reconstruct_error = mean_squared_error(X_mod_faces, X_mod_recovered)
print(reconstruct_error)

# plot a reconstructed image
plot_faces(X_mod_faces, y_mod)
plot_faces(X_mod_recovered, y_mod)
