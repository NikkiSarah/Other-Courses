#%% Exercise 1

# try a SVM with various hyperparameters such as a linear kernel with various values for
# C or a rbf kernel with various values for C and gamma. How does the best predictor
# perform?
import pandas as pd
from sklearn.model_selection import train_test_split
# data processing
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
# cluster similarity class
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
# model training
import os
import time
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

# load and split the data
housing = pd.read_csv("./datasets/housing.csv")
housing.info()

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

housing_data = housing.drop('median_house_value', axis=1)
housing_labels = housing.median_house_value.copy()

X_train, X_test, y_train, y_test = train_test_split(
    housing_data, housing_labels, test_size=0.2, stratify=housing.income_cat,
    random_state=42)
X_train.drop("income_cat", inplace=True, axis=1)
X_test.drop("income_cat", inplace=True, axis=1)

# clean and process the data
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self 

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())
cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"),
                             OneHotEncoder(handle_unknown="ignore"))
preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)

# X_train_prepared = preprocessing.fit_transform(X_train)
# print(X_train_prepared.shape)
# preprocessing.get_feature_names_out()

# train a SVM using a data sample and 3-fold grid search
n_cpu = os.cpu_count()
print("Number of CPUs in the system:", n_cpu)

X_train_sub = X_train[:5000]
y_train_sub = y_train[:5000]

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("svr", SVR()),
    ])
param_grid = [
    {"svr__kernel": ["linear"], "svr__C": [1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5]},
    {"svr__kernel": ["rbf"], "svr__C": [1e3, 3e3, 1e4, 3e4, 1e5, 3e5],
     "svr__gamma": ['scale', 'auto', 0.01, 0.03, 0.1, 0.3, 1., 3.]}
    ]
t0 = time.time()
grid_search = GridSearchCV(full_pipeline, param_grid,
                           scoring="neg_root_mean_squared_error", n_jobs=n_cpu-2, cv=3)
grid_search.fit(X_train_sub, y_train_sub)
run_time = time.time() - t0

print(run_time)
print(grid_search.best_params_)

cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
grid_search_best_rmse = -grid_search.best_score_
print(grid_search_best_rmse)

#%% Exercise 2
# try replacing grid search with random search
t0 = time.time()
random_search = RandomizedSearchCV(full_pipeline, param_grid, n_iter=50,
                                   scoring="neg_root_mean_squared_error", n_jobs=n_cpu-2,
                                   cv=3, random_state=42)
random_search.fit(X_train_sub, y_train_sub)
run_time = time.time() - t0

print(run_time)
print(random_search.best_params_)

cv_res = pd.DataFrame(random_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
random_search_best_rmse = -random_search.best_score_
print(random_search_best_rmse)

## Exercise 3: add a SelectFromModel transformer to the transformation pipeline to select
## only the most important attributes
selector_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("selector", SelectFromModel(RandomForestRegressor(random_state=42))),
        ("svr", SVR(C=random_search.best_params_["svr__C"],
                    gamma=random_search.best_params_["svr__gamma"],
                    kernel=random_search.best_params_["svr__kernel"])),
        ])
param_grid = [
    {"selector__threshold": ["median", "mean", 1e-5, 1e-3],
     "selector__max_features": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 24]}
    ]

t0 = time.time()
grid_search = GridSearchCV(selector_pipeline, param_grid,
                           scoring="neg_root_mean_squared_error", n_jobs=n_cpu-2, cv=3)
grid_search.fit(X_train_sub, y_train_sub)
run_time = time.time() - t0

print(run_time)
print(grid_search.best_params_)

cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
grid_search_best_rmse = -grid_search.best_score_
print(grid_search_best_rmse)
