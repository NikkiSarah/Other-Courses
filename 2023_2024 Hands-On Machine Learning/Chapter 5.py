#%% Exercise 9
# train a linear SVM on a linearly separable dataset
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt

iris = datasets.load_iris(as_frame=True)
X = iris.data.iloc[:, :2]
y = iris.target

setosa_or_versicolour = (y == 0) | (y == 1)
X = X[setosa_or_versicolour]
y = y[setosa_or_versicolour]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

C = 5
alpha = 0.05

linear_clf = LinearSVC(loss='hinge', C=C, dual=True, random_state=42)
linear_clf.fit(X_scaled, y)

# train a SVC and SGDClassifier on the same dataset. See if you can get them to produce
# roughly the same model
svc_clf = SVC(kernel='linear', C=C, tol=1e-4, random_state=42)
svc_clf.fit(X_scaled, y)

sgd_clf = SGDClassifier(alpha=0.05, tol=1e-4, random_state=42)
sgd_clf.fit(X_scaled, y)

def plot_decision_boundary(model):
    w = -model.coef_[0, 0] / model.coef_[0, 1]
    b = -model.intercept_[0] / model.coef_[0, 1]
    boundary = scaler.inverse_transform([[-10, -10 * w + b], [10, 10 * w + b]])
    return boundary

lin_line = plot_decision_boundary(linear_clf)
svc_line = plot_decision_boundary(svc_clf)
sgd_line = plot_decision_boundary(sgd_clf)

plt.plot(lin_line[:, 0], lin_line[:, 1], "k:", label="LinearSVC")
plt.plot(svc_line[:, 0], svc_line[:, 1], "b--", label="SVC")
plt.plot(sgd_line[:, 0], sgd_line[:, 1], "r-", label="SGD")
plt.plot(X.iloc[:, 0][y==0], X.iloc[:, 1][y==0], "bs")
plt.plot(X.iloc[:, 0][y==1], X.iloc[:, 1][y==1], "yo")
plt.axis([3, 8, 0, 5])

#%% Exercise 10
# train a SVM classifier on the wine dataset. Since SVMs are binary classifiers, you
# need to use one-versus-all to classify all 3 classes. What accuracy can you reach?
from sklearn.model_selection import train_test_split
import os
import time
import optuna
from sklearn.metrics import accuracy_score
from optuna_dashboard import run_server

n_cpu = os.cpu_count()
print("Number of CPUs in the system:", n_cpu)

wine = datasets.load_wine(as_frame=True)
X = wine.data
y = wine.target

y.value_counts(normalize=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y,
                                                    random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def objective(trial):
    # define model and hyperparameters
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
    C = trial.suggest_int("C", 1, 10)
    gamma = trial.suggest_float("gamma", 1e-3, 1e-1, log=True)

    # train and evaluate model
    svm_clf = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
    svm_clf.fit(X_train_scaled, y_train)

    # return the evaluation metric
    y_pred = svm_clf.predict(X_train_scaled)
    accuracy = accuracy_score(y_train, y_pred)

    return accuracy

t0 = time.time()
study = optuna.create_study(storage="sqlite:///./outputs/db.sqlite3",
                            study_name="Ch5_wine", direction="maximize")
study.optimize(objective, n_trials=10, n_jobs=n_cpu-2, show_progress_bar=True)
run_time = time.time() - t0

print("Best trial:", study.best_trial.number)
print("Best accuracy:", study.best_trial.value)
print("Best hyperparameters:", study.best_params)

final_svm = SVC(kernel=study.best_params['kernel'], C=study.best_params['C'],
                gamma=study.best_params['gamma'], random_state=42)
final_svm.fit(X_train_scaled, y_train)
print(final_svm.score(X_train_scaled, y_train))
print(final_svm.score(X_test_scaled, y_test))


svm_clf = SVC(random_state=42)
param_distributions = {
    "kernel": optuna.distributions.CategoricalDistribution(['rbf', 'linear']),
    "gamma": optuna.distributions.FloatDistribution(1e-3, 1e-1, log=True),
    "C": optuna.distributions.IntDistribution(1, 10),
}
storage = optuna.storages.InMemoryStorage()
cv_study = optuna.create_study(storage=storage, study_name="Ch5_wine_cv",
                               direction="maximize")
optuna_search = optuna.integration.OptunaSearchCV(
    svm_clf, param_distributions, cv=5, n_jobs=n_cpu-2, n_trials=50, random_state=42,
    refit=True, scoring='accuracy', study=cv_study, verbose=2)
optuna_search.fit(X_train_scaled, y_train)

run_server(storage)

print("Best trial:")
trial = optuna_search.study_.best_trial

print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

best_svm = optuna_search.best_estimator_
print("Best estimator:", best_svm)
print(best_svm.score(X_train_scaled, y_train))
print(best_svm.score(X_test_scaled, y_test))

#%% Exercise 11
# train and fine-tune a SVM regressor on the raw California housing dataset. What is your
# best model's RMSE?
from sklearn.datasets import fetch_california_housing
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_svr = SVR()
param_distributions = {
    "kernel": optuna.distributions.CategoricalDistribution(['rbf', 'linear']),
    "gamma": optuna.distributions.FloatDistribution(1e-3, 1e-1, log=True),
    "C": optuna.distributions.IntDistribution(1, 10),
}
storage = optuna.storages.InMemoryStorage()
cv_study = optuna.create_study(storage=storage, study_name="Ch5_housing_cv",
                               direction="maximize")
optuna_search = optuna.integration.OptunaSearchCV(
    svm_svr, param_distributions, cv=3, n_jobs=n_cpu-2, n_trials=50, random_state=42,
    refit=True, scoring='neg_root_mean_squared_error', study=cv_study, verbose=2)
optuna_search.fit(X_train_scaled, y_train)

run_server(storage)

trial = optuna_search.study_.best_trial
print("Best trial:", trial)

print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

best_svm = optuna_search.best_estimator_
print("Best estimator:", best_svm)

-cross_val_score(best_svm, X_train_scaled, y_train, scoring='neg_root_mean_squared_error')

y_pred = best_svm.predict(X_test_scaled)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(rmse)
