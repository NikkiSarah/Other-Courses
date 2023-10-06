import numpy as np
import pandas as pd
import seaborn as sns
import wget

from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# check the backend and change if required
import matplotlib as mpl

mpl_backend = mpl.get_backend()
if mpl_backend != "qt5agg":
    mpl.use("qt5agg")
else:
    pass

# read in the data
try:
    data = pd.read_csv("Data/CreditScoring.csv")
except:
    url = 'https://github.com/gastonstat/CreditScoring/raw/master/CreditScoring.csv'
    wget.download(url, out="Data")
    data = pd.read_csv("Data/CreditScoring.csv")

# Data Preparation
data.columns = data.columns.str.lower()

# categorical variables are in their numeric form, which will be mapped back to strings using dictionaries
status_dict = {
    1: 'ok',
    2: 'default',
    0: 'unknown'
}

home_dict = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unknown'
}

marital_dict = {
    1: 'single',
    2: 'married',
    3: 'widow',
    4: 'separated',
    5: 'divorced',
    0: 'unknown'
}

records_dict = {
    1: 'no',
    2: 'yes',
    3: 'unknown'
}

job_dict = {
    1: 'fixed',
    2: 'part_time',
    3: 'freelance',
    4: 'other',
    0: 'unknown'
}

dicts = [status_dict, home_dict, marital_dict, records_dict, job_dict]
vars = ['status', 'home', 'marital', 'records', 'job']
for v, d in zip(vars, dicts):
    data[v] = data[v].map(d)

# replace 99999999 with NaN
for var in ['income', 'assets', 'debt']:
    data[var] = data[var].replace(to_replace=99999999, value=np.nan)

# remove the single record with an unknown status
data = data[data.status != 'unknown']

# replace missing values in income, assets and debt with zeros
data = data.fillna(0)

# split the data into a training, validation and test set using a 60%/20%/20% split
X = data.drop('status', axis=1)
y = data.status

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=22)

# one-hot-encode the categorical variables
# (most ML models can't handle non-numeric features)
cat_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']

transformer = make_column_transformer(
    (OneHotEncoder(), cat_cols),
    remainder='passthrough', verbose_feature_names_out=False)
transformer.fit(X_train)
transformed_train = transformer.transform(X_train)
X_train_enc = pd.DataFrame(transformed_train, columns=transformer.get_feature_names_out())
transformed_val = transformer.transform(X_val)
X_val_enc = pd.DataFrame(transformed_val, columns=transformer.get_feature_names_out())

# Data Modelling
# train a random forest model with largely default parameters
rf_model = RandomForestClassifier(n_estimators=10)
rf_model.fit(X_train_enc, y_train)
y_pred = rf_model.predict_proba(X_val_enc)[:, 1]
roc_auc_score(y_val, y_pred)
# notice how if we run the model again, the accuracy changes
rf_model = RandomForestClassifier(n_estimators=10)
rf_model.fit(X_train_enc, y_train)
y_pred = rf_model.predict_proba(X_val_enc)[:, 1]
roc_auc_score(y_val, y_pred)

# train the model 100 times to understand how much the accuracy changes
aucs = []
for i in range(100):
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train_enc, y_train)
    y_pred = rf_model.predict_proba(X_val_enc)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    aucs.append(auc)

auc_mean = np.mean(aucs)
auc_std = np.std(aucs)
print('Mean: {mean:.4f}, StdDev: {std:.4f}'.format(mean=auc_mean, std=auc_std))

# set the seed to stop results changes with repeated training
rf_model = RandomForestClassifier(n_estimators=100, random_state=22)
rf_model.fit(X_train_enc, y_train)
y_pred = rf_model.predict_proba(X_val_enc)[:, 1]
roc_auc_score(y_val, y_pred)

# investigate how the number of trees affects performance
scores = []
for n_est in range(10, 201, 10):
    rf_model = RandomForestClassifier(n_estimators=n_est, random_state=22)
    rf_model.fit(X_train_enc, y_train)
    y_pred = rf_model.predict_proba(X_val_enc)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    scores.append((n_est, auc))
    print('n_estimators: {n_est} -> {auc:.4f}'.format(n_est=n_est, auc=auc))
# notice how the optimal number of trees is about 140. After that accuracy starts to decline.

scores_df = pd.DataFrame(scores, columns=['n_estimators', 'auc'])
sns.lineplot(data=scores_df, x='n_estimators', y='auc')
plt.xlabel("n_estimators")
plt.ylabel("auc")
plt.title("AUC scores for different numbers of trees")

# Hyperparameter Tuning
# observe the impact of changing max_depth
all_aucs = {}
for depth in [5, 10, 15, 20]:
    print('depth: {depth}'.format(depth=depth))

    scores = []
    for n_est in range(10, 201, 10):
        rf_model = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=22)
        rf_model.fit(X_train_enc, y_train)
        y_pred = rf_model.predict_proba(X_val_enc)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        scores.append((n_est, auc))
        print('n_estimators: {n_est} -> {auc:.4f}'.format(n_est=n_est, auc=auc))
    all_aucs[depth] = scores

# plot the results
scores_df = pd.DataFrame(all_aucs)
scores_df[['n_est', 'depth=5']] = pd.DataFrame(scores_df[5].tolist())
scores_df[['n_est2', 'depth=10']] = pd.DataFrame(scores_df[10].tolist())
scores_df[['n_est3', 'depth=15']] = pd.DataFrame(scores_df[15].tolist())
scores_df[['n_est4', 'depth=20']] = pd.DataFrame(scores_df[20].tolist())
scores_df = scores_df.drop([5, 10, 15, 20, 'n_est2', 'n_est3', 'n_est4'], axis=1)
scores_df.set_index('n_est', inplace=True)

sns.lineplot(data=scores_df)
plt.xlabel("n_estimators")
plt.ylabel("auc")
plt.title("AUC scores for different max_depths and numbers of trees")
# the best results are: max_depth=10 OR max_depth=15
MAX_DEPTH = 10

# observe the impact of changing min_samples_leaf (with a tuned max_depth)
all_aucs = {}
for leaf_size in [1, 3, 5, 10, 50]:
    print('min_samples_leaf: {leaf_size}'.format(leaf_size=leaf_size))

    scores = []
    for n_est in range(10, 201, 20):
        rf_model = RandomForestClassifier(n_estimators=n_est, max_depth=MAX_DEPTH, min_samples_leaf=leaf_size,
                                          random_state=22)
        rf_model.fit(X_train_enc, y_train)
        y_pred = rf_model.predict_proba(X_val_enc)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        scores.append((n_est, auc))
        print('n_estimators: {n_est} -> {auc:.4f}'.format(n_est=n_est, auc=auc))
    all_aucs[leaf_size] = scores

# plot the results
scores_df = pd.DataFrame(all_aucs)
scores_df[['n_est', 'leaf_size=1']] = pd.DataFrame(scores_df[1].tolist())
scores_df[['n_est', 'leaf_size=3']] = pd.DataFrame(scores_df[3].tolist())
scores_df[['n_est', 'leaf_size=5']] = pd.DataFrame(scores_df[5].tolist())
scores_df[['n_est', 'leaf_size=10']] = pd.DataFrame(scores_df[10].tolist())
scores_df[['n_est', 'leaf_size=50']] = pd.DataFrame(scores_df[50].tolist())
scores_df = scores_df.drop([1, 3, 5, 10, 50], axis=1)
scores_df.set_index('n_est', inplace=True)

sns.lineplot(data=scores_df)
plt.xlabel("n_estimators")
plt.ylabel("auc")
plt.title("AUC scores for different min_samples_leaf")

sns.lineplot(data=scores_df[['leaf_size=3', 'leaf_size=5', 'leaf_size=10']])
plt.xlabel("n_estimators")
plt.ylabel("auc")
plt.title("AUC scores for different min_samples_leaf")
# not entirely clear which leaf_size is best when MAX_DEPTH = 15 (possibly 5 as it has the best performance with smaller
# numbers of trees (early on). When MAX_DEPTH = 10, the best leaf_size is very clearly 5).
MIN_LEAF_SIZE = 5

# other useful hyperparameters to tune include max_features, bootstrap and n_jobs

# train the final model
final_model = RandomForestClassifier(n_estimators=200, max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF_SIZE,
                                     random_state=22)
final_model.fit(X_train_enc, y_train)
y_pred = final_model.predict_proba(X_val_enc)[:, 1]
print(roc_auc_score(y_val, y_pred))

# compare against the final decision tree classifier
final_dt_model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=50)
final_dt_model.fit(X_train_enc, y_train)
y_pred_dt = final_dt_model.predict_proba(X_val_enc)[:, 1]
print(roc_auc_score(y_val, y_pred_dt))

y_val_num = (y_val == "default").astype(int)
fpr, tpr, _ = roc_curve(y_val_num, y_pred)
fpr_dt, tpr_dt, _ = roc_curve(y_val_num, y_pred_dt)
roc_curve_rf = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
roc_curve_dt = pd.DataFrame({'fpr_dt': fpr_dt, 'tpr_dt': tpr_dt})

sns.lineplot(data=roc_curve_rf, x='tpr', y='fpr')
sns.lineplot(data=roc_curve_dt, x='tpr_dt', y='fpr_dt', linestyle='dashed')
plt.plot([0, 1], [0, 1], color='black', linestyle='dotted')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve for the final models")

# examine feature importance
feature_importance = list(zip(final_model.feature_names_in_, final_model.feature_importances_))
print(feature_importance)
feature_importance_df = pd.DataFrame(feature_importance, columns=['feature', 'gain'])
feature_importance_df.sort_values(by='gain', ascending=False, inplace=True)

sns.barplot(feature_importance_df, x='gain', y='feature')
plt.xlabel("gain importance")
plt.title("Random forest model feature importance by gain")
sns.despine()

# removing the very insignificant variables
feature_importance_df = feature_importance_df[feature_importance_df.gain > 0.01]
sns.barplot(feature_importance_df, x='gain', y='feature')
plt.xlabel("gain importance")
plt.title("Random forest model feature importance by gain")
sns.despine()

# for funsies, compare random forest with extratrees (extremely randomised trees)
# a similar algorithm, but extratrees picks a few candidate splits at random and then selects the best one from that
# random selection instead of selecting the best possible split from all possible splits
from sklearn.ensemble import ExtraTreesClassifier

scores = []
for n_est in range(10, 201, 20):
    et_model = ExtraTreesClassifier(n_estimators=n_est, max_depth=30)
    et_model.fit(X_train_enc, y_train)
    y_pred = et_model.predict_proba(X_val_enc)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    scores.append((n_est, auc))
    print('n_estimators: {n_est} -> {auc:.4f}'.format(n_est=n_est, auc=auc))

# train the final model
final_model = ExtraTreesClassifier(n_estimators=190, max_depth=30)
final_model.fit(X_train_enc, y_train)
y_pred_et = final_model.predict_proba(X_val_enc)[:, 1]
print(roc_auc_score(y_val, y_pred_et))

fpr_rf, tpr_rf, _ = roc_curve(y_val_num, y_pred)
fpr_et, tpr_et, _ = roc_curve(y_val_num, y_pred_et)
fpr_dt, tpr_dt, _ = roc_curve(y_val_num, y_pred_dt)
roc_curve_et = pd.DataFrame({'fpr': fpr_et, 'tpr': tpr_et})
roc_curve_rf = pd.DataFrame({'fpr': fpr_rf, 'tpr': tpr_rf})
roc_curve_dt = pd.DataFrame({'fpr': fpr_dt, 'tpr': tpr_dt})

sns.lineplot(data=roc_curve_et, x='tpr', y='fpr')
sns.lineplot(data=roc_curve_rf, x='tpr', y='fpr', linestyle='dashed')
sns.lineplot(data=roc_curve_dt, x='tpr', y='fpr', linestyle='dashdot')
plt.plot([0, 1], [0, 1], color='black', linestyle='dotted')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curves for the tree models")
