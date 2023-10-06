import numpy as np
import pandas as pd
import seaborn as sns
import wget
import xgboost as xgb

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
y = (data.status == "default").astype(int) # convert y into its numeric binary equivalent

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

# convert the datasets into DMatrices
features = transformer.get_feature_names_out().tolist()
dtrain = xgb.DMatrix(data=X_train_enc, label=y_train, feature_names=features)
dval = xgb.DMatrix(data=X_val_enc, label=y_val, feature_names=features)

# Data Modelling
# set the parameters
xgb_params = {
    'eta': 0.3,                         # learning rate
    'max_depth': 6,
    'min_child_weight': 1,              # number of obs in a leaf node (equivalent to min_samples_leaf in a random forest)
    'objective': 'binary:logistic',     # binary classification model
    'nthread': 8,
    'seed': 22,
    'verbosity': 1                      # warnings only
}

# train the model
model = xgb.train(xgb_params, dtrain, num_boost_round=200)
y_pred = model.predict(dval)
roc_auc_score(y_val, y_pred)

# monitor training performance
# (xgboost prone to overfitting)
watchlist = [(dtrain, 'train'), (dval, 'val')]
all_eval_results = {}

xgb_params = {
    'eta': 0.3,                         # learning rate
    'max_depth': 6,
    'min_child_weight': 1,              # number of obs in a leaf node (equivalent to min_samples_leaf in a random forest)
    'objective': 'binary:logistic',     # binary classification model
    'eval_metric': 'auc',               # use auc for evaluation instead of default log-loss
    'nthread': 8,
    'seed': 22,
    'verbosity': 1                      # warnings only
}

eval_result = {}
model = xgb.train(xgb_params, dtrain, num_boost_round=500, evals=watchlist, evals_result=eval_result)
y_pred = model.predict(dval)
roc_auc_score(y_val, y_pred)

eval_result_df = pd.DataFrame(data={'train': eval_result['train']['auc'], 'val': eval_result['val']['auc']})
eval_result_df['boost_round'] = range(1, len(eval_result_df)+1)
eval_result_df.set_index('boost_round', inplace=True)

sns.lineplot(eval_result_df)
plt.xlabel("Number of trees")
plt.ylabel("auc")
plt.title("Training and validation set performance")
sns.despine()
# starts overfitting at around 75 trees

sns.lineplot(eval_result_df[['val']])
plt.xlabel("Number of trees")
plt.ylabel("auc")
plt.title("Validation set performance")
sns.despine()

# add the results to the collating dictionary
key = 'eta=%s' % xgb_params['eta']
all_eval_results[key] = eval_result

# Hyperparameter Tuning
# tune eta (learning rate)
for ETA in [1.0, 0.1, 0.05, 0.01]:
    xgb_params = {
        'eta': ETA,
        'max_depth': 6,
        'min_child_weight': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 8,
        'seed': 22,
        'verbosity': 1
    }

    eval_result = {}
    model = xgb.train(xgb_params, dtrain, num_boost_round=500, evals=watchlist, evals_result=eval_result,
                      verbose_eval=10)
    y_pred = model.predict(dval)
    roc_auc_score(y_val, y_pred)

    # add the results to the collating dictionary
    key = 'eta=%s' % xgb_params['eta']
    all_eval_results[key] = eval_result

# convert the dictionary into a dataframe
all_results = pd.json_normalize(all_eval_results, sep='_')
all_results_df = pd.DataFrame()
for col in all_results.columns:
    temp = pd.DataFrame(all_results[col].tolist()).T
    all_results_df[col] = temp
all_results_df.columns = all_results_df.columns.str.rstrip("_auc")
all_results_df['boost_round'] = range(1, len(all_results_df)+1)
all_results_df.set_index('boost_round', inplace=True)

# compare the validation results
cols = [col for col in all_results_df.columns if "val" in col]

sns.lineplot(all_results_df[cols])
plt.xlabel("Number of trees")
plt.ylabel("auc")
plt.title("Validation set performance")
sns.despine()
# the best learning rates are 0.01, 0.05 and 0.1
# both 0.1 and 0.05 are better at lower numbers of trees, but 0.01 is the best learning rate when the number of trees
# is large

sns.lineplot(all_results_df.loc[:200, ['eta=0.3_val', 'eta=1.0_val', 'eta=0.1_val', 'eta=0.05_val']])
plt.xlabel("Number of trees")
plt.ylabel("auc")
plt.title("Validation set performance")
sns.despine()
# eta=0.1 is probably the best
# next tune max_depth and finally min_child_weight

# tune max_depth (number of trees)
all_eval_results = {}
ETA = 0.1
for DEPTH in [3, 4, 6, 10]:
    xgb_params = {
        'eta': ETA,
        'max_depth': DEPTH,
        'min_child_weight': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 8,
        'seed': 22,
        'verbosity': 1
    }

    eval_result = {}
    model = xgb.train(xgb_params, dtrain, num_boost_round=500, evals=watchlist, evals_result=eval_result,
                      verbose_eval=10)
    y_pred = model.predict(dval)
    roc_auc_score(y_val, y_pred)

    # add the results to the collating dictionary
    key = 'max_depth=%s' % xgb_params['max_depth']
    all_eval_results[key] = eval_result

# convert the dictionary into a dataframe
all_results = pd.json_normalize(all_eval_results, sep='_')
all_results_df = pd.DataFrame()
for col in all_results.columns:
    temp = pd.DataFrame(all_results[col].tolist()).T
    all_results_df[col] = temp
all_results_df.columns = all_results_df.columns.str.rstrip("_auc")
all_results_df['boost_round'] = range(1, len(all_results_df)+1)
all_results_df.set_index('boost_round', inplace=True)

# compare the validation results
cols = [col for col in all_results_df.columns if "val" in col]

sns.lineplot(all_results_df[cols])
plt.xlabel("Number of trees")
plt.ylabel("auc")
plt.title("Validation set performance")
sns.despine()
# except at very large numbers of trees, the best max_depth is 3

sns.lineplot(all_results_df.loc[:200, cols])
plt.xlabel("Number of trees")
plt.ylabel("auc")
plt.title("Validation set performance")
sns.despine()

# tune min_child_weight (minimum number of observations in a leaf node)
all_eval_results = {}
ETA = 0.1
MAX_DEPTH = 3
for WEIGHT in [1, 10, 30]:
    xgb_params = {
        'eta': ETA,
        'max_depth': MAX_DEPTH,
        'min_child_weight': WEIGHT,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 8,
        'seed': 22,
        'verbosity': 1
    }

    eval_result = {}
    model = xgb.train(xgb_params, dtrain, num_boost_round=500, evals=watchlist, evals_result=eval_result,
                      verbose_eval=10)
    y_pred = model.predict(dval)
    roc_auc_score(y_val, y_pred)

    # add the results to the collating dictionary
    key = 'min_child_weight=%s' % xgb_params['min_child_weight']
    all_eval_results[key] = eval_result

# convert the dictionary into a dataframe
all_results = pd.json_normalize(all_eval_results, sep='_')
all_results_df = pd.DataFrame()
for col in all_results.columns:
    temp = pd.DataFrame(all_results[col].tolist()).T
    all_results_df[col] = temp
all_results_df.columns = all_results_df.columns.str.rstrip("_auc")
all_results_df['boost_round'] = range(1, len(all_results_df)+1)
all_results_df.set_index('boost_round', inplace=True)

# compare the validation results
cols = [col for col in all_results_df.columns if "val" in col]

sns.lineplot(all_results_df[cols])
plt.xlabel("Number of trees")
plt.ylabel("auc")
plt.title("Validation set performance")
sns.despine()
# it's very clear that the best performance is with min_child_weight = 30

# note that other useful hyperparameters to tune (in order of importance) are subsample, colsample_bytree,
# lambda and alpha

# finally, tune the number of trees
ETA = 0.1
MAX_DEPTH = 3
CHILD_WEIGHT = 30
xgb_params = {
    'eta': ETA,
    'max_depth': MAX_DEPTH,
    'min_child_weight': CHILD_WEIGHT,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 22,
    'verbosity': 1
}

eval_result = {}
model = xgb.train(xgb_params, dtrain, num_boost_round=500, evals=watchlist, evals_result=eval_result,
                  verbose_eval=10)
y_pred = model.predict(dval)
roc_auc_score(y_val, y_pred)

eval_result_df = pd.DataFrame(data={'train': eval_result['train']['auc'], 'val': eval_result['val']['auc']})
eval_result_df['boost_round'] = range(1, len(eval_result_df)+1)
eval_result_df.set_index('boost_round', inplace=True)

print('Maximum validation AUC: %.4f' % np.max(eval_result_df['val']))
print('Number of trees: %.f' % eval_result_df['val'].idxmax())

sns.lineplot(eval_result_df)
plt.axvline(185, color='black', linestyle='dotted', linewidth=0.9)
plt.xlabel("Number of trees")
plt.ylabel("auc")
plt.title("Training and validation set performance")
sns.despine()

# train the final model
ETA = 0.1
MAX_DEPTH = 3
CHILD_WEIGHT = 30
NUM_TREES = 185
xgb_params = {
    'eta': ETA,
    'max_depth': MAX_DEPTH,
    'min_child_weight': CHILD_WEIGHT,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 22,
    'verbosity': 1
}

eval_result = {}
model = xgb.train(xgb_params, dtrain, num_boost_round=NUM_TREES, evals=watchlist, evals_result=eval_result,
                  verbose_eval=10)
y_pred = model.predict(dval)
print(roc_auc_score(y_val, y_pred))

# compare against the final random forest...
MAX_DEPTH = 10
MIN_LEAF_SIZE = 5
final_model = RandomForestClassifier(n_estimators=200, max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF_SIZE,
                                     random_state=22)
final_model.fit(X_train_enc, y_train)
y_pred_rf = final_model.predict_proba(X_val_enc)[:, 1]
print(roc_auc_score(y_val, y_pred_rf))

# ...and the final decision tree classifier
final_dt_model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=50)
final_dt_model.fit(X_train_enc, y_train)
y_pred_dt = final_dt_model.predict_proba(X_val_enc)[:, 1]
print(roc_auc_score(y_val, y_pred_dt))

fpr, tpr, _ = roc_curve(y_val, y_pred)
roc_curve_xgb = pd.DataFrame({'fpr_xgb': fpr, 'tpr_xgb': tpr})
fpr, tpr, _ = roc_curve(y_val, y_pred_rf)
roc_curve_rf = pd.DataFrame({'fpr_rf': fpr, 'tpr_rf': tpr})
fpr, tpr, _ = roc_curve(y_val, y_pred_dt)
roc_curve_dt = pd.DataFrame({'fpr_dt': fpr, 'tpr_dt': tpr})

sns.lineplot(data=roc_curve_xgb, x='fpr_xgb', y='tpr_xgb', linestyle='dotted')
sns.lineplot(data=roc_curve_rf, x='fpr_rf', y='tpr_rf', linestyle='dotted')
sns.lineplot(data=roc_curve_dt, x='fpr_dt', y='tpr_dt', linestyle='dashdot')
plt.plot([0, 1], [0, 1], color='black', linestyle='dotted')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve for the final models")
