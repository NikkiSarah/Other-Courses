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

# compare the final xgboost model...
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
val_auc = roc_auc_score(y_val, y_pred)
print(val_auc)

# ...with the final random forest...
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

# choose the xgboost model and train it on the entire training data (X_train_full)
X_train_full.head()

# one-hot-encode the categorical variables
cat_cols = [col for col in X_train_full.columns if X_train[col].dtype == 'object']

transformer = make_column_transformer(
    (OneHotEncoder(), cat_cols),
    remainder='passthrough', verbose_feature_names_out=False)
transformer.fit(X_train_full)
transformed_train_full = transformer.transform(X_train_full)
X_train_full_enc = pd.DataFrame(transformed_train_full, columns=transformer.get_feature_names_out())
transformed_test = transformer.transform(X_test)
X_test_enc = pd.DataFrame(transformed_test, columns=transformer.get_feature_names_out())

# convert the datasets into DMatrices
features = transformer.get_feature_names_out().tolist()
dtrain_full = xgb.DMatrix(data=X_train_full_enc, label=y_train_full, feature_names=features)
dtest = xgb.DMatrix(data=X_test_enc, feature_names=features)

# train the model
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

model = xgb.train(xgb_params, dtrain_full, num_boost_round=NUM_TREES)
y_pred = model.predict(dtest)
print(roc_auc_score(y_test, y_pred))
# note that the validation and test AUCs are very similar, which is great as it means the model is generalising well
# to unseen data

# examine feature importance
feature_scores = model.get_score(importance_type='gain')
feature_scores = sorted(feature_scores.items(), key=lambda x: x[1])
list(reversed(feature_scores))

features = [n for (n, s) in feature_scores]
scores = [s for (n, s) in feature_scores]

imp_by_gain = pd.DataFrame(reversed(feature_scores), columns=['feature', 'score'])
sns.barplot(imp_by_gain, x='score', y='feature')
plt.xlabel("gain importance")
plt.title("XGBoost model feature importance by gain")
sns.despine()

feature_scores = model.get_score(importance_type='weight')
feature_scores = sorted(feature_scores.items(), key=lambda x: x[1])
list(reversed(feature_scores))

imp_by_weight = pd.DataFrame(reversed(feature_scores), columns=['feature', 'score'])
sns.barplot(imp_by_weight, x='score', y='feature')
plt.xlabel("weight importance")
plt.title("XGBoost model feature importance by weight")
sns.despine()

