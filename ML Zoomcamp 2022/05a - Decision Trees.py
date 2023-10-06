# Note: this is actually chapter 6 in the course
import numpy as np
import pandas as pd
import seaborn as sns
import wget

from matplotlib import pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

# check the backend and change if required
import matplotlib as mpl

mpl_backend = mpl.get_backend()
if mpl_backend != "Qt5Agg":
    mpl.use("Qt5Agg")
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
print(data.head())
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

# double-check the mapping has occurred as expected
print(data.head())

# view the summary statistics for the numeric variables
summary_stats = data.describe()

# replace 99999999 with NaN
for var in ['income', 'assets', 'debt']:
    data[var] = data[var].replace(to_replace=99999999, value=np.nan)
data.isnull().sum()
# now the summary statistics are more meaningful
summary_stats = data.describe()

# view the distribution of the target variable
data.status.value_counts()
# remove the single record with an unknown status
data = data[data.status != 'unknown']

# replace missing values in income, assets and debt with zeros
data = data.fillna(0)

# split the data into a training, validation and test set using a 60%/20%/20% split
X = data.drop('status', axis=1)
y = data.status

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=22)

len(X_train), len(X_val), len(X_test)

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
# train a decision tree with default parameters
model = DecisionTreeClassifier()
model.fit(X_train_enc, y_train)

# get the training set predictions
y_pred = model.predict_proba(X_train_enc)[:, 1]
# check the auc score
roc_auc_score(y_train, y_pred)
# compare to the auc score for the validation data
y_pred = model.predict_proba(X_val_enc)[:, 1]
roc_auc_score(y_val, y_pred)
# note that the model is heavily overfitting as the auc score on the training data is perfect, but poor by comparison on
# the validation data

# restrict the depth of the tree to 2 levels
model2 = DecisionTreeClassifier(max_depth=2)
model2.fit(X_train_enc, y_train)

y_pred = model2.predict_proba(X_train_enc)[:, 1]
auc_train = roc_auc_score(y_train, y_pred)
y_pred = model2.predict_proba(X_val_enc)[:, 1]
auc_val = roc_auc_score(y_val, y_pred)
print('train auc: {auc_train:.4f}'.format(auc_train=auc_train))
print('validation auc: {auc_val:.4f}'.format(auc_val=auc_val))
# this is much better as not only is the performance much better on the validation data, but the training and
# validation performance is very similar

# view the rules the model came up with
rules = export_text(model2, feature_names=transformer.get_feature_names_out().tolist())

# Hyperparameter Tuning
# train a model with a maximum depth of 6 levels
model3 = DecisionTreeClassifier(max_depth=6)
model3.fit(X_train_enc, y_train)

y_pred = model3.predict_proba(X_val_enc)[:, 1]
roc_auc_score(y_val, y_pred)

# try out a range of depths
depths = [1, 2, 3, 4, 5, 6, 10, 15, 20, None]
for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(X_train_enc, y_train)
    y_pred = model.predict_proba(X_val_enc)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print('depth: {depth} -> auc: {auc:.4f}'.format(depth=depth, auc=auc))
# observe that the best was max_depth = 5

# try out a range of minimum leaf sizes (with a specified depth)
leaf_sizes = [1, 5, 10, 15, 20, 50, 100, 200]
for min_size in leaf_sizes:
    model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=min_size)
    model.fit(X_train_enc, y_train)
    y_pred = model.predict_proba(X_val_enc)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print('min leaf size: {min_size} -> auc: {auc:.4f}'.format(min_size=min_size, auc=auc))
# observe that the best was min_samples_leaf = 20

# try out a range of depths and minimum leaf sizes simultaneously
scores = []
for depth in [4, 5, 6, 7, 10, 15, 20, None]:
    print('depth: {depth}'.format(depth=depth))
    for min_size in [1, 5, 10, 15, 20, 50, 100, 200]:
        model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_size)
        model.fit(X_train_enc, y_train)
        y_pred = model.predict_proba(X_val_enc)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((depth, min_size, auc))
        # print('min leaf size: {min_size} -> auc: {auc:.4f}'.format(min_size=min_size, auc=auc))
# observe that the best was depth = 6 and min_samples_leaf = 50

# visualise the results in a heatmap
scores_df = pd.DataFrame(scores, columns=['max_depth', 'min_samples_leaf', 'auc'])
scores_df.sort_values(by='auc', ascending=False)

pivoted_scores_df = scores_df.pivot(index='min_samples_leaf', columns='max_depth', values='auc')
sns.heatmap(pivoted_scores_df, annot=True, fmt='.3f')

# plot the roc curve for the chosen model
final_model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=50)
final_model.fit(X_train_enc, y_train)
y_pred = final_model.predict_proba(X_val_enc)[:, 1]

y_val_num = (y_val == "default").astype(int)
fpr, tpr, _ = roc_curve(y_val_num, y_pred)
roc_curve_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})

sns.lineplot(data=roc_curve_df, x='tpr', y='fpr')
plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve for the final model")

# examine feature importance
feature_importance = list(zip(final_model.feature_names_in_, final_model.feature_importances_))
print(feature_importance)
feature_importance_df = pd.DataFrame(feature_importance, columns=['feature', 'gain'])
feature_importance_df.sort_values(by='gain', ascending=False, inplace=True)

sns.barplot(feature_importance_df, x='gain', y='feature')
plt.xlabel("gain importance")
plt.title("Decision tree model feature importance by gain")
sns.despine()

# removing the very insignificant variables
feature_importance_df = feature_importance_df[feature_importance_df.gain > 0]
sns.barplot(feature_importance_df, x='gain', y='feature')
plt.xlabel("gain importance")
plt.title("Decision model feature importance by gain")
sns.despine()
