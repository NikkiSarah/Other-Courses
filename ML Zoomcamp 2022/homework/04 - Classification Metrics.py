import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, plot_precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

# check the backend and change if required
import matplotlib as mpl

mpl_backend = mpl.get_backend()
if mpl_backend != "Qt5Agg":
    mpl.use("Qt5Agg")
else:
    pass

# read in the data
data = pd.read_csv("Data/AER_credit_card_data.csv")

# Data Preparation
# create the binary target variable
data['card_target'] = data.card.apply(lambda x: 1 if x == "yes" else 0)
# split the data into training, validation and test sets with a 60%/20%/20% distribution
X = data.drop(['card', 'card_target'], axis=1)
y = data.card_target

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=1)

# 1. Use ROC AUC to assess the feature importance of the numeric variables in the training data.
# Which variable has the highest AUC?
num_cols = [col for col in X_train.columns if X_train[col].dtype != 'object']

for col in num_cols:
    auc = roc_auc_score(y_train, X_train[col])
    # if an AUC is < 0.5, it means that the feature is negatively correlated with the target
    if auc < 0.5:
        auc = roc_auc_score(y_train, -X_train[col])
    print('Feature: {feature} -> AUC: {auc}'.format(feature=col, auc=auc))

# 2. What is the AUC on the validation data of a logistic regression model
cat_cols = [col for col in X_train.columns if col not in num_cols]

# encode the categorical features
transformer = make_column_transformer(
    (OneHotEncoder(), cat_cols),
    remainder='passthrough', verbose_feature_names_out=False)
transformer.fit(X_train)
transformed_train = transformer.transform(X_train)
X_train_enc = pd.DataFrame(transformed_train, columns=transformer.get_feature_names_out())
transformed_val = transformer.transform(X_val)
X_val_enc = pd.DataFrame(transformed_val, columns=transformer.get_feature_names_out())

# train the model
lr_model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
lr_model.fit(X_train_enc, y_train)
lr_pred = lr_model.predict_proba(X_val_enc)[:, 1]
print(roc_auc_score(y_val, lr_pred))

# 3. At which threshold do the precision and recall curves intersect
# evaluate the model on all thresholds from 0.0 to 1.0 with step 0.01
num_steps = int(1 / 0.01 + 1)
thresholds = np.linspace(0.0, 1.0, num_steps)

scores = []
for thresh in thresholds:
    lr_prec = precision_score(y_val, lr_pred >= thresh)
    lr_rec = recall_score(y_val, lr_pred >= thresh)
    scores.append((thresh, lr_prec, lr_rec))
scores_df = pd.DataFrame(scores, columns=['threshold', 'precision', 'recall'])
scores_df.set_index('threshold', inplace=True)
scores_df['intersect'] = scores_df['precision'] == scores_df['recall']

sns.lineplot(scores_df[['precision', 'recall']])
plt.axvline(0.265, linestyle='dotted', linewidth=0.9, color='black')
plt.title("Logistic regression precision and recall scores on the validation dataset")
sns.despine()

# 4. At which threshold is the F1 score at its maximum value?
scores_df['f1_score'] = 2 * ((scores_df.precision * scores_df.recall) / (scores_df.precision + scores_df.recall))
print(scores_df.f1_score.idxmax().round(2))

sns.lineplot(scores_df[['precision', 'recall', 'f1_score']])
plt.axvline(0.265, linestyle='dotted', linewidth=0.9, color='black')
plt.title("Logistic regression precision and recall scores on the validation dataset")
sns.despine()

# 5. Use five-fold cross-validation to evaluate the model. What is the standard deviation of the AUC scores?
# initiate the cross-validation algorithm
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
train_full = X_train_full.copy()
train_full['card'] = y_train_full.copy()

aucs = []
for train_idx, val_idx in kfold.split(train_full):
    # split the data
    train_data = train_full.iloc[train_idx]
    y_train = train_data.card.values

    val_data = train_full.iloc[val_idx]
    y_val = val_data.card.values

    # encode the training data
    cat_cols = [col for col in train_data.columns if train_data[col].dtype == "object"]
    X_train = train_data.drop('card', axis=1)

    transformer = make_column_transformer(
        (OneHotEncoder(), cat_cols),
        remainder='passthrough', verbose_feature_names_out=False)
    transformer.fit(X_train)
    transformed_X_train = transformer.transform(X_train)
    X_train_enc = pd.DataFrame(transformed_X_train, columns=transformer.get_feature_names_out())

    # train the model
    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    model.fit(X_train_enc, y_train)

    # encode the validation data
    X_val = val_data.drop('card', axis=1)

    transformed_X_val = transformer.transform(X_val)
    X_val_enc = pd.DataFrame(transformed_X_val, columns=transformer.get_feature_names_out())

    # make predictions
    val_pred_probs = model.predict_proba(X_val_enc)[:, 1]

    # evaluate model accuracy
    roc_auc = roc_auc_score(y_val, val_pred_probs)
    aucs.append(roc_auc)

# view all the auc values
np.array(aucs).round(3)
print('auc = %0.3f +/- %0.3f' % (np.mean(aucs), np.std(aucs)))

# 6. What value of C leads to the best mean score?
kfold = KFold(n_splits=5, shuffle=True, random_state=1)

for C in [0.01, 0.1, 1, 10]:
    aucs = []
    for train_idx, val_idx in kfold.split(train_full):
        # split the data
        train_data = train_full.iloc[train_idx]
        y_train = train_data.card.values

        val_data = train_full.iloc[val_idx]
        y_val = val_data.card.values

        # encode the training data
        cat_cols = [col for col in train_data.columns if
                    (train_data[col].dtype == "object") or (col == "SeniorCitizen")]
        X_train = train_data.drop('card', axis=1)

        transformer = make_column_transformer(
            (OneHotEncoder(), cat_cols),
            remainder='passthrough', verbose_feature_names_out=False)
        transformer.fit(X_train)
        transformed_X_train = transformer.transform(X_train)
        X_train_enc = pd.DataFrame(transformed_X_train, columns=transformer.get_feature_names_out())

        # train the model
        model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
        model.fit(X_train_enc, y_train)

        # encode the validation data
        X_val = val_data.drop('card', axis=1)

        transformed_X_val = transformer.transform(X_val)
        X_val_enc = pd.DataFrame(transformed_X_val, columns=transformer.get_feature_names_out())

        # make predictions
        val_pred_probs = model.predict_proba(X_val_enc)[:, 1]

        # evaluate model accuracy
        roc_auc = roc_auc_score(y_val, val_pred_probs)
        aucs.append(roc_auc)

    print('C = %s, auc = %0.3f +/- %0.3f' % (C, np.mean(aucs), np.std(aucs)))

