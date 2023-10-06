import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, mutual_info_score, precision_score, recall_score, roc_auc_score, \
    roc_curve
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
data = pd.read_csv("Data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Data Preparation
# observe the number of rows
print(len(data))
# and the first couple of rows
print(data.head())
# and all the data for the first and last customers
print(data.T)
# get rid of customerID
data.drop("customerID", inplace=True, axis=1)

# consider the datatypes and missingness of each variable
print(data.info())
# convert total charges to a numeric and fill missing values with zero
data.TotalCharges = pd.to_numeric(data.TotalCharges, errors='coerce')
data.TotalCharges = data.TotalCharges.fillna(0)

# convert the target variable into its numeric equivalent where churning is the positive class
data.Churn = (data.Churn == 'Yes').astype(int)
data.Churn.head()

# split the data into a training, validation and test set using a 60%/20%/20% split
X = data.drop('Churn', axis=1)
y = data.Churn

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=22)

# Exploratory Data Analysis
# check the distribution of the target variable
print(y_train_full.value_counts())

# calculate the global probability of churning
global_mean = y_train_full.mean()
print(round(global_mean, 2))

# observe the cardinality of the categorical columns
cat_cols = [col for col in X_train_full.columns if (X_train_full[col].dtype == "object") or (col == "SeniorCitizen")]

for col in X_train_full.columns:
    if col in cat_cols:
        print(col + ": " + str(X_train_full[col].nunique()))

# investigate the feature importance of the categorical columns
# looking for features with big differences in risk between categories
# add the target variable back to X_train_full
X_train_full['Churn'] = y_train_full

for col in X_train_full.columns:
    if col in cat_cols:
        group_means = X_train_full.groupby(col).Churn.agg(['mean'])
        group_means['diff'] = group_means['mean'] - global_mean
        group_means['risk'] = group_means['mean'] / global_mean
        print(group_means)
# SeniorCitizen, Partner, Dependents, InternetService, OnlineSecurity, OnlineBackUp, DeviceProtection, TechSupport,
# Contract, PaperlessBilling and PaymentMethod appear to be the most relevant to the churn decision

# alternative is to use mutual information, which is a concept borrowed from information theory and estimates how much
# we can learn about a (target) variable if the value of another variable is known
# note that the resulting scores are relative, so only the relative ordering of the variables can be deduced
cols = []
mi_scores = []
for col in X_train_full.columns:
    if col in cat_cols:
        mi_score = mutual_info_score(X_train_full[col], X_train_full.Churn)
        cols.append(col)
        mi_scores.append(mi_score)
mi_score_df = pd.DataFrame(list(zip(cols, mi_scores)), columns=['feature', 'score'])
mi_score_df = mi_score_df.sort_values(by='score', ascending=False)
print(mi_score_df)
# Contract, OnlineSecurity, TechSupport, InternetService and OnlineBackup appear to be the most relevant to the churn
# decision according to this method

# observe the correlation with the target variable for any numeric variables
num_cols = [col for col in X_train_full.columns if col not in cat_cols and col != "Churn"]

for col in X_train_full.columns:
    if col in num_cols:
        print(col + ": " + str(X_train_full[col].corr(X_train_full.Churn)))

# observe the propensity to churn by churn status
print(X_train_full.groupby('Churn')[num_cols].mean())

# one-hot-encode the categorical variables
# (most ML models can't handle non-numeric features)
transformer = make_column_transformer(
    (OneHotEncoder(), cat_cols),
    remainder='passthrough', verbose_feature_names_out=False)
transformer.fit(X_train)
transformed_train = transformer.transform(X_train)
X_train_enc = pd.DataFrame(transformed_train, columns=transformer.get_feature_names_out())
transformed_val = transformer.transform(X_val)
X_val_enc = pd.DataFrame(transformed_val, columns=transformer.get_feature_names_out())

# Data Modelling
# train a logistic regression model
model = LogisticRegression(solver='liblinear', random_state=22)
model.fit(X_train_enc, y_train)

val_pred_probs = model.predict_proba(X_val_enc)[:, 1]
print(val_pred_probs)

# extract the model intercept
print(model.intercept_)

# extract all the coefficients
print(dict(zip(model.feature_names_in_, model.coef_[0])))

# make predictions on the test data
transformed_test = transformer.transform(X_test)
X_test_enc = pd.DataFrame(transformed_test, columns=transformer.get_feature_names_out())

test_pred_probs = model.predict_proba(X_test_enc)[:, 1]
test_pred_classes = model.predict(X_test_enc)
print(test_pred_probs)
print(test_pred_classes)

# train a second model on just Contract, Tenure and TotalCharges
model2_vars = [col for col in X_train_enc.columns if col in ['TotalCharges', 'tenure'] or col.startswith('Contract')]

model2 = LogisticRegression(solver='liblinear', random_state=22)
model2.fit(X_train_enc[model2_vars], y_train)

val_pred_probs2 = model2.predict_proba(X_val_enc[model2_vars])[:, 1]
print(val_pred_probs2)

# extract the model intercept
print(model2.intercept_)

# extract all the coefficients
print(dict(zip(model2.feature_names_in_, model2.coef_[0])))

# make predictions on the test data
test_pred_probs2 = model2.predict_proba(X_test_enc[model2_vars])[:, 1]
test_pred_classes2 = model2.predict(X_test_enc[model2_vars])
print(test_pred_probs2)
print(test_pred_classes2)

# Model Performance Assessment
# accuracy score - by hand
churn = val_pred_probs > 0.5
print('accuracy of full model: ' + str((y_val == churn).mean()))
# accuracy score - using sklearn
val_preds = model.predict(X_val_enc)
model_acc = accuracy_score(y_val, val_preds)
print('accuracy of full model: ' + str(model_acc))

# accuracy score of smaller model
val_preds2 = model2.predict(X_val_enc[model2_vars])
model_acc2 = accuracy_score(y_val, val_preds2)
print('accuracy of smaller model: ' + str(model_acc2))
# note how accuracy has decreased, but not by much

# compare the performance against a baseline model
size_val = len(y_val)
baseline_preds = np.repeat(0, size_val)
baseline_acc = accuracy_score(y_val, baseline_preds)
print('accuracy of baseline model: ' + str(baseline_acc))
# note how the model in which the negative class is always predicted does not perform much worse than either of the
# 'real' models. This indicates that perhaps accuracy is not such a great metric for this dataset and/or seemingly
# decent performance of 75% to 80% accuracy is not actually that great.

# observe accuracy scores of the full model at different thresholds (the default is 0.5)
thresholds = np.linspace(0, 1, 11)

accs = []
for t in thresholds:
    acc = accuracy_score(y_val, val_pred_probs >= t)
    accs.append(acc)
    print('threshold: ' + str(t) + '; accuracy: ' + str(acc))

plot_df = pd.DataFrame(data={'threshold': thresholds, 'val_accuracy': accs})
sns.lineplot(data=plot_df, x="threshold", y="val_accuracy")
plt.xlabel("Threshold")
plt.ylabel("Accuracy score")
plt.title("Accuracy on the validation set for different thresholds")

# focus on the 'full model' from here on unless otherwise specified
# construct a confusion matrix - by hand
tp = ((val_pred_probs >= 0.5) & (y_val == 1)).sum()
fp = ((val_pred_probs >= 0.5) & (y_val == 0)).sum()
fn = ((val_pred_probs < 0.5) & (y_val == 1)).sum()
tn = ((val_pred_probs < 0.5) & (y_val == 0)).sum()

confusion_table = np.array(
    # predict   negative    positive
                [[tn,       fp],        # actual negative
                 [fn,        tp]])       # actual positive
print(confusion_table)
print(confusion_table / confusion_table.sum()) # to get the normalised results

# calculate precision and recall - by hand
model_precision = tp / (tp + fp)
model_recall = tp / (tp + fn)
print('precision: ' + str(model_precision))
print('recall: ' + str(model_recall))

# now compare to the sklearn functions - note how they're identical
model_precision = precision_score(y_val, val_preds)
model_recall = recall_score(y_val, val_preds)
print('precision: ' + str(model_precision))
print('recall: ' + str(model_recall))

# observe how the true and false positive rates change at different thresholds
def create_tpr_fpr_dataframe(y_val, y_preds):
    thresholds = np.linspace(0, 1, 101)

    scores = []
    for t in thresholds:
        tp = ((y_preds >= t) & (y_val == 1)).sum()
        fp = ((y_preds >= t) & (y_val == 0)).sum()
        fn = ((y_preds < t) & (y_val == 1)).sum()
        tn = ((y_preds < t) & (y_val == 0)).sum()
        scores.append((t, tp, fp, fn, tn))
    scores_df = pd.DataFrame(scores, columns=['threshold', 'tp', 'fp', 'fn', 'tn'])

    scores_df['tpr'] = scores_df.tp / (scores_df.tp + scores_df.fn)
    scores_df['fpr'] = scores_df.fp / (scores_df.fp + scores_df.tn)

    return scores_df

model_tpr_fpr_df = create_tpr_fpr_dataframe(y_val, val_pred_probs)

sns.lineplot(data=model_tpr_fpr_df[['tpr', 'fpr']])
plt.xlabel("Threshold")
plt.ylabel("Rate")
plt.title("True and false positive rates on the validation set for different thresholds");

# compare to the results for a random baseline model
np.random.seed(1)
val_pred_rand = np.random.uniform(0, 1, size=len(y_val))
rand_tpr_fpr_df = create_tpr_fpr_dataframe(y_val, val_pred_rand)

sns.lineplot(data=rand_tpr_fpr_df[['tpr', 'fpr']])
plt.xlabel("Threshold")
plt.ylabel("Rate")
plt.title("Random baseline model true and false positive rates on the validation set for different thresholds");

# and to the results for an ideal model
num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()

val_ideal = np.repeat([0, 1], [num_neg, num_pos])
val_pred_ideal = np.linspace(0, 1, num_neg+num_pos)

ideal_tpr_fpr_df = create_tpr_fpr_dataframe(val_ideal, val_pred_ideal)

sns.lineplot(data=ideal_tpr_fpr_df[['tpr', 'fpr']])
plt.xlabel("Threshold")
plt.ylabel("Rate")
plt.title("Ideal model true and false positive rates on the validation set for different thresholds");

# plot the ROC curve for each model (full, baseline and ideal) - by hand
sns.lineplot(data=model_tpr_fpr_df, x='fpr', y='tpr')
sns.lineplot(data=rand_tpr_fpr_df, x='fpr', y='tpr', linestyle='dashed')
sns.lineplot(data=ideal_tpr_fpr_df, x='fpr', y='tpr', linestyle='dashdot')
plt.plot([0, 1], [0, 1], color='black', linestyle='dotted')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curves for the full, baseline and ideal models")

# now use sklearn
fpr, tpr, thresholds = roc_curve(y_val, val_pred_probs)
roc_curve_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})

sns.lineplot(data=roc_curve_df, x='fpr', y='tpr')
plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve for the full model")

# compute the area under the curve for the full model
auc(model_tpr_fpr_df.fpr, model_tpr_fpr_df.tpr)
# and for the smaller model
model2_tpr_fpr_df = create_tpr_fpr_dataframe(y_val, val_pred_probs2)
auc(model2_tpr_fpr_df.fpr, model2_tpr_fpr_df.tpr)

# compare the ROC curves of multiple models
fpr_large, tpr_large, _ = roc_curve(y_val, val_pred_probs)
fpr_small, tpr_small, _ = roc_curve(y_val, val_pred_probs2)
large_roc_curve_df = pd.DataFrame({'fpr': fpr_large, 'tpr': tpr_large})
small_roc_curve_df = pd.DataFrame({'fpr': fpr_small, 'tpr': tpr_small})

sns.lineplot(data=large_roc_curve_df, x='fpr', y='tpr')
sns.lineplot(data=small_roc_curve_df, x='fpr', y='tpr', linestyle='dashed')
plt.plot([0, 1], [0, 1], color='black', linestyle='dotted')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curves for multiple models")

# use sklearn to calculate the auc scores
print(roc_auc_score(y_val, val_pred_probs))
print(roc_auc_score(y_val, val_pred_probs2))

# note the definition of AUC: the probability that a randomly chosen 'positive' example ranks higher than a randomly
# chosen 'negative' example

# K-Fold Cross-Validation
train_full, test = train_test_split(data, test_size=0.2, random_state=22)
# X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=22)

# initiate the cross-validation algorithm
kfold = KFold(n_splits=10, shuffle=True, random_state=1)

aucs = []
for train_idx, val_idx in kfold.split(train_full):
    # split the data
    train_data = train_full.iloc[train_idx]
    y_train = train_data.Churn.values

    val_data = train_full.iloc[val_idx]
    y_val = val_data.Churn.values

    # encode the training data
    cat_cols = [col for col in train_data.columns if
                (train_data[col].dtype == "object") or (col == "SeniorCitizen")]
    X_train = train_data.drop('Churn', axis=1)

    transformer = make_column_transformer(
        (OneHotEncoder(), cat_cols),
        remainder='passthrough', verbose_feature_names_out=False)
    transformer.fit(X_train)
    transformed_X_train = transformer.transform(X_train)
    X_train_enc = pd.DataFrame(transformed_X_train, columns=transformer.get_feature_names_out())

    # train the model
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train_enc, y_train)

    # encode the validation data
    X_val = val_data.drop('Churn', axis=1)

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

# Hyperparameter Tuning
# tune C (amount of regularisation)
kfold = KFold(n_splits=10, shuffle=True, random_state=1)

for C in [0.001, 0.01, 0.1, 0.5, 1, 10]:
    aucs = []
    for train_idx, val_idx in kfold.split(train_full):
        # split the data
        train_data = train_full.iloc[train_idx]
        y_train = train_data.Churn.values

        val_data = train_full.iloc[val_idx]
        y_val = val_data.Churn.values

        # encode the training data
        cat_cols = [col for col in train_data.columns if
                    (train_data[col].dtype == "object") or (col == "SeniorCitizen")]
        X_train = train_data.drop('Churn', axis=1)

        transformer = make_column_transformer(
            (OneHotEncoder(), cat_cols),
            remainder='passthrough', verbose_feature_names_out=False)
        transformer.fit(X_train)
        transformed_X_train = transformer.transform(X_train)
        X_train_enc = pd.DataFrame(transformed_X_train, columns=transformer.get_feature_names_out())

        # train the model
        model = LogisticRegression(solver='liblinear', C=C)
        model.fit(X_train_enc, y_train)

        # encode the validation data
        X_val = val_data.drop('Churn', axis=1)

        transformed_X_val = transformer.transform(X_val)
        X_val_enc = pd.DataFrame(transformed_X_val, columns=transformer.get_feature_names_out())

        # make predictions
        val_pred_probs = model.predict_proba(X_val_enc)[:, 1]

        # evaluate model accuracy
        roc_auc = roc_auc_score(y_val, val_pred_probs)
        aucs.append(roc_auc)

    print('C = %s, auc = %0.3f +/- %0.3f' % (C, np.mean(aucs), np.std(aucs)))

# train a model on all the 'training' data using C = 0.5
y_train = train_full.Churn
X_train = train_full.drop('Churn', axis=1)

cat_cols = [col for col in X_train.columns if
            (X_train[col].dtype == "object") or (col == "SeniorCitizen")]
transformer = make_column_transformer(
    (OneHotEncoder(), cat_cols),
    remainder='passthrough', verbose_feature_names_out=False)
transformer.fit(X_train)
transformed_X_train = transformer.transform(X_train)
X_train_enc = pd.DataFrame(transformed_X_train, columns=transformer.get_feature_names_out())

model = LogisticRegression(solver='liblinear', C=0.5)
model.fit(X_train_enc, y_train)

y_test = test.Churn
X_test = test.drop('Churn', axis=1)
transformed_X_test = transformer.transform(X_test)
X_test_enc = pd.DataFrame(transformed_X_test, columns=transformer.get_feature_names_out())

test_pred_probs = model.predict_proba(X_test_enc)[:, 1]
roc_auc = roc_auc_score(y_test, test_pred_probs)
print('auc = %0.3f' % roc_auc)
