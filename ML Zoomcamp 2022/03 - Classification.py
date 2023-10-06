import numpy as np
import pandas as pd
import wget

from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mutual_info_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# check the backend and change if required
import matplotlib as mpl

mpl_backend = mpl.get_backend()
if mpl_backend != "Qt5Agg":
    mpl.use("Qt5Agg")
else:
    pass

# read in the data
try:
    data = pd.read_csv("housing.csv")
except:
    url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv'
    wget.download(url)
    data = pd.read_csv("housing.csv")

# Data Preparation
# fill any missing values with the median of that variable
missing_values = data.isnull().sum()
print(missing_values[missing_values != 0])

median_bedrooms = data.total_bedrooms.median()
data.total_bedrooms = data.total_bedrooms.fillna(value=median_bedrooms)

# create rooms_per_household
data['rooms_per_household'] = data.total_rooms / data.households
# create bedrooms_per_room
data['bedrooms_per_room'] = data.total_bedrooms / data.total_rooms
# create population_per_household
data['population_per_household'] = data.population / data.households

# 1. What is the mode of 'ocean proximity'?
ocean_proximity_mode = data.ocean_proximity.mode()[0]
print(ocean_proximity_mode)

# 2. What are the two features with the largest correlation in this dataset?
corr_mat = data.corr(numeric_only=True)
print(corr_mat)

from itertools import combinations
var_pair = []
corr_list = []
for m, n in list(combinations(data.columns, 2)):
    if m != 'ocean_proximity':
        if n != 'ocean_proximity':
            var_pair.append(m + ", " + n)
            corr_list.append(np.corrcoef(data[m], data[n])[0][1])
corr_df = pd.DataFrame(data={'variable_pair': var_pair, 'correlation': corr_list})
corr_df.sort_values('correlation', ascending=False, inplace=True)
print(corr_df.iloc[0, :])

# convert median_house_value into a binary variable ('above_average')
mean_median_house_value = data.median_house_value.mean()
data_clf = data.copy()
data_clf['above_average'] = data_clf.median_house_value.apply(lambda x: 1 if x > mean_median_house_value else 0)
data_clf.drop('median_house_value', axis=1, inplace=True)

# split the data into train-test-validation with a 60%/20%/20% distribution
X = data_clf.drop('above_average', axis=1)
y = data_clf.above_average

X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# 3. What is the mutual information score between 'above_average' and 'ocean_proximity' in the training data?
mi_score = mutual_info_score(labels_true=X_train.ocean_proximity, labels_pred=y_train)
print(round(mi_score, 2))

# Train a logistic regression model
transformer = make_column_transformer(
    (OneHotEncoder(drop='first'), ['ocean_proximity']),
    remainder='passthrough', verbose_feature_names_out=False)
fitted = transformer.fit(X_train)
transformed = transformer.transform(X_train)
X_train_transformed = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
fitted_model = model.fit(X_train_transformed, y_train)

# 4. What is the validation set accuracy?
transformed = transformer.transform(X_val)
X_val_transformed = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())

validation_preds = fitted_model.predict(X_val_transformed)
validation_accuracy = accuracy_score(y_val, validation_preds)
print(round(validation_accuracy, 2))

# 5. The removal of which feature produces the smallest difference in accuracy on the validation data?
accuracy_scores = []
for feature in X_train.columns:
    X_train_sub = X_train.drop(feature, axis=1)
    X_val_sub = X_val.drop(feature, axis=1)

    if feature != 'ocean_proximity':
        fitted = transformer.fit(X_train_sub)
        transformed = transformer.transform(X_train_sub)
        X_train_transformed = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())

        model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
        fitted_model = model.fit(X_train_transformed, y_train)

        transformed = transformer.transform(X_val_sub)
        X_val_transformed = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())

        validation_preds = fitted_model.predict(X_val_transformed)
        validation_accuracy = accuracy_score(y_val, validation_preds)
    else:
        model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
        fitted_model = model.fit(X_train_sub, y_train)

        validation_preds = fitted_model.predict(X_val_sub)
        validation_accuracy = accuracy_score(y_val, validation_preds)

    # print(feature)
    # print(validation_accuracy)
    accuracy_scores.append(validation_accuracy)

accuracy_df = pd.DataFrame(data={'feature':X_train.columns, 'model_accuracy': accuracy_scores})
accuracy_df['accuracy_diff'] = np.abs(validation_accuracy - accuracy_df.model_accuracy)
accuracy_df.sort_values('accuracy_diff', inplace=True)
print(accuracy_df.iloc[0, :])

accuracy_df.sort_values('model_accuracy', ascending=False, inplace=True)
print(accuracy_df.iloc[0, :])

# Train a ridge regression model
data_reg = data.copy()
data_reg['median_house_value_log'] = np.log1p(data_reg.median_house_value)
data_reg.drop('median_house_value', inplace=True, axis=1)

X = data_reg.drop('median_house_value_log', axis=1)
y = data_reg.median_house_value_log

X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

fitted = transformer.fit(X_train)
transformed = transformer.transform(X_train)
X_train_transformed = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())

transformed = transformer.transform(X_val)
X_val_transformed = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())

# 6. Which value of alpha produces the lowest RMSE on the validation set
rmse_scores = []
for a in [0, 0.01, 0.1, 1, 10]:
    model = Ridge(alpha=a, solver="sag", random_state=42)
    fitted_model = model.fit(X_train_transformed, y_train)

    validation_preds = model.predict(X_val_transformed)
    validation_rmse = np.sqrt(mean_squared_error(y_val, validation_preds))
    rmse_scores.append(validation_rmse)

rmse_df = pd.DataFrame(data={'alpha':[0, 0.01, 0.1, 1, 10], 'model_rmse': rmse_scores})
rmse_df.sort_values(['model_rmse', 'alpha'], inplace=True, ascending=False)
print(rmse_df.iloc[0, :])
# note that they're actually all the same so the answer should be zero
