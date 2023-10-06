import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wget

from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
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
    data = pd.read_csv("Data/housing.csv")
except:
    url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv'
    wget.download(url)
    data = pd.read_csv("Data/housing.csv")
# finally:
    # data.drop('ocean_proximity', axis=1, inplace=True)

# observe the number of rows
print(len(data))

# and the first couple of rows
print(data.head())

# Exploratory Data Analysis
# observe the distribution of the target variable
num_bins = round(np.sqrt(len(data)), 0).astype('int')

plt.figure()
sns.histplot(data.median_house_value, bins=num_bins)
plt.xlabel("Median house price")
plt.ylabel("Count")
plt.title("Distribution of median prices")
sns.despine()

plt.figure()
sns.histplot(data.median_house_value[data.median_house_value < 500000], bins=num_bins)
plt.xlabel("Median house price")
plt.ylabel("Count")
plt.title("Distribution of median prices (excluding those valued at 500,000)")
sns.despine()

# note that a log transformation is a little too strong
plt.figure()
sns.histplot(np.log1p(data.median_house_value[data.median_house_value < 500000]), bins=num_bins)
plt.xlabel("log(Median house price + 1)")
plt.ylabel("Count")
plt.title("Distribution of median prices after a log transformation")
sns.despine()

# check for missing values
missing_values = data.isnull().sum()
print(missing_values[missing_values != 0])
print(round(missing_values[missing_values != 0][0] / len(data) * 100, 4))

plt.figure()
sns.histplot(data.total_bedrooms, bins=num_bins)
plt.xlabel("Total bedrooms")
plt.ylabel("Count")
plt.title("Distribution of total number of bedrooms")
sns.despine()

# replace them with the median
median_bedrooms = data.total_bedrooms.median()
data.total_bedrooms = data.total_bedrooms.fillna(median_bedrooms)

# one-hot/dummy encode the categorical variable 'ocean_proximity'
transformer = make_column_transformer(
    (OneHotEncoder(drop='first'), ['ocean_proximity']),
    remainder='passthrough', verbose_feature_names_out=False)
transformed = transformer.fit_transform(data)
data_enc = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())

# split the data into a training, validation and test set using a 60%/20%/20% split
X = data_enc.drop('median_house_value', axis=1)
y = np.log1p(data_enc.median_house_value)

X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.4, random_state=22)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=22)

# train a simple linear regression model
model = LinearRegression().fit(X_train, y_train)

# make predictions on the validation set and compare them against the actual values
train_preds = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
print('train rmse: ', round(train_rmse, 4))

validation_preds = model.predict(X_val)
validation_rmse = np.sqrt(mean_squared_error(y_val, validation_preds))
print('validation rmse: ', round(validation_rmse, 4))

num_train_bins = round(np.sqrt(len(y_train)), 0).astype('int')
num_val_bins = round(np.sqrt(len(y_val)), 0).astype('int')

plt.figure()
sns.histplot(y_train, label='target', bins=num_train_bins)
sns.histplot(train_preds, label='prediction', bins=num_train_bins)
plt.legend()
plt.xlabel("log(Median house price + 1)")
plt.ylabel("Count")
plt.title("Distribution of predicted vs actual log-transformed median prices (training data)")
sns.despine()

plt.figure()
sns.histplot(y_val, label='target', bins=num_val_bins)
sns.histplot(validation_preds, label='prediction', bins=num_val_bins)
plt.legend()
plt.xlabel("log(Median house price + 1)")
plt.ylabel("Count")
plt.title("Distribution of predicted vs actual log-transformed median prices (validation data)")
sns.despine()

# Perform some simple feature engineering and repeat the process
data_enc['rooms_per_household'] = data_enc.total_rooms / data_enc.households
data_enc['bedrooms_per_room'] = data_enc.total_bedrooms / data_enc.total_rooms
data_enc['population_per_household'] = data_enc.population / data_enc.households

# split the data into a training, validation and test set using a 60%/20%/20% split
X = data_enc.drop('median_house_value', axis=1)
y = np.log1p(data_enc.median_house_value)

X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.4, random_state=22)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=22)

# train a simple linear regression model
model = LinearRegression().fit(X_train, y_train)

# make predictions on the validation set and compare them against the actual values
train_preds = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
print('train rmse: ', round(train_rmse, 4))

validation_preds = model.predict(X_val)
validation_rmse = np.sqrt(mean_squared_error(y_val, validation_preds))
print('validation rmse: ', round(validation_rmse, 4))

num_train_bins = round(np.sqrt(len(y_train)), 0).astype('int')
num_val_bins = round(np.sqrt(len(y_val)), 0).astype('int')

plt.figure()
sns.histplot(y_train, label='target', bins=num_train_bins)
sns.histplot(train_preds, label='prediction', bins=num_train_bins)
plt.legend()
plt.xlabel("log(Median house price + 1)")
plt.ylabel("Count")
plt.title("Distribution of predicted vs actual log-transformed median prices (training data)")
sns.despine()

plt.figure()
sns.histplot(y_val, label='target', bins=num_val_bins)
sns.histplot(validation_preds, label='prediction', bins=num_val_bins)
plt.legend()
plt.xlabel("log(Median house price + 1)")
plt.ylabel("Count")
plt.title("Distribution of predicted vs actual log-transformed median prices (validation data)")
sns.despine()

# Repeat the process with a regularised model
# note that it has very little impact on the model results
for a in [0, 0.001, 0.01, 0.1, 1, 10]:
    print('alpha: ', a)
    model = Ridge(alpha=a).fit(X_train, y_train)
    train_preds = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    print('train rmse: ', train_rmse)

    validation_preds = model.predict(X_val)
    validation_rmse = np.sqrt(mean_squared_error(y_val, validation_preds))
    print('validation rmse: ', validation_rmse)
