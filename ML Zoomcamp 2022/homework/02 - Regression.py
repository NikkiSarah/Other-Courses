import numpy as np
import pandas as pd
import seaborn as sns
import wget

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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
finally:
    data.drop('ocean_proximity', axis=1, inplace=True)

# check the distribution of the target variable
sns.histplot(data, x="median_house_value")

# 1. Find the feature with missing values. How many are there?
missing_values = data.isnull().sum()
missing_values[missing_values != 0]

# 2. What is the median of 'population'
pop_median = data.population.median()
pop_median

# split the data and apply a log transformation to 'median_house_value'
shuffled_data = data.sample(frac=1, random_state=42)

X = shuffled_data.drop('median_house_value', axis=1)
y = np.log1p(shuffled_data.median_house_value)

X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.4)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5)

# 3. Compare the RMSE of a linear regression when missing values are filled with zero and the variable (training) mean
X_traina = X_train.copy()
X_vala = X_val.copy()
X_traina.total_bedrooms = X_train.total_bedrooms.fillna(0)
X_vala.total_bedrooms = X_val.total_bedrooms.fillna(0)

linrega = LinearRegression().fit(X_traina, y_train)
apreds = linrega.predict(X_vala)
rmsea = np.sqrt(mean_squared_error(y_val, apreds))
(round(rmsea, 2))

X_trainb = X_train.copy()
X_valb = X_val.copy()
X_trainb.total_bedrooms = X_train.total_bedrooms.fillna(X_train.total_bedrooms.mean())
X_valb.total_bedrooms = X_val.total_bedrooms.fillna(X_train.total_bedrooms.mean())

linregb = LinearRegression().fit(X_trainb, y_train)
bpreds = linrega.predict(X_valb)
rmseb = np.sqrt(mean_squared_error(y_val, bpreds))
(round(rmseb, 2))
# both models result in the same RMSE on the validation data (when rounded to two decimal places)

# 4. Train a regularised regression model with the missing values filled with zeros
r_list = [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]

for r in r_list:
    linregc = Ridge(alpha=r).fit(X_traina, y_train)
    cpreds = linregc.predict(X_vala)
    rmsec = np.sqrt(mean_squared_error(y_val, cpreds))
    # print(r)
    print(round(rmsec, 2))
# the best model is when r = 0 (need to include around 12 numbers after the decimal to see this)

# 5. Observe how selecting the seed affects the RMSE score
seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

rmse_list = []
for seed in seed_list:
    shuffled_data = data.sample(frac=1, random_state=seed)

    X = shuffled_data.drop('median_house_value', axis=1)
    y = np.log1p(shuffled_data.median_house_value)

    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5)

    X_traind = X_train.copy()
    X_vald = X_val.copy()
    X_traind.total_bedrooms = X_train.total_bedrooms.fillna(0)
    X_vald.total_bedrooms = X_val.total_bedrooms.fillna(0)

    linregd = LinearRegression().fit(X_traind, y_train)
    dpreds = linregd.predict(X_vald)
    rmsed = np.sqrt(mean_squared_error(y_val, dpreds))
    rmse_list.append(rmsed)

rmse_std = np.std(np.array(rmse_list))
(round(rmse_std, 3))
# The fact that the value is low indicates that the model is stable, because it means all the values are approximately
# the same.

# 6. Split the data using a seed of 9, fill the missing values with zeros and train a model with regularisation of 0.001
shuffled_data = data.sample(frac=1, random_state=9)

X = shuffled_data.drop('median_house_value', axis=1)
y = np.log1p(shuffled_data.median_house_value)

X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.4)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5)

X_traine = X_train.copy()
X_vale = X_val.copy()
X_teste = X_test.copy()
X_traine.total_bedrooms = X_train.total_bedrooms.fillna(0)
X_vale.total_bedrooms = X_val.total_bedrooms.fillna(0)
X_teste.total_bedrooms = X_test.total_bedrooms.fillna(0)

linrege = Ridge(alpha=0.001).fit(X_traine, y_train)
epreds = linrege.predict(X_teste)
rmsee = np.sqrt(mean_squared_error(y_test, epreds))
rmsee
