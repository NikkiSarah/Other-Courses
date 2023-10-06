import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
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
data = pd.read_csv("Data/housing.csv")

# Data Preparation
# fill any missing values with zeros
data.isna().sum()
data.total_bedrooms = data.total_bedrooms.fillna(0)

# apply a log transformation to median_house_value
data.median_house_value = np.log(data.median_house_value)

# adjust ocean_proximity values
proximity_dict = {
    '<1H OCEAN': 'LT1H_OCEAN',
    'INLAND': 'INLAND',
    'NEAR OCEAN': 'NEAR_OCEAN',
    'NEAR BAY': 'NEAR_BAY',
    'ISLAND': 'ISLAND'
}
data['ocean_proximity'] = data['ocean_proximity'].map(proximity_dict)
data['ocean_proximity'].value_counts()

# split the data into training, validation and test sets with a 60%/20%/20% distribution
X = data.drop('median_house_value', axis=1)
y = data.median_house_value

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=1)

# encode ocean_proximity
transformer = make_column_transformer(
    (OneHotEncoder(), ['ocean_proximity']),
    remainder='passthrough', verbose_feature_names_out=False)
transformer.fit(X_train)
transformed_train = transformer.transform(X_train)
X_train_enc = pd.DataFrame(transformed_train, columns=transformer.get_feature_names_out())
transformed_val = transformer.transform(X_val)
X_val_enc = pd.DataFrame(transformed_val, columns=transformer.get_feature_names_out())

# 1. Train a decision tree model to predict median_house_value. Which feature is used to split the data?
dt_model = DecisionTreeRegressor(max_depth=1)
dt_model.fit(X_train_enc, y_train)

# view the rules the model came up with
rules = export_text(dt_model, feature_names=dt_model.feature_names_in_.tolist())
rules

# 2. Train a random forest model. What is the validation RMSE?
rf_model = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf_model.fit(X_train_enc, y_train)
y_pred = rf_model.predict(X_val_enc)
rmse = np.sqrt(mean_squared_error(y_val, y_pred)).round(4)
print(rmse)

# 3. Experiment with different numbers of trees. When does RMSE stop improving on the validation dataset?
rmse_list = []
for n_est in range(10, 201, 10):
    rf_model = RandomForestRegressor(n_estimators=n_est, random_state=1, n_jobs=-1)
    rf_model.fit(X_train_enc, y_train)
    y_pred = rf_model.predict(X_val_enc)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse_list.append((n_est, rmse))

rmse_df = pd.DataFrame(rmse_list, columns=['n_estimators', 'rmse'])
rmse_df.set_index('n_estimators', inplace=True)

sns.lineplot(rmse_df)
plt.xlabel('n_estimators')
plt.ylabel('RMSE')
plt.title('Performance on the validation data for different numbers of trees')
sns.despine()

# 4. Experiment with different numbers of trees and tree depth. What is the best max depth?
scores = []
for depth in [10, 15, 20, 25]:
    print('depth: {depth}'.format(depth=depth))
    for n_est in range(10, 201, 10):
        model = RandomForestRegressor(max_depth=depth, n_estimators=n_est, random_state=1, n_jobs=-1)
        model.fit(X_train_enc, y_train)
        y_pred = model.predict(X_val_enc)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        scores.append((depth, n_est, rmse))

scores_df = pd.DataFrame(scores, columns=['max_depth', 'n_estimators', 'rmse'])
scores_df.sort_values(by='rmse', ascending=False)

pivoted_scores_df = scores_df.pivot(index='n_estimators', columns='max_depth', values='rmse')
sns.heatmap(pivoted_scores_df, annot=True, fmt='.4f')

# 5. What is the most important feature from a random forest model?
rf_model = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
rf_model.fit(X_train_enc, y_train)

feature_importance = list(zip(rf_model.feature_names_in_, rf_model.feature_importances_))
print(feature_importance)
feature_importance_df = pd.DataFrame(feature_importance, columns=['feature', 'gain'])
feature_importance_df.sort_values(by='gain', ascending=False, inplace=True)

sns.barplot(feature_importance_df, x='gain', y='feature')
plt.xlabel("gain importance")
plt.title("Random forest model feature importance by gain")
sns.despine()

# 6. Train a xgboost model. What learning rate produces the best RMSE score on the validation dataset?
# convert the datasets into DMatrices
features = transformer.get_feature_names_out().tolist()
dtrain = xgb.DMatrix(data=X_train_enc, label=y_train, feature_names=features)
dval = xgb.DMatrix(data=X_val_enc, label=y_val, feature_names=features)

watchlist = [(dtrain, 'train'), (dval, 'val')]
all_eval_results = {}

# tune eta (learning rate)
for ETA in [0.3, 0.1]:
    xgb_params = {
        'eta': ETA,
        'max_depth': 6,
        'min_child_weight': 1,
        'objective': 'reg:squarederror',
        'nthread': 8,
        'seed': 1,
        'verbosity': 1
    }

    eval_result = {}
    model = xgb.train(xgb_params, dtrain, num_boost_round=100, evals=watchlist, evals_result=eval_result,
                      verbose_eval=10)
    y_pred = model.predict(dval)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    # add the results to the collating dictionary
    key = 'eta=%s' % xgb_params['eta']
    all_eval_results[key] = eval_result

# convert the dictionary into a dataframe
all_results = pd.json_normalize(all_eval_results, sep='_')
all_results_df = pd.DataFrame()
for col in all_results.columns:
    temp = pd.DataFrame(all_results[col].tolist()).T
    all_results_df[col] = temp
all_results_df.columns = all_results_df.columns.str.rstrip("_rmse")
all_results_df['boost_round'] = range(1, len(all_results_df)+1)
all_results_df.set_index('boost_round', inplace=True)

# compare the validation results
cols = [col for col in all_results_df.columns if "val" in col]

sns.lineplot(all_results_df[cols])
plt.xlabel("Number of trees")
plt.ylabel("auc")
plt.title("Validation set performance")
sns.despine()
