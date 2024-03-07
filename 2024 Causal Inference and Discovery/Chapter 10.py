import pandas as pd
from dowhy import CausalModel
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
import json
from sklearn.model_selection import train_test_split
from econml.metalearners import SLearner, XLearner, TLearner
from econml.dml import LinearDML, CausalForestDML
from econml.dr import DRLearner
import time
from tqdm import tqdm

import matplotlib as mpl
mpl_backend = mpl.get_backend()
if mpl_backend != "Qt5Agg":
    mpl.use("Qt5Agg")
else:
    pass

#%% doubly-robust learners
earnings_train = pd.read_csv('./data/ml_earnings_interaction_train.csv')
earnings_test = pd.read_csv('./data/ml_earnings_interaction_test.csv')

print(earnings_train.shape, earnings_test.shape)
print(earnings_train.head())
print(earnings_test.head())

nodes = ['took_a_course', 'earnings', 'age', 'python_proficiency']
edges = [('took_a_course', 'earnings'),
         ('age', 'took_a_course'),
         ('age', 'earnings'),
         ('python_proficiency', 'earnings')]
gml_string = 'graph [directed 1\n'
for node in nodes:
    gml_string += f'\tnode [id "{node}" label "{node}"]\n'
for edge in edges:
    gml_string += f'\tedge [source "{edge[0]}" target "{edge[1]}"]\n'
gml_string += ']'

model = CausalModel(data=earnings_train, treatment='took_a_course', outcome='earnings',
                    effect_modifiers='python_proficiency', graph=gml_string)
model.view_model()

estimand = model.identify_effect()
print(estimand)

est = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dr.LinearDRLearner',
                            target_units='ate',
                            method_params={
                                'init_params': {
                                    'model_propensity': LogisticRegression(),
                                    'model_regression': LGBMRegressor(n_estimators=1000, max_depth=10)
                                },
                                'fit_params': {}
                            })

earnings_test2 = earnings_test.drop(['true_effect', 'took_a_course'], axis=1)
est_test = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dr.LinearDRLearner',
                                 target_units=earnings_test2, fit_estimator=False, method_params={})

effect_pred = est_test.cate_estimates.flatten()
effect_true = earnings_test.true_effect.values
print(mean_absolute_percentage_error(effect_true, effect_pred))


def plot_effect(true_effect, predicted_effect):
    plt.scatter(true_effect, predicted_effect, color='#00B0F0')
    plt.plot(np.sort(true_effect), np.sort(true_effect), color='#FF0000', alpha=0.7, label='Perfect model')
    plt.xlabel('True effect', alpha=0.5)
    plt.ylabel('Predicted effect', alpha=0.5)
    plt.legend()


plot_effect(effect_true, effect_pred)

# repeat with a more complicated model
est = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dr.DRLearner',
                            target_units='ate',
                            method_params={
                                'init_params': {
                                    'model_propensity': LogisticRegression(),
                                    'model_regression': LGBMRegressor(n_estimators=1000, max_depth=10),
                                    'model_final': LGBMRegressor(n_estimators=500, max_depth=10)
                                },
                                'fit_params': {}
                            })

est_test = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dr.DRLearner',
                                 target_units=earnings_test2, fit_estimator=False, method_params={})

effect_pred = est_test.cate_estimates.flatten()
print(mean_absolute_percentage_error(effect_true, effect_pred))

plot_effect(effect_true, effect_pred)


#%% doubly-robust ML
est = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dml.LinearDML',
                            target_units='ate',
                            method_params={
                                'init_params': {
                                    'model_y': LGBMRegressor(n_estimators=500, max_depth=10),
                                    'model_t': LogisticRegression(),
                                    'discrete_treatment': True
                                },
                                'fit_params': {}
                            })

est_test = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dml.LinearDML',
                                 target_units=earnings_test2, fit_estimator=False, method_params={})

effect_pred = est_test.cate_estimates.flatten()
print(mean_absolute_percentage_error(effect_true, effect_pred))

plot_effect(effect_true, effect_pred)

# reduce the complexity of the outcome model and increase the number of cross-fitting folds
est = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dml.LinearDML',
                            target_units='ate',
                            method_params={
                                'init_params': {
                                    'model_y': LGBMRegressor(n_estimators=50, max_depth=10),
                                    'model_t': LogisticRegression(),
                                    'discrete_treatment': True,
                                    'cv': 4
                                },
                                'fit_params': {}
                            })

est_test = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dml.LinearDML',
                                 target_units=earnings_test2, fit_estimator=False, method_params={})

effect_pred = est_test.cate_estimates.flatten()
print(mean_absolute_percentage_error(effect_true, effect_pred))

plot_effect(effect_true, effect_pred)

# tune hyperparameters
model_y = GridSearchCV(
    estimator=LGBMRegressor(),
    param_grid={
        'max_depth': [3, 10, 20, 100],
        'n_estimators': [10, 50, 100]
    },
    cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
)

model_t = GridSearchCV(
    estimator=LGBMClassifier(),
    param_grid={
        'max_depth': [3, 10, 20, 100],
        'n_estimators': [10, 50, 100]
    },
    cv=10, n_jobs=-1, scoring='accuracy'
)

est = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dml.LinearDML',
                            target_units='ate',
                            method_params={
                                'init_params': {
                                    'model_y': model_y,
                                    'model_t': model_t,
                                    'discrete_treatment': True,
                                    'cv': 4
                                },
                                'fit_params': {}
                            })

est_test = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dml.LinearDML',
                                 target_units=earnings_test2, fit_estimator=False, method_params={})

effect_pred = est_test.cate_estimates.flatten()
print(mean_absolute_percentage_error(effect_true, effect_pred))

plot_effect(effect_true, effect_pred)


#%% causal forests
est = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dml.CausalForestDML',
                            target_units='ate',
                            method_params={
                                'init_params': {
                                    'model_y': LGBMRegressor(n_estimators=50, max_depth=10),
                                    'model_t': LGBMClassifier(n_estimators=50, max_depth=10),
                                    'discrete_treatment': True,
                                    'cv': 4
                                },
                                'fit_params': {}
                            })

est_test = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dml.CausalForestDML',
                                 target_units=earnings_test2, fit_estimator=False, method_params={})

effect_pred = est_test.cate_estimates.flatten()
print(mean_absolute_percentage_error(effect_true, effect_pred))

plot_effect(effect_true, effect_pred)


#%% heterogeneous treatment effects with experimental data
hillstrom_clean = pd.read_csv('./data/hillstrom_clean.csv')

with open('./data/hillstrom_clean_label_mapping.json', 'r') as f:
    hillstrom_labels_mapping = json.load(f)

print(hillstrom_clean.head())

hillstrom_clean = hillstrom_clean.drop(['zip_code__urban', 'channel__web'], axis=1)

# test the validity of the randomisation/data collection process
hillstrom_X = hillstrom_clean.drop(['visit', 'conversion', 'spend', 'treatment'], axis=1)
hillstrom_y = hillstrom_clean.spend
hillstrom_T = hillstrom_clean.treatment

sample_size = hillstrom_clean.shape[0]
print(sample_size)
print(hillstrom_T.value_counts(normalize=True))

X_train_eda, X_test_eda, T_train_eda, T_test_eda = train_test_split(hillstrom_X, hillstrom_T, test_size=0.5)
print(T_test_eda.value_counts(normalize=True))

lgbm_eda = LGBMClassifier()
lgbm_eda.fit(X_train_eda, T_train_eda)

T_pred_eda = lgbm_eda.predict(X_test_eda)
print(accuracy_score(T_test_eda, T_pred_eda))

random_scores = []
test_eda_sample_size = T_test_eda.shape[0]
for i in range(10000):
    random_scores.append(
        (np.random.choice([0, 1, 2], test_eda_sample_size) == np.random.choice([0, 1, 2], test_eda_sample_size)).mean())
print(np.quantile(random_scores, 0.025), np.quantile(random_scores, 0.975))

plt.hist(random_scores, bins=100, color='#00B0F0', alpha=0.7, label='Random models (n=10e3)')
plt.axvline(accuracy_score(T_test_eda, T_pred_eda), color='#FF0000', ls='--', label='Model accuracy')
plt.fill_betweenx(x1=np.quantile(random_scores, 0.025), x2=np.quantile(random_scores, 0.975), y=np.arange(0, 300),
                  color='#00B0F0', alpha=0.1, label='95% empirical CI')
plt.legend()


#%% kevin's challenge
# if you could eliminate 10000 customers from the campaign, which ones would they be?
print(hillstrom_labels_mapping)
print((hillstrom_y[hillstrom_T > 0] > 0).sum())


def create_model(model_type, n_estimators=100, max_depth=10, learning_rate=0.01):
    if model_type == "regressor":
        return LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    elif model_type == "classifier":
        return LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    else:
        raise NotImplementedError(f'Model type `{model_type}` not implemented.')


s_learner = SLearner(overall_model=create_model("regressor"))
x_learner = XLearner(models=[create_model("regressor"), create_model("regressor"), create_model("regressor")],
                     cate_models=[create_model("regressor"), create_model("regressor"), create_model("regressor")])
t_learner = TLearner(models=[create_model("regressor"), create_model("regressor"), create_model("regressor")])
dml = LinearDML(model_y=create_model("regressor"), model_t=create_model("classifier"), discrete_treatment=True, cv=5)
dr = DRLearner(model_propensity=LogisticRegression(), model_regression=create_model("regressor"),
               model_final=create_model("regressor"))
cf = CausalForestDML(model_y=create_model("regressor"), model_t=create_model("classifier"), discrete_treatment=True,
                     cv=5)

X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(hillstrom_X, hillstrom_y, hillstrom_T,
                                                                     test_size=0.5)
print((y_train[T_train > 0] > 0).sum())
print((y_test[T_test > 0] > 0).sum())

models = {'SLearner': s_learner,
          'TLearner': t_learner,
          'XLearner': x_learner,
          'DRLearner': dr,
          'LinearDML': dml,
          'CausalForestDML': cf
          }
model_times = []
for model_name, model in models.items():
    start = time.time()
    model.fit(Y=y_train, T=T_train, X=X_train, inference='bootstrap')
    stop = time.time()
    model_times.append(f'{model_name} fitted in {stop - start:.4f} seconds.')
print(pd.DataFrame(model_times))

effects_train = {'treatment_1': {}, 'treatment_2': {}}
effects_test = {'treatment_1': {}, 'treatment_2': {}}
for treatment in [1, 2]:
    for model_name, model in tqdm(models.items()):
        effects_local_train = models[model_name].effect(X_train.values, T0=0, T1=treatment)
        effects_train[f'treatment_{treatment}'][model_name] = effects_local_train

        effects_local_test = models[model_name].effect(X_test.values, T0=0, T1=treatment)
        effects_test[f'treatment_{treatment}'][model_name] = effects_local_test


def get_uplift_by_decile(uplifts, t_true, t_pred, y_true):
    all_data = pd.DataFrame(
        dict(
            uplifts=uplifts,
            y_true=y_true,
            t_true=t_true)
    ).query(f't_true == 0 | t_true == {t_pred}').sort_values('uplifts')

    all_data['deciles'] = pd.qcut(all_data.uplifts, q=10, duplicates='drop')
    all_data['deciles'] = pd.factorize(all_data.deciles, sort=True)[0]

    mean_decile_resp = all_data.groupby(['deciles', 't_true']).mean()

    if len(mean_decile_resp) == 20:
        mean_decile_resp['true_uplift'] = mean_decile_resp.y_true * np.array([-1, 1] * 10)
    else:
        mean_decile_resp['true_uplift'] = mean_decile_resp.y_true * np.array([-1, 1] * 9)
    true_uplift = mean_decile_resp.groupby(level=[0]).sum()['true_uplift']

    return true_uplift[::-1]

i = 1
for model_name in models.keys():
    uplifts_by_decile = {'treatment_1': {}, 'treatment_2': {}}
    global_min = np.inf
    global_max = -np.inf

    for treatment in ['treatment_1', 'treatment_2']:
        uplift_by_decile_train = get_uplift_by_decile(uplifts=effects_train[treatment][model_name], t_true=T_train,
                                                      t_pred=int(treatment.split('_')[-1]), y_true=y_train)
        uplift_by_decile_test = get_uplift_by_decile(uplifts=effects_test[treatment][model_name], t_true=T_test,
                                                     t_pred=int(treatment.split('_')[-1]), y_true=y_test)

        if i == 1:
            plt.subplot(6, 4, i)
            plt.bar(np.arange(9), uplift_by_decile_train, color='#00B0F0')
            plt.title(f'{model_name} {treatment} - Train')

            plt.subplot(6, 4, i+1)
            plt.bar(np.arange(9), uplift_by_decile_test, color='#FF0000')
            plt.title(f'{model_name} {treatment} - Test')
        else:
            plt.subplot(6, 4, i)
            plt.bar(np.arange(10), uplift_by_decile_train, color='#00B0F0')
            plt.title(f'{model_name} {treatment} - Train')

            plt.subplot(6, 4, i + 1)
            plt.bar(np.arange(10), uplift_by_decile_test, color='#FF0000')
            plt.title(f'{model_name} {treatment} - Test')

        plt.tight_layout(pad=0.01)
        i += 2


def get_effects_argmax(effects_arrays, return_matrix=False):
    """Returns argmax for each row of predicted effects for the arbitrary no. of treatments.

    :param effects_arrays: A list of arrays for K treatments, where K>=1 (without control null effects)
    :type effects_arrays: list of np.arrays

    :param return_matrix: Determines if the function returns a matrix of all effects
        (with added null effect for control)
    :type return_matrix: bool

    ...
    :return: A stacked matrix of all effects with added column for control effects (which is always 0)
    :rtype: np.array
    """

    n_rows = effects_arrays[0].shape[0]
    null_effect_array = np.zeros(n_rows)
    stacked = np.stack([null_effect_array] + effects_arrays).T

    if return_matrix:
        return np.argmax(stacked, axis=1), stacked

    return np.argmax(stacked, axis=1)


def get_expected_response(y_true, t_true, effects_argmax):
    """Computes the average expected response for an uplift model according to the formula
        proposed by:
        Zhao, Y., Fang, X., & Simchi-Levi, D. (2017). Uplift Modeling with Multiple Treatments and General Response Types.
        Proceedings of the 2017 SIAM International Conference on Data Mining, 588-596.
        Society for Industrial and Applied Mathematics.
    """

    proba_t = pd.Series(t_true).value_counts() / np.array(t_true).shape[0]
    treatments = proba_t.index.values

    z_vals = 0

    for treatment in treatments:
        h_indicator = effects_argmax == treatment
        t_indicator = t_true == treatment
        t_proba_local = proba_t[treatment]

        z_vals += (1 / t_proba_local) * y_true * h_indicator * t_indicator

    return z_vals.mean()


print('Expecetd response on train:\n')
for model_name in models:
    effects_argmax = get_effects_argmax(
        [
            effects_train['treatment_1'][model_name],
            effects_train['treatment_2'][model_name]
        ]
    )

    expected_response = get_expected_response(
        y_true=y_train,
        t_true=T_train,
        effects_argmax=effects_argmax
    )

    print(f'{model_name}: {expected_response}')

print('\n' + '-' * 30)

print('Expected response on test:\n')
for model_name in models:
    effects_argmax = get_effects_argmax(
        [
            effects_test['treatment_1'][model_name],
            effects_test['treatment_2'][model_name]
        ]
    )

    expected_response = get_expected_response(
        y_true=y_test,
        t_true=T_test,
        effects_argmax=effects_argmax
    )

    print(f'{model_name}: {expected_response}')

print(models['LinearDML'].effect_interval(X=X_test, T0=0, T1=1))
ints = np.stack(models['LinearDML'].effect_interval(X=X_test, T0=0, T1=1, alpha=0.05)).T
perc_ints = (np.sign(ints[:, 0]) == np.sign(ints[:, 1])).sum() / ints.shape[0]
print(perc_ints)
