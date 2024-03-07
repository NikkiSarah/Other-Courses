import pandas as pd
from dowhy import CausalModel
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
mpl_backend = mpl.get_backend()
if mpl_backend != "Qt5Agg":
    mpl.use("Qt5Agg")
else:
    pass

earnings_data = pd.read_csv(r'./data/ml_earnings.csv')
print(earnings_data.head())
print(earnings_data.groupby(['age', 'took_a_course']).mean())

#%% naive estimate of causal effect of training on earnings
treatment_avg = earnings_data.groupby('took_a_course')['earnings'].mean().iloc[1]
control_avg = earnings_data.groupby('took_a_course')['earnings'].mean().iloc[0]
print(treatment_avg - control_avg)

#%% estimate of causal effect using matching
nodes = ['took_a_course', 'earnings', 'age']
edges = [('took_a_course', 'earnings'),
         ('age', 'took_a_course'),
         ('age', 'earnings')]
gml_string = 'graph [directed 1\n'
for node in nodes:
    gml_string += f'\tnode [id "{node}" label "{node}"]\n'
for edge in edges:
    gml_string += f'\tedge [source "{edge[0]}" target "{edge[1]}"]\n'
gml_string += ']'

model = CausalModel(data=earnings_data, treatment='took_a_course', outcome='earnings', graph=gml_string, common_causes='age')
model.view_model()

estimand = model.identify_effect()
print(estimand)

est_matching = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.distance_matching',
                                     target_units='ate', method_params={'distance_metric': 'minkowski', 'p': 2})
print(est_matching.value)

refutation = model.refute_estimate(estimand=estimand, estimate=est_matching, method_name='random_common_cause')
print(refutation)

#%% estimate of causal effect using IPTW
est_iptw = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.propensity_score_weighting',
                                 target_units='ate', method_params={"weighting_scheme": "ips_weight"})
print(est_iptw.value)

refutation = model.refute_estimate(estimand=estimand, estimate=est_iptw, method_name='random_common_cause')
print(refutation)


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

est = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.metalearners.SLearner',
                            target_units='ate',
                            method_params={
                                'init_params': {
                                    'overall_model': LGBMRegressor(n_estimators=500, max_depth=10)
                                },
                                'fit_params': {}
                            })
print(est.cate_estimates.mean())

refutation = model.refute_estimate(estimand=estimand, estimate=est, method_name='random_common_cause')
print(refutation)

refutation = model.refute_estimate(estimand=estimand, estimate=est, method_name='placebo_treatment_refuter')
print(refutation)

# predict on the test data
earnings_test2 = earnings_test.drop(['true_effect', 'took_a_course'], axis=1)
est_test = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.metalearners.SLearner',
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

#%% repeat on a small dataset
model_small = CausalModel(data=earnings_train.sample(100), treatment='took_a_course', outcome='earnings',
                          effect_modifiers='python_proficiency', graph=gml_string)

estimand = model_small.identify_effect()
print(estimand)

est = model_small.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.metalearners.SLearner',
                                  target_units='ate',
                                  method_params={
                                      'init_params': {
                                          'overall_model': LGBMRegressor(n_estimators=500, max_depth=10)
                                      },
                                      'fit_params': {}
                                  })
print(est.cate_estimates.mean())

refutation = model_small.refute_estimate(estimand=estimand, estimate=est, method_name='random_common_cause')
print(refutation)

refutation = model_small.refute_estimate(estimand=estimand, estimate=est, method_name='placebo_treatment_refuter')
print(refutation)

earnings_test2 = earnings_test.drop(['true_effect', 'took_a_course'], axis=1)
est_test = model_small.estimate_effect(identified_estimand=estimand,
                                       method_name='backdoor.econml.metalearners.SLearner', target_units=earnings_test2,
                                       fit_estimator=False, method_params={})

effect_pred = est_test.cate_estimates.flatten()
effect_true = earnings_test.true_effect.values
print(mean_absolute_percentage_error(effect_true, effect_pred))

plot_effect(effect_true, effect_pred)

#%% implement a t-learner
model = CausalModel(data=earnings_train, treatment='took_a_course', outcome='earnings',
                    effect_modifiers='python_proficiency', graph=gml_string)

estimand = model.identify_effect()
print(estimand)

est = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.metalearners.TLearner',
                            target_units='ate',
                            method_params={
                                'init_params': {
                                    'models': [LGBMRegressor(n_estimators=200, max_depth=10),
                                               LGBMRegressor(n_estimators=200, max_depth=10)
                                               ],
                                },
                                'fit_params': {}
                            })
print(est.cate_estimates.mean())

earnings_test2 = earnings_test.drop(['true_effect', 'took_a_course'], axis=1)
est_test = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.metalearners.TLearner',
                                 target_units=earnings_test2, fit_estimator=False, method_params={})

effect_pred = est_test.cate_estimates.flatten()
effect_true = earnings_test.true_effect.values
print(mean_absolute_percentage_error(effect_true, effect_pred))

plot_effect(effect_true, effect_pred)

#%% implement an x-learner
model = CausalModel(data=earnings_train, treatment='took_a_course', outcome='earnings',
                    effect_modifiers='python_proficiency', graph=gml_string)

estimand = model.identify_effect()
print(estimand)

est = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.metalearners.XLearner',
                            target_units='ate',
                            method_params={
                                'init_params': {
                                    'models': [LGBMRegressor(n_estimators=50, max_depth=10),
                                               LGBMRegressor(n_estimators=50, max_depth=10)
                                               ],
                                    'cate_models': [LGBMRegressor(n_estimators=50, max_depth=10),
                                                    LGBMRegressor(n_estimators=50, max_depth=10)
                                                    ],
                                },
                                'fit_params': {}
                            })

# alternatively for identical models...
est_alt = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.metalearners.XLearner',
                                target_units='ate',
                                method_params={
                                    'init_params': {
                                        'models': LGBMRegressor(n_estimators=50, max_depth=10),
                                        },
                                    'fit_params': {}
                                })
print(est_alt.cate_estimates.mean())

earnings_test2 = earnings_test.drop(['true_effect', 'took_a_course'], axis=1)
est_test = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.metalearners.XLearner',
                                 target_units=earnings_test2, fit_estimator=False, method_params={})

effect_pred = est_test.cate_estimates.flatten()
effect_true = earnings_test.true_effect.values
print(mean_absolute_percentage_error(effect_true, effect_pred))

plot_effect(effect_true, effect_pred)

# repeat on a small dataset
model = CausalModel(data=earnings_train.sample(100), treatment='took_a_course', outcome='earnings',
                    effect_modifiers='python_proficiency', graph=gml_string)

estimand = model.identify_effect()
print(estimand)

est = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.metalearners.XLearner',
                            target_units='ate',
                            method_params={
                                'init_params': {
                                    'models': LGBMRegressor(n_estimators=50, max_depth=10),
                                    },
                                'fit_params': {}
                            })
print(est.cate_estimates.mean())

earnings_test2 = earnings_test.drop(['true_effect', 'took_a_course'], axis=1)
est_test = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.metalearners.XLearner',
                                 target_units=earnings_test2, fit_estimator=False, method_params={})

effect_pred = est_test.cate_estimates.flatten()
effect_true = earnings_test.true_effect.values
print(mean_absolute_percentage_error(effect_true, effect_pred))

plot_effect(effect_true, effect_pred)
