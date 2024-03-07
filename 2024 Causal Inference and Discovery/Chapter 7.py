from scipy import stats
import numpy as np
import pandas as pd
from dowhy import CausalModel, gcm
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
import networkx as nx

import matplotlib as mpl
mpl_backend = mpl.get_backend()
if mpl_backend != "Qt5Agg":
    mpl.use("Qt5Agg")
else:
    pass

class GPSMemorySCM:
    def __init__(self, random_seed=None):
        self.random_seed = random_seed
        self.u_x = stats.truncnorm(0, np.infty, scale=5)
        self.u_y = stats.norm(scale=2)
        self.u_z = stats.norm(scale=2)
        self.u = stats.truncnorm(0, np.infty, scale=4)

    def sample(self, sample_size=100, treatment_value=None):
        """Samples from the SCM"""
        if self.random_seed:
            np.random.seed(self.random_seed)

        u_x = self.u_x.rvs(sample_size)
        u_y = self.u_y.rvs(sample_size)
        u_z = self.u_z.rvs(sample_size)
        u = self.u.rvs(sample_size)

        if treatment_value:
            gps = np.array([treatment_value] * sample_size)
        else:
            gps = u_x + 0.7 * u

        hippocampus = -0.6 * gps + 0.25 * u_z
        memory = 0.7 * hippocampus + 0.25 * u

        return gps, hippocampus, memory

    def intervene(self, treatment_value, sample_size=100):
        """Intervenes on the SCM"""
        return self.sample(treatment_value=treatment_value, sample_size=sample_size)


scm = GPSMemorySCM()
gps_obs, hippocampus_obs, memory_obs = scm.sample(1000)
df = pd.DataFrame(np.vstack([gps_obs, hippocampus_obs, memory_obs]).T, columns=['X', 'Z', 'Y'])

# step 1 : model the problem
gml_graph = """
graph [
    directed 1
    node [
        id "X"
        label "X"
    ]
    node [
        id "Z"
        label "Z"
    ]
    node [
        id "Y"
        label "Y"
    ]
    node [
        id "U"
        label "U"
    ]
    edge [
        source "X"
        target "Z"
    ]
    edge [
        source "Z"
        target "Y"
    ]
    edge [
        source "U"
        target "X"
    ]
    edge [
        source "U"
        target "Y"
    ]
]
"""

model = CausalModel(data=df, treatment='X', outcome='Y', graph=gml_graph)
model.view_model()

# step 2: identify the estimands
estimand = model.identify_effect()
print(estimand)

# step 3: obtain the estimates
est = model.estimate_effect(identified_estimand=estimand, method_name='frontdoor.two_stage_regression')
print(f'Estimate of causal effect (linear regression): {est.value}')

# step 4: refutation tests
refute_subset = model.refute_estimate(estimand=estimand, estimate=est, method_name='data_subset_refuter',
                                      subset_fraction=0.4)
print(refute_subset)

# complete example
sample_size = 1000
S = np.random.random(sample_size)
Q = 0.2 * S + 0.67 * np.random.random(sample_size)
X = 0.14 * Q + 0.4 * np.random.random(sample_size)
Y = 0.7 * X + 0.11 * Q + 0.32 * S + 0.24 * np.random.random(sample_size)
P = 0.43 * X + 0.21 * Y + 0.22 * np.random.random(sample_size)

df = pd.DataFrame(np.vstack([S, Q, X, Y, P]).T, columns=['S', 'Q', 'X', 'Y', 'P'])

# step 1: encode the assumptions
nodes = ['S', 'Q', 'X', 'Y', 'P']
edges = ['SQ', 'SY', 'QX' , 'QY', 'XP', 'YP', 'XY']
gml_string = 'graph [directed 1\n'

for node in nodes:
    gml_string += f'\tnode [id "{node}" label "{node}"]\n'
for edge in edges:
    gml_string += f'\tedge [source "{edge[0]}" target "{edge[1]}"]\n'
gml_string += ']'
print(gml_string)

model = CausalModel(data=df, treatment='X', outcome='Y', graph=gml_string)
model.view_model()

# step 2: get the estimand
estimand = model.identify_effect()
print(estimand)

# step 3: get the estimate/s
est = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dml.DML',
                            method_params={
                                'init_params': {
                                    'model_y': GradientBoostingRegressor(),
                                    'model_t': GradientBoostingRegressor(),
                                    'model_final': LassoCV(fit_intercept=False),
                                },
                                'fit_params': {}
                            }
                            )
print(f'Estimate of causal effect (DML): {est.value}')

est_lr = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.linear_regression')
print(f'Estimate of causal effect (linear regression): {est_lr.value}')

# step 4: refute the estimate
random_cause = model.refute_estimate(estimand=estimand, estimate=est, method_name='random_common_cause')
print(random_cause)

placebo_refuter = model.refute_estimate(estimand=estimand, estimate=est, method_name='placebo_treatment_refuter')
print(placebo_refuter)

# complete example using the GCM API
graph_nx = nx.DiGraph([(edge[0], edge[1]) for edge in edges])
nx.draw(graph_nx, with_labels=True, node_size=900, font_color='white', node_color='#00B0F0')

# the invertible SCM is the only one that can generate counterfactuals without providing values for all noise variables
causal_model = gcm.InvertibleStructuralCausalModel(graph_nx)
# S is the only node without parents amongst the endogenous variables
causal_model.set_causal_mechanism('S', gcm.EmpiricalDistribution())
causal_model.set_causal_mechanism('X', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
causal_model.set_causal_mechanism('Y', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
causal_model.set_causal_mechanism('P', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
causal_model.set_causal_mechanism('Q', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))

# fit the model and estimate causal strengths
gcm.fit(causal_model, df)
print(gcm.arrow_strength(causal_model, 'Y'))

# generate counterfactuals
obs_data = pd.DataFrame(data=dict(X=[0.5], Y=[0.75], S=[0.5], Q=[0.4], P=[0.34]))
print(gcm.counterfactual_samples(causal_model, {'X': lambda x: 0.21}, observed_data=obs_data))

