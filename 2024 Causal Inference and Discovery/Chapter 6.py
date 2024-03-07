# front-door
import numpy as np
import networkx as nx
import graphviz
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import matplotlib as mpl

mpl_backend = mpl.get_backend()
if mpl_backend != "Qt5Agg":
    mpl.use("Qt5Agg")
else:
    pass

adj_mat = np.array([
    [0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
])
graph = nx.DiGraph(adj_mat)

num_nodes = adj_mat.shape[0]
vars = ['U', 'X', 'Y', 'Z', 'U_x', 'U_y', 'U_z']
var_names = ['Motivation', 'GPS usage', 'Spatial memory', 'Hippocampus volume', '', '', '']
for i in range(num_nodes):
    graph.nodes[i]['var'] = vars[i]
    graph.nodes[i]['var_name'] = var_names[i]
print(graph.nodes.data())

label_dict = {0: 'U', 1: 'X', 2: 'Y', 3: 'Z', 4: 'U_x', 5: 'U_y', 6: 'U_z'}
graph = nx.relabel_nodes(graph, label_dict)

options = {
    'node_color': 'grey',
    'node_size': 1000,
    'width': 1,
}
nx.draw(graph, with_labels=True, font_weight='bold', pos=nx.planar_layout(graph), **options)

# alternatively
graph = graphviz.Digraph(format='png', engine='neato')
nodes = ['U\nMotivation', 'X\nGPS usage', 'Z\nHippocampus volume', 'Y\nSpatial memory', 'U_x', 'U_z', 'U_y']
positions = ['3,1.5!', '0,0!', '3,0!', '6,0!', '0,-1.5!', '3,-1.5!', '6,-1.5!']

[graph.node(n, pos=pos) for n, pos in zip(nodes, positions)]

graph.node('U\nMotivation', style='dashed')
graph.node('U_x', style='dashed')
graph.node('U_z', style='dashed')
graph.node('U_y', style='dashed')

graph.edge('U\nMotivation', 'X\nGPS usage')
graph.edge('U\nMotivation', 'Y\nSpatial memory')
graph.edge('X\nGPS usage', 'Z\nHippocampus volume')
graph.edge('Z\nHippocampus volume', 'Y\nSpatial memory')
graph.edge('U_x', 'X\nGPS usage', style='dashed')
graph.edge('U_y', 'Y\nSpatial memory', style='dashed')
graph.edge('U_z', 'Z\nHippocampus volume', style='dashed')

graph.render(f'./img/ch06_gps', view=True)

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
gps_obs, hippocampus_obs, memory_obs = scm.sample(500)

treatments = []
exp_results = []
for treatment in np.arange(1, 21):
    gps_hours, hippocampus, memory = scm.intervene(treatment_value=treatment, sample_size=30)
    exp_results.append(memory)
    treatments.append(gps_hours)

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
axs[0].scatter(gps_obs, memory_obs, alpha=0.2)
axs[0].set_xlabel('GPS usage')
axs[0].set_ylabel('Spatial memory change')
axs[0].set_title('Observational')
axs[1].scatter(treatments, exp_results, alpha=0.2)
axs[1].set_xlabel('GPS usage')
axs[1].set_title('Interventional')
fig.tight_layout()

lr_naive = LinearRegression()
lr_naive.fit(X=gps_obs.reshape(-1, 1), y=memory_obs)

treatments_unpack = np.array(treatments).flatten()
results_unpack = np.array(exp_results).flatten()
lr_exp = LinearRegression()
lr_exp.fit(X=treatments_unpack.reshape(-1, 1), y=results_unpack)

X_test = np.arange(1, 21).reshape(-1, 1)
preds_naive = lr_naive.predict(X_test)
preds_exp = lr_exp.predict(X_test)

plt.scatter(treatments, exp_results, alpha=0.2)
plt.plot(X_test, preds_exp, label="Experimental", color="red")
plt.plot(X_test, preds_naive, label="Naive", color="blue")
plt.xlabel('GPS usage')
plt.ylabel('Spatial memory change')
plt.legend()

print(f'Naive model:\n{lr_naive.coef_}\n')
print(f'Experimental model:\n{lr_exp.coef_}\n')

lr_zx = LinearRegression()
lr_zx.fit(X=gps_obs.reshape(-1, 1), y=hippocampus_obs)
lr_yxz = LinearRegression()
lr_yxz.fit(X=np.array([gps_obs, hippocampus_obs]).T, y=memory_obs)
print(lr_zx.coef_[0] * lr_yxz.coef_[1])
