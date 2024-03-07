#%% gcastle
import numpy as np
from castle.datasets import DAG, IIDSimulation
import networkx as nx
import matplotlib.pyplot as plt

seed=18
np.random.seed(seed)


def scale_free(n_nodes, n_edges, weight_range=None, seed=None):
    assert (n_nodes > 0 and n_edges >= n_nodes and n_edges < n_nodes * n_nodes)
    np.random.seed(seed)
    m = int(round(n_edges / n_nodes))
    G_und = nx.barabasi_albert_graph(n=n_nodes, m=m)
    B_und = np.asmatrix(nx.to_numpy_array(G_und))
    B = DAG._random_acyclic_orientation(B_und)
    if weight_range is None:
        return B
    else:
        W = DAG._BtoW(B, n_nodes, weight_range)
    return W


adj_mat = scale_free(n_nodes=17, n_edges=10, seed=18)

g = nx.DiGraph(adj_mat)
nx.draw(G=g, node_color="#00B0F0", node_size=1200, pos=nx.circular_layout(g))

#%% constraint-based causal discovery
from castle.algorithms import PC
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.common.independence_tests import CITest

pc_dag = np.array([
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])

N = 1000
p = np.random.randn(N)
q = np.random.randn(N)
r = p + q + .1 * np.random.randn(N)
s = .7 * r + .1 * np.random.randn(N)
pc_dataset = np.vstack([p, q, r, s]).T

pc = PC()
pc.learn(pc_dataset)

GraphDAG(est_dag=pc.causal_matrix, true_dag=pc_dag)
MetricsDAG(B_est=pc.causal_matrix, B_true=pc_dag).metrics

pc_stable = PC(variant='stable')
pc_parallel = PC(variant='parallel')
pc_cat = PC(ci_test='chi2')
pc_cat_alt = PC(ci_test=CITest().cressie_read)


