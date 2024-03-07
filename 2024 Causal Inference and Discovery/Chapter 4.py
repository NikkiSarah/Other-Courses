import networkx as nx
import numpy as np

import matplotlib as mpl

mpl_backend = mpl.get_backend()
if mpl_backend != "Qt5Agg":
    mpl.use("Qt5Agg")
else:
    pass

sample_gml = """graph [
directed 1

node [
id 0
label "0"
]

node [
id 1
label "1"
]

node [
id 2
label "2"
]

edge [
source 0
target 1
]

edge [
source 2
target 1
]
]
"""
graph = nx.parse_gml(sample_gml)
nx.draw(graph)


adj_matrix = np.array([
    [0, 1, 0],
    [0, 0, 0],
    [0, 1, 0]
])
graph = nx.DiGraph(adj_matrix)
nx.draw(graph)


practice_gml = """graph [
directed 1

node [
id 0
label "0"
]

node [
id 1
label "1"
]

node [
id 2
label "2"
]

node [
id 3
label "3"
]

node [
id 4
label "4"
]

node [
id 5
label "5"
]

edge [
source 0
target 1
]

edge [
source 0
target 3
]

edge [
source 0
target 5
]

edge [
source 3
target 2
]

edge [
source 2
target 4
]

edge [
source 4
target 5
]
]
"""
graph = nx.parse_gml(practice_gml)
nx.draw(graph)


adj_matrix = np.array([
    [0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0]
])
graph = nx.DiGraph(adj_matrix)
nx.draw(graph)
