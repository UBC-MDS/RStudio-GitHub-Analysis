"""
From Trevor Campbell:
To identify the K-node motifs in a graph, you could always use NetworkX and do something simple like:

Iterate:

sample a random node
pick a neighbor, add to the motif
keep adding neighbors of the motif until you reach K nodes
(if you want to keep track of frequencies:) compare to previously found motifs; increment the counter of this one by 1
"""
import random

import pandas as pd
import networkx as nx

from big_cloud_scratch import commit_query,query_ght


def sample_initial_node(G):
    """G: the graph to sample from."""
    node_list = list(G.nodes)
    return random.sample(node_list, 1)[0]


def get_random_child(G, node):
    children = []
    for child, _ in G.adj[node].items():
        children.append(child)
    return random.sample(children, 1)[0]


def get_sample_motif(G, k):
    current_node = sample_initial_node(G)
    motif_nodes = [current_node]
    for i in range(k-1):
        current_node = get_random_child(G, current_node)
        motif_nodes.append(current_node)
    return G.subgraph(motif_nodes)


# query_p1 = commit_query(22003900)
#
# data_p1 = query_ght(query_p1)
# data_p1.to_csv('example_data.csv')

