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

from big_cloud_scratch import commit_query, query_ght, git_graph


def sample_initial_node(G):
    """
    Given a graph, randomly sample one of its nodes.

    :param G: the nx graph to sample from.
    :return randomly-sampled nx node.
    """
    node_list = list(G.nodes)
    return random.sample(node_list, 1)[0]


def get_random_child(G, node):
    """
    Given a graph and a node in that graph, get a random child from that node.
    :param G: the nx graph to search within.
    :param node: the nx node to start at.
    :return: the nx node child that was sampled, or None if input node has no children.
    """
    children = []
    for child, _ in G.adj[node].items():
        children.append(child)
    if len(children) == 0:
        return None
    else:
        return random.sample(children, 1)[0]


def get_sample_motif(G, k):
    """
    Given a graph, get a random motif (subgraph) of length k. Note: will start at a random nodein the graph, but will
    only return motifs that have at least k children.
    :param G: the nx graph to sample from.
    :param k: the desired length of the sampled motif.
    :return: a motif (nx subgraph) of length k.
    """
    current_node = sample_initial_node(G)
    motif_nodes = [current_node]
    for i in range(k-1):
        current_node = get_random_child(G, current_node)
        if current_node is None: # resample if this motif isnt large enough
            return get_sample_motif(G, k)
        motif_nodes.append(current_node)
    return G.subgraph(motif_nodes)


def get_motif_samples(G, k, num_samples):
    """
    Given a graph, get n random motifs (subgraphs) of length k. Note: will start at a random node in the graph, but will
    only return motifs that have at least k children.

    :param G: the nx graph to sample from.
    :param k: the desired length of the sampled motif.
    :param num_samples: how many motifs to sample from the graph.
    :return: a dictionary where the keys are motifs (nx subgraph) of length k and the keys are how many times similar
    (isomorphic) motifs occur in the graph.
    """
    graphs = []
    for i in range(num_samples):
        graphs.append(get_sample_motif(G, k))

    motifs = {}
    for n, graph in enumerate(graphs):
        if n == 0:
            motifs[graph] = 1
        already_seen = 0
        for motif in motifs.keys():
            if nx.is_isomorphic(graph, motif):
                motifs[motif] += 1
                already_seen = 1
                break
        if already_seen == 0:
            motifs[graph] = 1
    return motifs


if __name__ == '__main__':
    query_p1 = commit_query(22003900)
    data_p1 = query_ght(query_p1)
    print(get_motif_samples(git_graph(data_p1), k=10, num_samples=1000))
