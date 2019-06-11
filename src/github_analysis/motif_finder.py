"""
Sample usage (from project root dir): python src/github_analysis/motif_finder.py 0

Functions for implementing the following algo, suggested by Trevor Campbell:
To identify the K-node motifs in a graph, you could always use NetworkX and do something simple like:

Iterate:

sample a random node
pick a neighbor, add to the motif
keep adding neighbors of the motif until you reach K nodes
(if you want to keep track of frequencies:) compare to previously found motifs; increment the counter of this one by 1
"""
import sys
import random
import pickle
import logging
import time
from random import choice

import networkx as nx

from nxutils import git_graph
#from data_layer import getCommitsByProjectIds
from cluster import get_embedding_clusters

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", filename="log.log", level=logging.INFO)


def main(random_state=None):
    """Function that runs the cluster and gets the motif of each cluster."""
    clusters = get_embedding_clusters(random_state=random_state)
    get_motifs_by_cluster(clusters, data_layer)


class MotifFinder:
    def __init__(self, G):
        """
        Object for dealing with motifs within networkx graphs.

        :param G: the nx graph to get motifs from.
        """
        self.G = G
        self.node_list = list(G.nodes)

    def sample_initial_node(self):  # TODO: let users pick a random state
        """
        Given a graph, randomly sample one of its nodes.

        :param G: the nx graph to sample from.
        :return randomly-sampled nx node.
        """
        return choice(self.node_list)

    def get_random_child(self, node):
        """
        Given a graph and a node in that graph, get a random child from that node. NOT CURRENTLY USED FOR ANYTHING!
        :param G: the nx graph to search within.
        :param node: the nx node to start at.
        :return: the nx node child that was sampled, or None if input node has no children.
        """
        children = []
        for child, _ in self.G.adj[node].items():
            children.append(child)
        if len(children) == 0:
            return None
        else:
            return random.sample(children, 1)[0]

    def get_sample_motif(self, k, recursion_limit=5000):
        """
        Given a graph, get a random motif (subgraph) of length k. Note: will start at a random nodein the graph, but will
        only return motifs that have at least k children.

        :param k: the desired length of the sampled motif.
        :param recursion_limit: how many times to recurse (in this case, to keep sampling). NB: This sets recursion at
        the sys level, and the function is using recursion in kinda a weird way, not sure how cool this is.
        :return: a motif (nx subgraph) of length k.
        """
        sys.setrecursionlimit(recursion_limit)
        root = self.sample_initial_node()
        edges = nx.bfs_edges(self.G, root) # https://networkx.github.io/documentation/networkx-2.2/reference/algorithms/generated/networkx.algorithms.traversal.breadth_first_search.bfs_edges.html#networkx.algorithms.traversal.breadth_first_search.bfs_edges
        nodes = [root] + [v for u, v in edges]
        if len(nodes) >= k:
            return nx.DiGraph(self.G.subgraph(nodes[:k]))
        else: # resample if this motif isnt large enough
            return self.get_sample_motif(k)

    def get_motif_samples(self, k, num_samples):
        """
        Given a graph, get n random motifs (subgraphs) of length k and associate identical motifs together. #TODO: seperate into two functions?
        Note: will start at a random node in the graph, but will only return motifs that have at least k children.

        :param k: the desired length of the sampled motif.
        :param num_samples: how many motifs to sample from the graph.
        :return: a dictionary where the keys are motifs (nx subgraph) of length k and the values are how many times similar
        (isomorphic) motifs occur in the graph.
        """
        graphs = []
        for i in range(num_samples):
            graphs.append(self.get_sample_motif(k))

        motifs = {}
        for n, graph in enumerate(graphs):
            if n == 0:
                motifs[graph] = 1
                continue
            already_seen = 0
            for motif in motifs.keys():
                if nx.is_isomorphic(graph, motif):
                    motifs[motif] += 1
                    already_seen = 1
                    break
            if already_seen == 0:
                motifs[graph] = 1
        return motifs


def get_motifs(github_project_ids, k_for_motifs, number_of_samples, data_layer):
    """Given a list of github prof"""
    # Get graph for this cluster TODO: update to pull from pickle of project graphs
    projects_cluster = data_layer.getCommitsByProjectIds(github_project_ids)
    G = git_graph(projects_cluster)
    mf = MotifFinder(G)  # Instantiate MotifFinder object looking at that cluster's graph

    # Trying to pull out the motifs of each cluster here. Need error handling for clusters where we can't pull out
    # common motifs (e.g. can't build motifs of length k because there aren't many subgraphs at least k long)
    try:
        motifs = mf.get_motif_samples(k_for_motifs, number_of_samples)  # Get most common motifs for that cluster
    except RecursionError:
        logging.info('Graph has too many short paths.')
        return None
    except ValueError:
        logging.info('Graph has no connections.')
        return None
    return motifs


def get_motifs_by_cluster(clusters, data_layer, k_for_motifs=5, number_of_samples=1000, output_file='./results/motifs_by_cluster.pickle'):
    """
    A way to take in a group of GitHub project clusters and output their most common motifs. For each cluster, get most common subgraphs

    :param clusters: a dictionary where the keys are the cluster labels and the values are lists of GitHub
                    projectIds that fall in that cluster. Note that ids need to be in the graph. EVENTUALLY GOING TO HOOK TO PROJECT GRAPH
    :param k_for_motifs: the desired length of the sampled motifs.
    :param number_of_samples: how many motifs to sample from the graph.
    :param output_file: string with the filename to output the results to as a pickle. If this param is set to None, no file will be outputted.
    :return: None.
    """
    motifs_by_clusters = {}
    for cluster in clusters:
        cluster_motif = get_motifs(clusters[cluster], k_for_motifs, number_of_samples, data_layer)
        if cluster_motif is not None:
            motifs_by_clusters[cluster] = cluster_motif

    if output_file is not None:
        with open(output_file, 'wb') as output:
            pickle.dump(motifs_by_clusters, output)
        logging.info('Cluster file outputted!')

    return motifs_by_clusters


if __name__ == '__main__':
    main()
# TODO: this is generating 1 more k than specified, idk why.
# TODO: output tsne graph with k-means labels.



# def get_most_common_motifs_by_cluster(clusters, k_for_motifs=10, number_of_samples=1000,output_file='./results/motifs_by_cluster.pickle'):#output_folder_suffix='results'):
#     """
#     A way to take in a group of GitHub project clusters and output their most common motifs. For each cluster, get most common subgraphs
#
#     :param clusters: a dictionary where the keys are the cluster labels and the values are lists of GitHub
#                     projectIds that fall in that cluster. Note that ids need to be in the graph. EVENTUALLY GOING TO HOOK TO PROJECT GRAPH
#     :param k_for_motifs: the desired length of the sampled motifs.
#     :param number_of_samples: how many motifs to sample from the graph.
#     :param output_folder_suffix: suffix to put on end of output folder.
#     :return: None.
#     """
#     motifs_by_clusters = {}
#     sorted_cluster_keys = list(clusters.keys())
#     sorted_cluster_keys.sort()
#     # I either need to subset graph here or do this outside of this class
#     for cluster in sorted_cluster_keys:
#         try:
#             motifs = self.get_motif_samples(k_for_motifs, number_of_samples)
#         except RecursionError:
#             print('too many short paths for Cluster {}, no file outputted.'.format(cluster))
#             continue
#         except ValueError:
#             print('Cluster {} has no connections'.format(cluster))
#             continue
#         motifs_by_clusters[cluster] = motifs
#     return motifs_by_clusters

    # query_p1 = commit_query(33470153)
    # data_p1 = query_ght(query_p1)
    # motifs = get_motif_samples(git_graph(data_p1), k=10, num_samples=1000)
    # visualize_motif_samples(motifs, 'imgs/commonly_occurring_motifs_proj_15059440_10.pdf')
    # clusters = get_embedding_clusters(random_state=1)
    #
    # projects_cluster = getCommitsByProjectIds(clusters[cluster])
    # G = git_graph(projects_cluster)
    # get_most_common_motifs_from_clusters(clusters, k_for_motifs=5)

    #    get_most_common_motifs_from_clusters(clusters, k_for_motifs=i, output_folder_suffix='motif_size_is_' + str(i))
    # for i in range(2,102,10):
    #   get_most_common_motifs_from_clusters(clusters, k_for_motifs=i, output_folder_suffix='motif_size_is_' + str(i))
