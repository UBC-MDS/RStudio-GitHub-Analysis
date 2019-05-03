"""
Sample usage (from project root dir): python src/motif_finder.py

Functions for implementing the following algo, suggested by Trevor Campbell:
To identify the K-node motifs in a graph, you could always use NetworkX and do something simple like:

Iterate:

sample a random node
pick a neighbor, add to the motif
keep adding neighbors of the motif until you reach K nodes
(if you want to keep track of frequencies:) compare to previously found motifs; increment the counter of this one by 1
"""
import sys
from os import makedirs
import random

import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans

from big_cloud_scratch import commit_query, query_ght, git_graph
from data_layer import getCommitsByProjectIds


sys.setrecursionlimit(5000)


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
    root = sample_initial_node(G)
    edges = nx.bfs_edges(G, root) #https://networkx.github.io/documentation/networkx-2.2/reference/algorithms/generated/networkx.algorithms.traversal.breadth_first_search.bfs_edges.html#networkx.algorithms.traversal.breadth_first_search.bfs_edges
    nodes = [root] + [v for u, v in edges]
    if len(nodes) < k: # resample if this motif isnt large enough
        return get_sample_motif(G, k)
    else:
        return G.subgraph(nodes[:k])

    # current_node = sample_initial_node(G)
    # motif_nodes = [current_node]
    # for i in range(k-1):
    #     current_node = get_random_child(G, current_node)
    #
    #     if current_node is None: # resample if this motif isnt large enough
    #         return get_sample_motif(G, k)
    #     motif_nodes.append(current_node)
    # return G.subgraph(motif_nodes)


def get_motif_samples(G, k, num_samples):
    """
    Given a graph, get n random motifs (subgraphs) of length k. Note: will start at a random node in the graph, but will
    only return motifs that have at least k children.

    :param G: the nx graph to sample from.
    :param k: the desired length of the sampled motif.
    :param num_samples: how many motifs to sample from the graph.
    :return: a dictionary where the keys are motifs (nx subgraph) of length k and the values are how many times similar
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


def visualize_motif_samples(motifs, output_file):
    """
    Given a sample of motifs, output a file with their graphs and how often they occurred.

    :param motifs: a dictionary where the keys are motifs (nx subgraph) of length k and the keys are how many times similar
    (isomorphic) motifs occur in the graph.
    :param output_file: string thats apath of a pdf file to output the graphs to
    :return: a pdf file with name output_file with the graphs and how often they occured
    """
    motif_count = sum(motifs.values())
    motifs_sorted = sorted(motifs.items(), key=lambda kv: kv[1], reverse=True)
    with PdfPages(output_file) as pdf:
        for motif in motifs_sorted:
            fig = plt.figure()
            nx.draw_kamada_kawai(motif[0],node_size=25, arrowsize=5)
            fig.suptitle('{} Occurences ({}%)'.format(motif[1],round(100*motif[1]/motif_count,3)))
            pdf.savefig(fig)
            plt.close()


def get_embedding_clusters(embedding_input_file='./results/embeddings.csv', k_for_clustering=10, random_state=None):
    """
    Given a file with embeddings (or other features) cluster similar rows together using kmeans.

    :param embedding_input_file: file where every row is a project and every col a feature
    :param k_for_clustering: how many groups to cluster
    :param random_state: random state for clustering algo
    :return: a dictionary where the keys are the cluster labels and the values are lists of GitHub projectIds that fall in that cluster.
    """
    embeddings = pd.read_csv(embedding_input_file, index_col=0)

    # Run k-means algo
    if random_state is None:
        kmeans = KMeans(n_clusters=k_for_clustering).fit(embeddings.values)
    else:
        kmeans = KMeans(n_clusters=k_for_clustering, random_state=random_state).fit(embeddings.values)

    # Make dict where key is cluster # and value are projects in that clusters
    clusters = {}
    for n, label in enumerate(kmeans.labels_):
        if label in clusters:
            clusters[label].append(embeddings.index[n])
        else:
            clusters[label] = [embeddings.index[n]]
    return clusters


def get_most_common_motifs_from_clusters(clusters, k_for_motifs=6, number_of_samples=1000,output_folder_suffix='results'):
    """
    A way to take in a group of GitHub project clusters and output their most common motifs.

    :param clusters: a dictionary where the keys are the cluster labels and the values are lists of GitHub projectIds that fall in that cluster.
    :param k_for_motifs: the desired length of the sampled motifs.
    :param number_of_samples: how many motifs to sample from the graph.
    :param output_folder_suffix: suffix to put on end of output folder.
    :return: None.
    """
    try:
        makedirs('results/clustering_{}'.format(output_folder_suffix)) # make output folder
    except FileExistsError:
        print('About to overwrite existing output folder and files...')
        #TODO: Have user have to type 'y' or something continue, then also delete all files in folder so theres not like one cluster left over from before.

    # For each cluster, get most common subgraph
    for cluster in clusters:
        projects_cluster = getCommitsByProjectIds(clusters[cluster])
        G = git_graph(projects_cluster)
        try:
            motifs = get_motif_samples(G, k_for_motifs, number_of_samples)
        except RecursionError:
            print('too many short paths for Cluster {}, no file outputted.'.format(cluster))
            continue
        except ValueError:
            print('Cluster {} has no connections'.format(cluster))
            continue
        #TODO: output subgraphs themselves.
        visualize_motif_samples(motifs, './results/clustering_{}/cluster_{}.pdf'.format(output_folder_suffix,cluster))


if __name__ == '__main__':
    # query_p1 = commit_query(33470153)
    # data_p1 = query_ght(query_p1)
    # motifs = get_motif_samples(git_graph(data_p1), k=10, num_samples=1000)
    # visualize_motif_samples(motifs, 'imgs/commonly_occurring_motifs_proj_15059440_10.pdf')
    clusters = get_embedding_clusters(random_state=1)
    print(clusters)
    get_most_common_motifs_from_clusters(clusters)



# TODO: this is generating 1 more k than specified, idk why.
# TODO: output tsne graph with k-means labels.

