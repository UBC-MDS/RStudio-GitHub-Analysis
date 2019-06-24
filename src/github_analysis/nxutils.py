import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import logging

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", filename="log.log", level=logging.INFO)

def git_graph(commitData):
    """ Generate the commit graph in networkx from the query results.

        Parameters
        ----------
        commitData: Data pulled from the commit_query string.

        Returns
        -------
        nxGraph: Networkx graph
    """
    source_target_commits = commitData[["parent_id", "commit_id"]].dropna().astype("int64")
    source_target_commits.columns = ["source", "target"]

    return nx.from_pandas_edgelist(source_target_commits, create_using=nx.OrderedDiGraph())

def plot_commits(graph):
    """ Plot the commit graph from the networkx graph.
    
        Parameters
        ----------
        graph: The graph to plot.

        Return
        ----------
        None
    """
    nx.draw_kamada_kawai(graph, alpha=0.5, node_color='blue', node_size = 2)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(12, 8)
