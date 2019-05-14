"""
This script tests the classes and functions from motif_finder.py.

Parameters
----------
None
Returns
-------
Assertion errors if tests fail
"""

import sys
import random
import pickle

import networkx as nx

from big_cloud_scratch import git_graph
from data_layer import getCommitsByProjectIds
from cluster import get_embedding_clusters
from motif_finder import *

clusters = get_embedding_clusters(random_state=0)
projects_cluster = getCommitsByProjectIds(clusters[0])
G = git_graph(projects_cluster)
mf = MotifFinder(G)


# Unit tests
def test_sample_initial_node_output_type():
    """Check that MotifFinder.sample_initial_node outputs an integer."""
    assert type(mf.sample_initial_node()) == int


def test_sample_initial_node_output():
    """Check that MotifFinder.sample_initial_node outputs a node in the given graph."""
    assert mf.sample_initial_node() in G


def test_get_random_child_output_type():
    """Check that MotifFinder.get_random_child outputs an integer."""
    initial_node = mf.sample_initial_node()
    print(initial_node)
    assert type(mf.get_random_child(initial_node)) in (int,None)


def test_get_random_child_output():
    """Check that MotifFinder.get_random_child outputs a child of the node its been given."""
    initial_node = mf.sample_initial_node()
    child = mf.get_random_child(initial_node)
    assert (child in G.successors(initial_node) or child is None)


def test_get_sample_motif_bad_input():
    """Check that MotifFinder.get_sample_motif raises an error when not given an integer for the k param."""
    try:
        mf.get_sample_motif('5')
    except TypeError:
        return True
    raise TypeError


def test_get_sample_motif_output_type():
    """Check that MotifFinder.get_sample_motif outputs a networkx directed graph."""
    assert type(mf.get_sample_motif(5)) == nx.classes.digraph.DiGraph


def test_get_sample_motif_output():
    """Check that MotifFinder.get_sample_motif outputs a networkx directed graph that is a subgraph of G."""
    subgraph = mf.get_sample_motif(5)
    for node in subgraph:
        if node in G:
            continue
        else:
            raise ValueError('Subgraph doesnt contain same nodes as graph')


def test_get_motif_samples_bad_input():
    """Check that MotifFinder.get_motif_samples raises an error when not given an integer for the k and num_samples
    param."""
    try:
        mf.get_motif_samples('5', '5')
    except TypeError:
        return True
    raise TypeError


def test_get_motif_samples_output_type():
    """Check that MotifFinder.get_sample_motif outputs a dictionary."""
    assert type(mf.get_motif_samples(5,5)) == dict


def test_get_motif_samples_output_keys():
    """Check that the output of get MotifFinder.get_sample_motif is a dictionary where the keys are a directed networkx
    graph."""
    pass