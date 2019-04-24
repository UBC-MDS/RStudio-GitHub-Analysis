import hashlib
import logging
import pandas as pd
import networkx as nx
import numpy.distutils.system_info as sysinfo
from gensim.models.doc2vec import TaggedDocument

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """
    def __init__(self, graph, features, iterations):
        """
        Initialization method which executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k,v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = "_".join([str(self.features[node])]+list(set(sorted([str(deg) for deg in degs]))))
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for iteration in range(self.iterations):
            self.features = self.do_a_recursion()

def feature_extractor(graph, rounds, name):
    """
    Function to extract WL features from a graph.
    :param graph: The nx graph.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
    features = nx.degree(graph)
    features = {int(k):v for k,v, in features}

    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    doc = TaggedDocument(words = machine.extracted_features , tags = ["g_" + name])
    return doc

def save_embedding(output_path, model, n_graphs, dimensions):
    """
    Function to save the embedding.
    :param output_path: Path to the embedding csv.
    :param model: The embedding model object.
    :param n_graphs: The number of graphs used to train the model.
    :param dimensions: The embedding dimension parameter.
    """
    out = []
    for identifier in range(n_graphs):
        out.append([identifier] + list(model.docvecs["g_"+str(identifier)]))

    out = pd.DataFrame(out,columns = ["type"] +["x_" +str(dimension) for dimension in range(dimensions)])
    out = out.sort_values(["type"])
    out.to_csv(output_path, index = None)
