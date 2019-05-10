import hashlib
import logging
import pandas as pd
import networkx as nx
import numpy.distutils.system_info as sysinfo

from joblib import Parallel, delayed
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

class Graph2Vec:
    def __init__(self, size=128, epochs=10, workers=4):
        self.size = size
        self.epochs = epochs
        self.workers = workers
        self.fitted = False

    def extract_features(self, projectGraphs):
        document_collections = Parallel(n_jobs = self.workers)(delayed(self.feature_extractor)(projectGraphs[g], self.epochs, str(g)) for g in tqdm(range(len(projectGraphs))))

        return document_collections

    def fit(self, projectGraphs):
        self.model = Doc2Vec(self.extract_features(projectGraphs),
                        size = self.size,
                        window = 0,
                        min_count = 5,
                        dm = 0,
                        sample = 0.0001,
                        workers = self.workers,
                        epochs = self.epochs,
                        alpha = 0.025)

        self.fitted = True

    def feature_extractor(self, graph, rounds, name):
        """
        Function to extract WL features from a graph.
        :param graph: The nx graph.
        :param rounds: Number of WL iterations.
        :param name: ProjectId to output
        :return doc: Document collection object.
        """
        features = nx.degree(graph)
        features = {int(k):v for k,v, in features}

        machine = WeisfeilerLehmanMachine(graph, features, rounds)
        doc = TaggedDocument(words = machine.extracted_features , tags = ["g_" + name])
        return doc

    def get_embeddings(self, n_graphs, dimensions):
        """
        Function to get embeddings from the model.
        :param n_graphs: The number of graphs used to train the model.
        :param dimensions: The embedding dimension parameter.
        """
        if not self.fitted:
            print("Model has not been fit, run Graph2Vec.fit() before getting embeddings")
            return

        out = []
        for identifier in range(n_graphs):
            out.append([identifier] + list(self.model.docvecs["g_"+str(identifier)]))

        out = pd.DataFrame(out,columns = ["type"] +["x_" +str(dimension) for dimension in range(dimensions)])
        out = out.sort_values(["type"])

        return out

    def save_embeddings(self, output_path, n_graphs, dimensions):
        """
        Function to save the embedding.
        :param output_path: Path to the embedding csv.
        :param n_graphs: The number of graphs used to train the model.
        :param dimensions: The embedding dimension parameter.
        """
        if not self.fitted:
            print("Model has not been fit, run Graph2Vec.fit() before saving embeddings")
            return

        embeddings = self.get_embeddings(n_graphs, dimensions)
        embeddings.to_csv(output_path, index = None)

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
            # TODO: Change neighbours to children
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
