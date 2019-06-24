import pandas as pd
import networkx as nx
import numpy as np

class GraphExtracter():
    def __init__(self, graph):
        """ Initializes the GraphExtracter class

        Parameters
        ----------
        graph: nx.Graph()
            networkx graph object to be analyzed

        Returns
        -------
        None
        """
        self.graph = graph

    def get_nodes(self):
        """Gets the number of nodes of the graph

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of nodes in the graph
        """
        self.num_nodes = self.graph.number_of_nodes()
        return self.num_nodes

    def get_edges(self):
        """Gets the number of edges of the graph

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of edges in the graph
        """
        self.num_edges = self.graph.number_of_edges()
        return self.num_edges

    def get_density(self):
        """Gets the density of the graph

        Parameters
        ----------
        None

        Returns
        -------
        float
            Density of the graph
        """
        self.density = nx.density(self.graph)
        return self.density

    def get_avg_clustering(self):
        """Gets the average clustering of the graph

        Parameters
        ----------
        None

        Returns
        -------
        float
            Average Clustering of the graph
        """
        self.avg_clustering = nx.algorithms.cluster.average_clustering(self.graph)
        return self.avg_clustering

    def get_transitivity(self):
        """Gets the transitivity of the graph

        Parameters
        ----------
        None

        Returns
        -------
        float
            Transitivity of the graph
        """
        self.transitivity = nx.algorithms.cluster.transitivity(self.graph)
        return self.transitivity

    def get_weakly_connected(self):
        """Determines if graph is weakly connected

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if graph is weakly connected
        """
        self.weakly_connected = nx.algorithms.components.is_weakly_connected(self.graph)
        return self.weakly_connected

    def get_num_weakly_connected(self):
        """Gets the number of nodes that are weakly connected on the graph

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of nodes that are weakly connected on the graph
        """
        self.num_weakly_connected = nx.algorithms.components.number_weakly_connected_components(self.graph)
        return self.num_weakly_connected

    def get_num_attrac_components(self):
        """Gets the number of nodes that are attracting components on the graph

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of nodes that are attracting components on the graph
        """
        self.num_attrac_components = nx.algorithms.components.number_attracting_components(self.graph)
        return self.num_attrac_components

    def get_avg_degree(self):
        """Gets the average degree of the nodes on the graph
        ### Look to add the median depending on the distribution
        Parameters
        ----------
        None

        Returns
        -------
        float
            Average degree of the nodes on the graph
        """
        degree = nx.degree(self.graph)
        degree = list(degree)
        degree = [entry[1] for entry in degree]
        self.avg_degree = np.mean(degree)
        return self.avg_degree

    def get_avg_degree_centrality(self):
        """Gets the average degree centrality of the nodes on the graph

        Parameters
        ----------
        None

        Returns
        -------
        float
            Average degree centrality of the nodes on the graph
        """
        self.avg_degree_centrality = np.mean(list(nx.algorithms.centrality.degree_centrality(self.graph).values()))
        return self.avg_degree_centrality

    def get_avg_in_degree(self):
        """Gets the average in degree of the nodes on the graph
        ### Adding other stats as parameters, (i.e. median)
        Parameters
        ----------
        None

        Returns
        -------
        float
            Average in degree of the nodes on the graph
        """
        in_ = self.graph.in_degree
        in_ = list(in_)
        in_ = [entry[1] for entry in in_]
        self.avg_in_degree = np.mean(in_)
        return self.avg_in_degree

    def get_avg_in_degree_centrality(self):
        """Gets the average in degree centrality of the nodes on the graph

        Parameters
        ----------
        None

        Returns
        -------
        float
            Average in degree centrality of the nodes on the graph
        """
        self.avg_in_degree_centrality = np.mean(list(nx.algorithms.centrality.in_degree_centrality(self.graph).values()))
        return self.avg_in_degree_centrality

    def get_avg_out_degree(self):
        """Gets the average out degree of the nodes on the graph

        Parameters
        ----------
        None

        Returns
        -------
        float
            Average out degree of the nodes on the graph
        """
        out_ = self.graph.out_degree
        out_ = list(out)
        out_ = [entry[1] for entry in out]
        self.avg_out_degree = np.mean(out)
        return self.avg_out_degree

    def get_avg_out_degree_centrality(self):
        """Gets the average out degree centrality of the nodes on the graph

        Parameters
        ----------
        None

        Returns
        -------
        float
            Average out degree centrality of the nodes on the graph
        """
        self.avg_out_degree_centrality = np.mean(list(nx.algorithms.centrality.out_degree_centrality(self.graph).values()))
        return self.avg_out_degree_centrality

    def get_eigen_centrality(self):
        """Gets the mean eigen centrality of the nodes in the graph

        Parameters
        ----------
        None

        Returns
        -------
        float
             Mean eigen centrality of the nodes in the graph
        """
        self.eigen_centrality = np.mean(list(nx.algorithms.centrality.eigenvector_centrality(self.graph, max_iter=int(1e6)).values()))
        return self.eigen_centrality

    def get_katz_centrality(self):
        """Gets the mean katz centrality of the nodes in the graph

        Parameters
        ----------
        None

        Returns
        -------
        float
             Mean katz centrality of the nodes in the graph
        """
        self.katz_centrality = np.mean(list(nx.algorithms.centrality.katz_centrality(self.graph, max_iter=int(1e6)).values()))
        return self.katz_centrality

    def get_num_triangles(self):
        """Gets the mean number of triangles of the nodes in the graph

        Parameters
        ----------
        None

        Returns
        -------
        float
             Mean number of triangles of the nodes in the graph
        """
        self.num_triangles = np.mean(list(nx.algorithms.cluster.clustering(self.graph).values()))
        return self.num_triangles

    def set_all_features(self):
        """Function that extracts all of the graph features

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.get_nodes()
        self.get_edges()
        self.get_density()
        self.get_avg_clustering()
        self.get_transitivity()
        self.get_weakly_connected()
        self.get_num_weakly_connected()
        self.get_num_attrac_components()
        self.get_avg_degree()
        self.get_avg_degree_centrality()
        self.get_avg_in_degree()
        self.get_avg_in_degree_centrality()
        self.get_avg_out_degree()
        self.get_avg_out_degree_centrality()
        self.get_eigen_centrality()
        self.get_katz_centrality()
        self.get_num_triangles()
        return None

    def get_all_features(self):
        """Gets all of the features extracted from the graph object

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dictionary keyed by the features extracted from the graphs
        """
        return {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'density': self.density,
            'avg_clustering': self.avg_clustering,
            'transitivity': self.transitivity,
            'weakly_connected': self.weakly_connected,
            'num_weakly_connected': self.num_weakly_connected,
            'num_attrac_components': self.num_attrac_components,
            'avg_degree': self.avg_degree,
            'avg_degree_centrality': self.avg_degree_centrality,
            'avg_in_degree': self.avg_in_degree,
            'avg_in_degree_centrality': self.avg_in_degree_centrality,
            'avg_out_degree': self.avg_out_degree,
            'avg_out_degree_centrality': self.avg_out_degree_centrality,
            'eigen_centrality': self.eigen_centrality,
            'katz_centrality': self.katz_centrality,
            'num_triangles': self.num_triangles}

#for idx, project in enumerate(project_names_sample.project_name.values):
#    if idx%10 == 0:
#        print(f'Percentage Completed: {idx/5}%')
#    graph_df = commits_sample.loc[commits_sample['project_name'] == project]
#    graph_df = graph_df[['commit_id', 'parent_id']]
#    graph_df.columns = pd.Index(['target', 'source'])

#    if graph_df.shape[0] > 5000:
#        continue

#    graph = nx.from_pandas_edgelist(graph_df, create_using=nx.DiGraph)
#    graph.name = project
#    graph_extracter = GraphExtracter(graph)
#    graph_extracter.set_all_features()
#    graphs[project] = graph_extracter.get_all_features()

#graph_features = pd.DataFrame(graphs).T
