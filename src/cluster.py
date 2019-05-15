import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN

class Cluster():
    def __init__(self, raw_data):
        """ Initializes the Cluster class

        Parameters
        ----------
        raw_data: pd.DataFrame or np.ndarray
            Data in a 2 dimensional ndarray or a pandas Data Frame

        Returns
        -------
        None
        """
        self.raw_data = raw_data
        self.data = None
        self.algorithm = None
        self.transformed_data  = None

    def scale_data(self, min_max = False):
        """ Scales the data in all columns to a same scale

        Parameters
        ----------
        min_max: bool
            If True uses the MinMaxScaler, if False uses the StandardScaler

        Returns
        -------
        None
        """
        data = self.raw_data

        if min_max:
            scaled_data = MinMaxScaler().fit_transform(data)
        else:
            scaled_data = StandardScaler().fit_transform(data)

        self.data = scaled_data

    def set_algorithm(self, name, **kwargs):
        """ Sets the clustering algorithm to use

        Parameters
        ----------
        name: str
            Name of the algorithm to use
        **kwargs
            Named arguments specific to the algorithm to use

        Returns
        -------
        None
        """
        if name == 'k_means':
            self.algorithm = KMeans(**kwargs)
        elif name == 'dbscan':
            self.algorithm = DBSCAN(**kwargs)

    def fit_algorithm(self):
        """ Fits the algorithm to the scaled data

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.scale_data()
        self.algorithm.fit(self.data)

    def get_labels(self):
        """ Gets the cluster labels

        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            Array of cluster labels
        """
        self.labels = self.algorithm.labels_
        return self.labels

    def get_inertia(self):
        """ Gets the inertia of the clusters

        Parameters
        ----------
        None

        Returns
        -------
        float
            Returns the intertia if the algorithm has an inertia attribute
        """
        try:
            self.inertia = self.algorithm.inertia_
            return self.inertia
        except:
            print('Not Inertia in this algorithm')

cluster = Cluster([[1,2,3], [2, 4, 6]])
cluster.set_algorithm('k_means', n_clusters = 2)
print('Up to here')
print(cluster.fit_algorithm())
print(cluster.get_labels())
