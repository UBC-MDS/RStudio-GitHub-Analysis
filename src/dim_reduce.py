from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np

class ReduceDim():
    def __init__(self, raw_data, n_dimensions):
        """ Initializes the ReduceDim class

        Parameters
        ----------
        raw_data: pd.DataFrame or np.ndarray
            Data in a 2 dimensional ndarray or a pandas Data Frame
        n_dimensions: int
            Number of dimensions we want to reduce to

        Returns
        -------
        None
        """
        self.raw_data = raw_data
        self.data = None
        self.algorithm = None
        self.transformed_data  = None
        self.dimensions = n_dimensions

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
        """ Sets the dimensionality reduction algorithm to use

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
        if name == 'pca':
            self.algorithm = PCA(n_components = self.dimensions, **kwargs)
        elif name == 't_sne':
            self.algorithm = TSNE(n_components = seld.dimensions, **kwargs)

    def fit_transform(self, transform = True):
        """ Fits the algorithm to the scaled data

        Parameters
        ----------
        transform: bool
            If True then returns transformed data

        Returns
        -------
        ndarray
            Dimensionality reduced data
        """
        self.scale_data()
        self.algorithm.fit(self.data)

        if transform:
            self.transformed_data = self.algorithm.transform(self.data)
            return self.transformed_data
        else:
            return None
