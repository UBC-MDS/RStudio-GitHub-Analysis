from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

class ReduceDim():
    def __init__(self, n_dimensions):
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
        self.dimensions = n_dimensions
        self.raw_data = None
        self.data = None
        self.algorithm = None
        self.transformed_data  = None

    def open_embeddings(self, input_file):
        self.raw_data = pd.read_csv(input_file, index_col = 0)

    def scale_data(self, min_max = True):
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
            Name of the algorithm to use ### (*add algorithms available in the docstring*)
        **kwargs
            Named arguments specific to the algorithm to use

        Returns
        -------
        None
        """
        name = name.lower()

        if name == 'pca':
            self.algorithm = PCA(n_components = self.dimensions, **kwargs)
        elif name == 't_sne':
            self.algorithm = TSNE(n_components = self.dimensions, **kwargs)
        elif name == 'isomap':
            self.algorithm = Isomap(n_components = self.dimensions, **kwargs)
        elif name == 'locally_linear':
            self.algorithm = LocallyLinearEmbedding(n_components = self.dimensions, **kwargs)
        elif name == 'mds':
            self.algorithm = MDS(n_components = self.dimensions, **kwargs)
        elif name == 'spectral':
            self.algorithm = SpectralEmbedding(n_components = self.dimensions, **kwargs)

    def fit_transform(self):
        """ Fits the algorithm to the scaled data

        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            Dimensionality reduced data
        """
        self.scale_data()

        self.transformed_data = self.algorithm.fit_transform(self.data)
        self.transformed_data = pd.DataFrame(self.transformed_data, columns = ['x', 'y'])
        self.transformed_data.index = self.raw_data.index
        return self.transformed_data

    def plot_tsne(self, file_name):
        fig, ax = plt.subplots()
        ax.scatter(self.transformed_data.x, self.transformed_data.y)
        ax.set_title('Embedding Clusters (t-SNE Transformed)')
        plt.savefig(file_name)

    def save_reduced_data(self, output_file):
        self.transformed_data.to_csv(output_file)
