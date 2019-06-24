from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding
import dim_reduce
import numpy as np

embeddings = np.array([[1, 2, 3], [4, 5, 6], [6, 5 , 4], [3, 2, 1]])

reducer = dim_reduce.ReduceDim(embeddings, n_dimensions = 2)

def test_init_():
    """Tests that the ReduceDim class is initializing correctly"""
    assert reducer.dimensions == 2, 'The ReduceDim() class is not initializing correctly.'
    assert reducer.data is None, 'The ReduceDim() class is not initializing correctly.'
    assert reducer.raw_data is None, 'The ReduceDim() class is not initializing correctly.'
    assert reducer.algorithm is None, 'The ReduceDim() class is not initializing correctly.'

def test_scale_data():
    """Tests that both methods of scaling work properly"""
    # Tests Min-Max Scaling
    reducer.scale_data(min_max = True)
    assert np.min(reducer.data) == 0, 'It should be 0 as it is Min-Max'
    assert np.isclose(np.max(reducer.data), 1), 'It should be 1 as it is Min-Max'
    assert reducer.data is not None, 'The scaling is going correctly but not the assignments to the class attributes'

    # Resets the state of the  attributes
    reducer.data = None

    #Tests StandardScaler
    reducer.scale_data(min_max = False)
    assert np.array_equal(np.mean(embeddings, axis = 1), np.array([2., 5., 5., 2.]))  #This is not testing StandardScaler
    assert np.array_equal(np.mean(embeddings, axis = 0), np.array([3.5, 3.5, 3.5]))  #This is not testing StandardScaler
    assert reducer.data is not None, 'The scaling is going correctly but not the assignments to the class attributes'

def test_set_algorithm():
    """Tests that the method assigns the correct algorithms."""
    # Tests PCA
    reducer.set_algorithm('pca')
    assert isinstance(reducer.algorithm, PCA), 'PCA is not implemented correctly'

    # Tests t-SNE
    reducer.set_algorithm('t_sne')
    assert isinstance(reducer.algorithm, TSNE), 'TSNE is not implemented correctly'

    # Tests Isomap
    reducer.set_algorithm('isomap')
    assert isinstance(reducer.algorithm, Isomap), 'Isomap is not implemented correctly'

    # Tests LocallyLinearEmbedding
    reducer.set_algorithm('locally_linear')
    assert isinstance(reducer.algorithm, LocallyLinearEmbedding), 'LocallyLinearEmbedding is not implemented correctly'

    # Tests MDS
    reducer.set_algorithm('mds')
    assert isinstance(reducer.algorithm, MDS), 'MDS is not implemented correctly'

    # Tests Spectral
    reducer.set_algorithm('spectral')
    assert isinstance(reducer.algorithm, SpectralEmbedding), 'SpectralEmbedding is not implemented correctly'

def test_fit_transform():
    """Tests that the fit_transform methods work correctly"""
    reducer.set_algorithm('t_sne')

    # Transform test
    reduced_data = reducer.fit_transform()
    assert reduced_data is not None
    assert reduced_data.shape[1] == 2

    # Reset state
    reducer.transformed_data = None

    # Don't transform test
    assert reducer.fit_transform(transform = False) is None
