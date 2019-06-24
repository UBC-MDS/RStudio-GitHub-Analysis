import numpy as np
import cluster as clustering
from sklearn.cluster import KMeans, DBSCAN

embeddings = np.array([[1, 2, 3], [4, 5, 6], [6, 5 , 4], [3, 2, 1]])

cluster = clustering.Cluster(embeddings)

### Look to add explanations about what should happen after the asserts

def test_init_():
    """Tests that the Cluster class is initializing correctly"""
    assert cluster.raw_data is None, 'The Cluster() class is not initializing correctly.'
    assert cluster.data is None, 'The Cluster() class is not initializing correctly.'
    assert cluster.algorithm is None, 'The Cluster() class is not initializing correctly.'

def test_scale_data():
    """Tests that both methods of scaling work properly"""
    # Tests Min-Max Scaling
    cluster.scale_data(min_max = True)
    assert np.min(cluster.data) == 0, 'It should be 0 as it is Min-Max'
    assert np.isclose(np.max(cluster.data), 1), 'It should be 1 as it is Min-Max'
    assert cluster.data is not None, 'The scaling is going correctly but not the assignments to the class attributes'

    # Resets the state of the  attributes
    cluster.data = None

    #Tests StandardScaler
    cluster.scale_data(min_max = False)
    assert np.array_equal(np.mean(embeddings, axis = 1), np.array([2., 5., 5., 2.])) #This is not testing StandardScaler
    assert np.array_equal(np.mean(embeddings, axis = 0), np.array([3.5, 3.5, 3.5])) #This is not testing StandardScaler
    assert cluster.data is not None

def test_set_algorithm():
    """Tests that the method assigns the correct algorithms."""
    # Tests K-Means
    cluster.set_algorithm('k_means')
    # Look at class names for comparison
    assert isinstance(cluster.algorithm, MiniBatchKMeans), 'MiniBatchKMeans is not implemented correctly'

    # Tests DBSCAN
    cluster.set_algorithm('dbscan')
    assert isinstance(cluster.algorithm, DBSCAN), 'DBSCAN is not implemented correctly'

def test_fit_algorithm():
    """Tests that the selected algorithm is fitted correctly"""
    cluster.set_algorithm('k_means')
    cluster.fit_algorithm()
    assert cluster.fitted == True, 'Didn\'t fit the algorithm correctly'

def test_get_labels():
    """Tests that the label assignments are working as expected"""
    labels = cluster.labels
    assert labels is not None, 'Labels assignments are not correctly done'
    assert len(labels) == embeddings.shape[0], 'Labels are not the same length as the input'

def test_get_inertia():
    """Tests that the inertia we get is an actual number and not empty"""
    inertia = cluster.get_inertia
    assert len(inertia) == 1, 'Not getting the correct inertia values'
    assert inertia is not None
    assert inertia is float, 'Not getting the correct inertia values'
