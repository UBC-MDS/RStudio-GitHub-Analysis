import logging
import pickle

import pandas as pd
from sklearn.cluster import KMeans

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",
                    filename="log.log", level=logging.INFO)


def get_embedding_clusters(embedding_input_file='./results/embeddings.csv', k_for_clustering=10, random_state=None,
                           output_file='./results/clusters.pickle'):
    """
    Given a file with embeddings (or other features) cluster similar rows together using kmeans.

    :param embedding_input_file: file where every row is a project and every col a feature
    :param k_for_clustering: how many groups to cluster
    :param random_state: random state for clustering algo
    :param output_file: string with the filename to output the results to as a pickle. If this param is set to None no file will be outputted.
    :return: a dictionary where the keys are the cluster labels and the values are lists of GitHub projectIds that fall in that cluster.
    """
    embeddings = pd.read_csv(embedding_input_file, index_col=0)

    # Run k-means algo TODO: spend more time on this algo: tune hyperparams, consider algo that better handles high dim, etc.
    kmeans = KMeans(n_clusters=k_for_clustering,
                    random_state=random_state).fit(embeddings.values)

    # Make dict where key is cluster # and value are projects in that clusters
    clusters = {}
    for n, label in enumerate(kmeans.labels_):
        if label in clusters:
            clusters[label].append(embeddings.index[n])
        else:
            clusters[label] = [embeddings.index[n]]

    if output_file is not None:
        with open(output_file, 'wb') as output:
            pickle.dump(clusters, output)
        logging.info('Cluster file outputted!')

    return clusters


if __name__ == '__main__':
    get_embedding_clusters()
