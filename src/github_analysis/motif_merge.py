# Using `get_embedding_clusters`, clustering
# motif_by_clusters: a dictionary where the keys are the cluster labels and the values are lists of motif indices within that cluster.
import graph2vec as g2v
import motif_finder as mf
import pickle
n_dimensions = 128

def motif_merging(input_file_motif_clusters='motifs_by_cluster.pickle', k_for_clustering=10):
    """ Group similar motifs together and add up their frequencies.

        Parameters
        ----------
        input_file_motif_clusters: motifs from clusters of projects, stored in pickle file, as dictionary.
        cluster_id: motifs from the specific cluster of projects we want to focus on.
        k_for_clustering: number of clusters.

        Returns
        -------
        Groups of similar motifs(noted by their index from the embedding file) and frequencies of each group
    """

    with open(input_file_motif_clusters, 'rb') as pickle_in:
        motifs_by_cluster = pickle.load(pickle_in)

    motif_dict = motifs_by_cluster[cluster_id]

    freq_by_motif = {}
    freq_list = list(motif_dict.values())
    for i in range(0,len(freq_list)):
        freq_by_motif[i] = freq_list[i]


    m2vModel = g2v.Graph2Vec(size=n_dimensions)
    m2vModel = m2vModel.fit_transform(list(motif_dict.keys()), output_path='./results/motif_embeddings.csv')
    #m2vModel.save_embeddings(len(motif_dict), output_path='./results/motif_embeddings.csv')
    clusters_of_motif = mf.get_embedding_clusters(embedding_input_file='./results/motif_embeddings.csv', k_for_clustering=k_for_clustering, random_state=None,
                           output_file='./results/clusters_of_motif.pickle')

    freq_by_clusters = {}
    for cluster in clusters_of_motif:
        freq_by_clusters[cluster] = sum(freq_by_motif[i] for i in clusters_of_motif[cluster])

    return clusters_of_motif, freq_by_clusters
