# Using `get_embedding_clusters`, clustering
# motif_by_clusters: a dictionary where the keys are the cluster labels and the values are lists of motif indices within that cluster.
import graph2vec as g2v
import motif_finder as mf
import pickle
n_dimensions = 128

def motif_merging(motif_k_list, cluster_id, n_dimensions=4, epochs=3, workers=2, iter=4, input_file_path = "", k_for_clustering=10):
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

    motif_dict = {}
    for i in motif_k_list:
        input_file_name = input_file_path + 'motifs_by_cluster_%s.pickle' %i
        with open(input_file_name, 'rb') as pickle_in:
            motifs_by_cluster = pickle.load(pickle_in)
        motif_dict_k = motifs_by_cluster[cluster_id]
        motif_dict.update(motif_dict_k)

    freq_by_motif = {}
    freq_list = list(motif_dict.values())
    for i in range(0,len(freq_list)):
        freq_by_motif[i] = freq_list[i]


    m2vModel = g2v.Graph2Vec(size=n_dimensions, epochs= epochs, workers=workers, iter=iter)
    m2vModel = m2vModel.fit_transform(list(motif_dict.keys()), output_path=input_file_path+'motif_embeddings.csv')
    clusters_of_motif = mf.get_embedding_clusters(embedding_input_file=input_file_path+'motif_embeddings.csv', k_for_clustering=k_for_clustering, random_state=None,
                           output_file=input_file_path+'clusters_of_motif.pickle')

    freq_by_clusters = {}
    for cluster in clusters_of_motif:
        freq_by_clusters[cluster] = sum(freq_by_motif[i] for i in clusters_of_motif[cluster])

    return freq_by_clusters
