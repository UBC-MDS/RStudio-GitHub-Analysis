# Using cosine similarity
import graph2vec as g2v
import motif_finder as mf
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
n_dimensions = 128

def get_most_similar(motif_dict, motif):
    cos_sim_list = []
    for m in motif_dict:
        cos_sim = cosine_similarity(list(motif_dict[m].values())[0].reshape(1,-1), motif.reshape(1,-1))
        cos_sim = cos_sim.tolist()[0][0]
        cos_sim_list.append(cos_sim)
    sim_motif = list(motif_dict.keys())[cos_sim_list.index(min(cos_sim_list))]
    sim_motif_embedding = list(motif_dict[sim_motif].values())[0]
    return [sim_motif,sim_motif_embedding]

def motif_merging1(input_file_motif_clusters='motifs_by_cluster.pickle', cluster_id=0, min_sim=0.01):
    with open(input_file_motif_clusters, 'rb') as pickle_in:
        motifs_by_cluster = pickle.load(pickle_in)

    motif_dict = motifs_by_cluster[cluster_id]

    m2vModel = g2v.Graph2Vec(size=n_dimensions)
    m2vModel = m2vModel.fit_transform(list(motif_dict.keys()))
    # m2vModel.save_embeddings(len(motif_dict), output_path='./results/motif_embeddings.csv')
    motif_embedding = pd.read_csv('embeddings.csv', index_col=0)

    i = 0
    for motif in motif_dict:
        motif_dict[motif]={motif_dict[motif]:motif_embedding.values[i]}
        i+=1

    new_dict = {}
    merge_dict = {}
    i = 0
    for motif in motif_dict:
        if len(new_dict)==0:
            new_dict[motif] = motif_dict[motif]
            merge_dict.update({i:[]})
        else:
            m_embedding = list(motif_dict[motif].values())[0]
            sim_motif = get_most_similar(new_dict, m_embedding)
            cos_sim = cosine_similarity(sim_motif[1].reshape(1,-1), m_embedding.reshape(1,-1))
            if cos_sim < min_sim:
                new_dict[sim_motif[0]] = {(list(new_dict[sim_motif[0]].keys())[0]+list(motif_dict[motif].keys())[0]):sim_motif[1]}
                sim_motif_index = list(motif_dict.keys()).index(sim_motif[0])
                merge_dict[sim_motif_index].append(i)
            else:
                new_dict[motif] = motif_dict[motif]
                merge_dict.update({i:[]})
        i+=1

    return new_dict, merge_dict

# Using `get_embedding_clusters`, clustering
# motif_by_clusters: a dictionary where the keys are the cluster labels and the values are lists of motif indices within that cluster.
import graph2vec as g2v
import motif_finder as mf
import pickle
n_dimensions = 128

def motif_merging2(input_file_motif_clusters='motifs_by_cluster.pickle', cluster_id=0):
    with open(input_file_motif_clusters, 'rb') as pickle_in:
        motifs_by_cluster = pickle.load(pickle_in)

    motif_dict = motifs_by_cluster[cluster_id]

    m2vModel = g2v.Graph2Vec(size=n_dimensions)
    m2vModel = m2vModel.fit_transform(list(motif_dict.keys()))
    #m2vModel.save_embeddings(len(motif_dict), output_path='./results/motif_embeddings.csv')
    clusters_of_motif = mf.get_embedding_clusters(embedding_input_file='./results/motif_embeddings.csv', k_for_clustering=10, random_state=None,
                           output_file='./results/clusters_of_motif.pickle')

    freq_by_clusters = {}
    for cluster in clusters_of_motif:
        freq_by_clusters[cluster] = sum([list(motif_dict.values())[i] for i in clusters_of_motif[cluster]])

    return clusters_of_motif, freq_by_clusters
