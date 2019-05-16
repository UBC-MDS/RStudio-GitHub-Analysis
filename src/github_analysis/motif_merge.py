# Using cosine similarity

from sklearn.metrics.pairwise import cosine_similarity
motif_dict = motifs_by_cluster[0] # a {motif in nx graph:freq} dictionary

def get_most_similar(motif_dict, motif):
    cos_sim_list = []
    for m in motif_dict:
        cos_sim_list.append(cosine_similarity(list(motif_dict[m].values())[0], motif))
    sim_motif = list(motif_dict.keys())[cos_sim_list.index(min(cos_sim_list))]
    return sim_motif

def motif_merging(motif_dict, min_sim=0.1):
    m2vModel = g2v.Graph2Vec(size=n_dimensions)
    m2vModel = m2vModel.fit_transform(list(motif_dict.keys()))
    m2vModel.save_embeddings(len(motif_dict), output_path='./results/motif_embeddings.csv')

    for motif in motif_dict:
        motif_dict[motif]={motif_dict[motif]:embedding}

    new_dict = {}
    for motif in motif_dict:
        if len(new_dict)==0:
            new_dict[motif] = motif_dict[motif]
        else:
            sim_motif = get_most_similar(new_dict, list(motif_dict[motif].values())[0])
            cos_sim = cosine_similarity(motif, sim_motif)
            if cos_sim < min_sim:
                new_dict[sim_motif] = new_dict[sim_motif]+list(motif_dict[motif].keys())[0]
            else:
                new_dict[motif] = motif_dict[motif]
    return new_dict

# Using `get_embedding_clusters`, clustering
# motif_clusters: a dictionary where the keys are the cluster labels and the values are lists of motif indices within that cluster.

def motif_merging(motif_clusters):
    motif_dict = motifs_by_cluster[cluster_id]
    m2vModel = g2v.Graph2Vec(size=n_dimensions)
    m2vModel = m2vModel.fit_transform(list(motif_dict.keys()))
    m2vModel.save_embeddings(len(motif_dict), output_path='./results/motif_embeddings.csv')
    motif_clusters = mf.get_embedding_clusters(embedding_input_file='./results/motif_embeddings.csv', k_for_clustering=10, random_state=None,
                           output_file='./results/motif_clusters.pickle')

    freq_by_clusters = {}
    for cluster in motif_clusters:
        freq_by_clusters[cluster] = sum([list(motif_dict.values())[i] for i in motif_clusters[cluster]])
    return freq_by_clusters  
