from os import remove
import pickle
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import networkx as nx


def generate_motif_visualisations_by_cluster(input_file_motif_clusters='./results/motifs_by_cluster.pickle', output_file='./results/clustering_output.pdf'):
    """ Visualize the motif clustering result and output the visualize result to pdf file.

        Parameters
        ----------
        input_file_motif_clusters: string with a filepath to a pickled dictionary where the keys are cluster names and the values
                                   are dictionaries where the keys are motifs (nx subgraph) of length k and the values are how
                                   many times similar(isomorphic) motifs occur in the graph.
        output_file: string thats a path of a pdf file to output the graphs to.

        Returns
        -------
        Visulization result of motif clustering, by cluster, saved down to a pdf file.
    """
    with open(input_file_motif_clusters, 'rb') as pickle_in:
        motif_clusters = pickle.load(pickle_in)

    # Sort keys in cluster dictionary so they are outputted in order
    sorted_cluster_keys = list(motif_clusters.keys())
    sorted_cluster_keys.sort()

    with PdfPages(output_file) as pdf:
        for cluster in sorted_cluster_keys:
            cluster_visual = visualize_motif_samples_bar_graph(motif_clusters[cluster], 'Cluster ' + str(cluster))
            pdf.savefig(cluster_visual, pad_inches=5)


def visualize_motif_samples_bar_graph(motifs, plot_title='Motif Frequency in Dataset', motifs_to_show=8):
    """ Given a collection of motifs and their frequency in a graph, output a file with a bar chart showing the motifs and
        their associated frequencies.

        Parameters
        ----------
        motifs: dictionary where the keys are motifs (nx subgraph) of length k and the values are how many times similar
                (isomorphic) motifs occur in the graph.
        plot_title: string thats the tile of your plot.

        Returns
        -------
        A bar chart figure of the most common motifs and how often they occurred.
    """
    motifs_sorted = sorted(motifs.items(), key=lambda kv: kv[1], reverse=True)

    # output files with individual motif images to be used in bar graph
    occurrences = []
    for n, motif in enumerate(motifs_sorted):
        # print(motif[1])
        # nx.draw_spectral(motif[0], node_size=500, arrowsize=40, width=6)
        # plt.show()

        fig = plt.figure(figsize=(3, 3))

        # check for single chain motif (this assumes there always is one)
        if sum([motif[0].out_degree(node) in [0, 1] for node in motif[0]]) == len(motif[0].nodes):
            single_chain_occurences = motif[1]
            continue

        if n > motifs_to_show:
            break

        nx.draw_spectral(motif[0], node_size=500, arrowsize=40, width=6)
        plt.savefig('graph_{}.png'.format(n))
        plt.close()
        occurrences.append(motif[1])

    # Make bar graph of motif frequency
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    y_pos = np.arange(len(occurrences))
    number_of_samples = sum(motifs.values())
    ax.bar(y_pos, [100 * occurrence / number_of_samples for occurrence in occurrences], align='center')

    # Annotate the bar graph with the motif images
    motif_graph_file_list = glob.glob('graph_*.png')
    # print(motif_graph_file_list)
    motif_graph_file_list.sort()
    for n, file_ in enumerate(motif_graph_file_list):
        arr_img = plt.imread(file_, format='png', )
        remove(file_)  # delete file

        imagebox = OffsetImage(arr_img, zoom=0.2)
        imagebox.image.axes = ax

        ab = AnnotationBbox(imagebox, (n, 0),
                            xybox=(0, -7),
                            xycoords=("data", "axes fraction"),
                            boxcoords="offset points",
                            box_alignment=(.5, .4),
                            bboxprops={"edgecolor": "none", "facecolor": "none"})

        ax.add_artist(ab)

    fig.suptitle(plot_title)
    ax.set_ylabel('Rate Motif Occurred (%)')
    #ax.set_xlabel('Motifs')
    ax.set_title(
        '{}% of Sampled Motifs are a Single Chain'.format(round(100 * single_chain_occurences / number_of_samples, 3)))
    return fig

if __name__ == '__main__':
    main()
