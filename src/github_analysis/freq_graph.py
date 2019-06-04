from os import remove
import pickle
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import networkx as nx


def generate_motif_visualisations_by_cluster(input_file_motif_clusters='./results/motifs_by_cluster.pickle', output_file='./results/clustering_output.pdf'):
    """
    :param input_file_motif_clusters: string with a filepath to a pickled dictionary where the keys are cluster names and the values
    are dictionaries where the keys are motifs (nx subgraph) of length k and the values are how many times similar
    (isomorphic) motifs occur in the graph.
    :param output_file: string thats a path of a pdf file to output the graphs to
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
    """
    Given a collection of motifs and their frequency in a graph, output a file with a bar chart showing the motifs and
    their associated frequencies.

    :param motifs: dictionary where the keys are motifs (nx subgraph) of length k and the values are how many times similar
    (isomorphic) motifs occur in the graph.
    :param plot_title: string thats the tile of your plot.
    :return: fig that is a bar chart of the most common motifs and how often they occurred

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

#    plt.savefig(output_file, pad_inches=2)
#    plt.close()


# def visualize_motif_samples(motifs, output_file):
#     """
#     Given a sample of motifs, output a file with their graphs and how often they occurred.
#
#     :param motifs: a dictionary where the keys are motifs (nx subgraph) of length k and the keys are how many times similar
#     (isomorphic) motifs occur in the graph.
#     :param output_file: string thats apath of a pdf file to output the graphs to
#     :return: a pdf file with name output_file with the graphs and how often they occured
#     """
#     motif_count = sum(motifs.values())
#     motifs_sorted = sorted(motifs.items(), key=lambda kv: kv[1], reverse=True)
#     with PdfPages(output_file) as pdf:
#         for motif in motifs_sorted:
#             fig = plt.figure()
#             nx.draw_kamada_kawai(motif[0], node_size=25, arrowsize=5)
#             fig.suptitle('{} Occurrences ({}%)'.format(motif[1], round(100 * motif[1] / motif_count, 3)))
#             pdf.savefig(fig)
#             plt.close()
#

if __name__ == '__main__':
    main()

 # try:
        #     makedirs('results/clustering_{}'.format(output_folder_suffix)) # make output folder
        # except FileExistsError:
        #     print('About to overwrite existing output folder and files...')
        #     #TODO: Have user have to type 'y' or something continue, then also delete all files in folder so theres not like one cluster left over from before.




#         cluster_visual = visualize_motif_samples_bar_graph(motifs, 'Cluster ' + str(cluster), number_of_samples)
#         pdf.savefig(cluster_visual,pad_inches=2)
#         #visualize_motif_samples(motifs, './results/clustering_{}/cluster_{}.pdf'.format(output_folder_suffix,cluster))
#


    # # Sort keys in cluster dictionary so they are outputted
    # sorted_cluster_keys = list(clusters.keys())
    # sorted_cluster_keys.sort()
