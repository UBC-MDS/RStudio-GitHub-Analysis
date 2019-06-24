"""To have this run in a reproducible manner, run: `PYTHONHASHSEED=0 python src/github_analysis/main.py` Setting
the env variable PYTHONHASHSEED to 0 will disable hash randomization."""

import argparse
import os
import time
import logging
import numpy.distutils.system_info as sysinfo
import collections

import graph2vec as g2v
import reduce_embedding_dim as red
import data_layer as dl
import cluster as c
import motif_finder as mf
import freq_graph as fg
import persona as p
import nxutils

import pandas as pd

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", filename="log.log", level=logging.INFO)

def main(args):
    logging.info("===START===")
    startTime = time.time()

    commits_dl = dl.data_layer(args.data_path, min_number_commits=args.min_commits)

    project_data = commits_dl.getRandomProjects(args.n_projects, args.random_state)
    getDataTime = time.time()
    logging.info("Query Complete: " + str(getDataTime - startTime) + " seconds")

    project_ids = dl.getUniqueProjectIdsFromDf(project_data)
    project_groups = commits_dl.getGroupedCommitsByProjectIds(project_ids)

    project_graphs = []
    project_ids_ordered = []
    for name, group in project_groups:
        project_graphs.append(nxutils.git_graph(group))
        project_ids_ordered.append(name)

    generateGraphsTime = time.time()
    logging.info("NxGraphs Built: " + str(generateGraphsTime - getDataTime) + " seconds")

    embeddings_path = args.results_path + "embeddings.csv"
    g2vModel = g2v.Graph2Vec(workers=args.n_workers, size=args.n_neurons, min_count=args.min_count, iter=args.n_iter, seed=args.random_state)
    g2vEmbeddings = g2vModel.fit_transform(project_graphs, project_ids_ordered, output_path=embeddings_path)
    buildModelTime = time.time()
    logging.info("G2V Model Built: " + str(buildModelTime - generateGraphsTime) + " seconds")

    red.reduce_dim(workers=args.n_workers, output_path=args.results_path + str(iter) + "/", input_path=embeddings_path, random_state=args.random_state)
    reduceTime = time.time()
    logging.info("Dims Reduced: " + str(reduceTime - buildModelTime) + " seconds")

    clusters = c.get_embedding_clusters(embedding_input_file=embeddings_path, output_file=args.results_path + "clusters.pickle", random_state=args.random_state)
    projectClusterTime = time.time()
    logging.info("Projects Clustered: " + str(projectClusterTime - reduceTime) + " seconds")

    cluster_personas = p.Personas(clusters, commits_dl, args.n_personas, 1, output_path=args.results_path + "cluster_personas.csv")
    personaGenerationTime = time.time()
    logging.info("Personas Generated: " + str(personaGenerationTime - projectClusterTime) + " seconds")

    motifs_by_cluster = mf.get_motifs_by_cluster(clusters, commits_dl, output_file=args.results_path + "motifs_by_cluster.pickle")
    motifTime = time.time()
    logging.info("Motifs Generated: " + str(motifTime - personaGenerationTime) + " seconds")

    fg.generate_motif_visualisations_by_cluster(input_file_motif_clusters=args.results_path + "motifs_by_cluster.pickle", output_file=args.results_path + "clustering_output.pdf")
    freqGraphTime = time.time()
    logging.info("Frequency Graphs Built: " + str(freqGraphTime- motifTime) + " seconds")

    print()
    print("Query Time:\t\t" +           str(getDataTime - startTime) +                      "\tseconds")
    print("NxGraphs Time:\t\t" +        str(generateGraphsTime - getDataTime) +             "\tseconds")
    print("Model Build Time:\t" +       str(buildModelTime - generateGraphsTime) +          "\tseconds")
    print("Dim Reduce Time:\t" +        str(reduceTime - buildModelTime) +                  "\tseconds")
    print("Project Cluster Time:\t" +   str(projectClusterTime - reduceTime) +              "\tseconds")
    print("Personas Generated:\t" +     str(personaGenerationTime - projectClusterTime) +   "\tseconds")
    print("Motif Generation Time:\t" +  str(motifTime - personaGenerationTime) +            "\tseconds")
    print("Frequency Graph Time:\t" +   str(freqGraphTime - motifTime) +                    "\tseconds")
    print("Total Time:\t\t" +           str(freqGraphTime - startTime) +                    "\tseconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-rp",      "--results_path",   help="The folder to output results of the analysis. e.g. embeddings and plots", default="./results/")
    parser.add_argument("-nw",      "--n_workers",      help="The number of workers to use when running the analysis.", default=8, type=int)
    parser.add_argument("-dp",      "--data_path",      help="The path to the commits.feather file. e.g. /home/user/RStudio-Data-Repository/clean_data/commits.feather", default="./results/")
    parser.add_argument("-np",      "--n_projects",     help="The number of projects to sample from the dataset.", default=1000, type=int)
    parser.add_argument("-mc",      "--min_commits",    help="The minimum number of commits for a project to be included in the sample.", default=None, type=int)
    parser.add_argument("-mcount",  "--min_count",      help="The min_count parameter for the graph2vec model.", default=5, type=int)
    parser.add_argument("-nps",     "--n_personas",     help="The number of personas to extract from each cluster.", default=5, type=int)
    parser.add_argument("-nn",      "--n_neurons",      help="The number of neurons to use for Graph2Vec (project level)", default=128, type=int)
    parser.add_argument("-ni",      "--n_iter",         help="The number of iteration to use to run the WeisfeilerLehmanMachine", default=10, type=int)
    parser.add_argument("-rs",      "--random_state",   help="The random state to initalize all random states.", default=1, type=int)

    args = parser.parse_args()

    if not os.path.exists(args.results_path):
        os.mkdir(args.results_path)

    main(args)

    # plt.clf()
    # for graph in range(len(projectGraphs)):
    #     plot_commits(projectGraphs[graph])
    #     plt.savefig("./imgs/branch_test" + str(graph))
    #     plt.clf()
    #     #plt.show()

    # def query_ght(queryString):
    #     """
    #     Function to query with the provided query string.
    #     :param queryString: String with which to perform the query.
    #     :return query_result_df: Dataframe that holds the query results.
    #     """
    #     query_result_df = pandas_gbq.read_gbq(queryString)
    #
    #     return query_result_df
    #
    # def commit_query(projectId):
    #     """
    #     Function to generate the query that will pull all commits for a given projectId.
    #     :param projectId: Project ID that you'd like to get commits for.
    #     :return queryString: Query string for the given projectId.
    #     """
    #     return """
    #             select
    #               c.id as c_id,
    #               p.id as p_id,
    #               cp.commit_id as cp_commit_id,
    #               cp.parent_id as cp_parent_id
    #             from `ghtorrent-bq.ght.commits` c
    #             left join `ghtorrent-bq.ght.projects` p on (p.id = c.project_id)
    #             left join `ghtorrent-bq.ght.commit_parents` cp on (cp.commit_id = c.id)
    #             where (p.id = """ + str(projectId) + """)
    #         """
    #