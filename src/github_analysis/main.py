"""To have this run in a reproducible manner, run: `PYTHONHASHSEED=0 python src/github_analysis/main.py` Setting
the env variable PYTHONHASHSEED to 0 will disable hash randomization."""
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
import nxutils
import pandas as pd

n_workers = 4

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", filename="log.log", level=logging.INFO)

def main():
    logging.info("===START===")
    startTime = time.time()

    project_data = dl.getRandomProjects(50000, 1)
    getDataTime = time.time()

    logging.info("Query Complete: " + str(getDataTime - startTime) + " seconds")

    project_ids = dl.getUniqueProjectIdsFromDf(project_data)
    project_groups = dl.getGroupedCommitsByProjectIds(project_ids)

    project_graphs = []
    for name, group in project_groups:
        project_graphs.append(nxutils.git_graph(group))

    generateGraphsTime = time.time()
    logging.info("NxGraphs Built: " + str(generateGraphsTime - getDataTime) + " seconds")

    g2vModel = g2v.Graph2Vec(workers=n_workers)
    g2vEmbeddings = g2vModel.fit_transform(project_graphs, project_ids)
    buildModelTime = time.time()
    logging.info("G2V Model Built: " + str(buildModelTime - generateGraphsTime) + " seconds")

    red.reduce_dim(workers=n_workers, random_state=1)
    reduceTime = time.time()
    logging.info("Dims Reduced: " + str(reduceTime - buildModelTime) + " seconds")

    clusters = c.get_embedding_clusters(random_state=1)
    projectClusterTime = time.time()
    logging.info("Projects Clustered: " + str(projectClusterTime - reduceTime) + " seconds")

    motifs_by_cluster = mf.get_motifs_by_cluster(clusters)
    motifTime = time.time()
    logging.info("Motifs Generated: " + str(motifTime - projectClusterTime) + " seconds")

    fg.generate_motif_visualisations_by_cluster()
    freqGraphTime = time.time()
    logging.info("Frequency Graphs Built: " + str(freqGraphTime- motifTime) + " seconds")

    print()
    print("Query Time:\t\t" +           str(getDataTime - startTime) +              "\tseconds")
    print("NxGraphs Time:\t\t" +        str(generateGraphsTime - getDataTime) +     "\tseconds")
    print("Model Build Time:\t" +       str(buildModelTime - generateGraphsTime) +  "\tseconds")
    print("Dim Reduce Time:\t" +        str(reduceTime - buildModelTime) +          "\tseconds")
    print("Project Cluster Time:\t" +   str(projectClusterTime - reduceTime) +      "\tseconds")
    print("Motif Generation Time:\t" +  str(motifTime - projectClusterTime) +       "\tseconds")
    print("Frequency Graph Time:\t" +   str(freqGraphTime - motifTime) +            "\tseconds")
    print("Total Time:\t\t" +           str(freqGraphTime - startTime) +            "\tseconds")

if __name__ == '__main__':
    main()

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
