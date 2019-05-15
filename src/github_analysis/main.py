import pandas_gbq
import networkx as nx
import matplotlib.pyplot as plt
import graph2vec as g2v
import time
import numpy.distutils.system_info as sysinfo
import feather
import pandas as pd
import multiprocessing

import reduce_embedding_dim as red
import data_layer as dl
import motif_finder as mf
import freq_graph as fg
import nxutils as nxutils

n_dimensions = 128

if __name__ == '__main__':
    startTime = time.time()

    # query_p1 = commit_query(22003900)
    # query_p2 = commit_query(33470153)

    # data_p1 = query_ght(query_p1)
    # data_p2 = query_ght(query_p2)

    projectData = dl.getRandomProjects(5000, 12)
    #projectData = dl.getProjectsDf()

    getDataTime = time.time()

    print("Query Time:\t\t" +           str(getDataTime - startTime) +              "\tseconds")

    #manager = multiprocessing.Manager()
    #project_graphs = manager.dict()

    project_graphs = {}
    for projectId in dl.getUniqueProjectIdsFromDf(projectData):
        projectCommits = dl.getCommitsByProjectId(projectId)
        project_graphs[projectId] = nxutils.git_graph(projectCommits)
        # TODO: Fix this cause it's slower than just doing it on one core.. .hmmmmmmm
        # You're doing this wrong self..
        #multicore_git_graph(project_graphs, projectId, projectCommits)

    generateGraphsTime = time.time()

    print("NxGraphs Built:\t\t" +        str(generateGraphsTime - getDataTime) +     "\tseconds")

    g2vModel = g2v.Graph2Vec(size=n_dimensions)
    g2vEmbeddings = g2vModel.fit_transform(list(project_graphs.values()))

    print(g2vEmbeddings)

    buildModelTime = time.time()

    print("Model Built:\t" +       str(buildModelTime - generateGraphsTime) +  "\tseconds")

    red.reduce_dim()

    reduceTime = time.time()

    print()
    print("Query Time:\t\t" +           str(getDataTime - startTime) +              "\tseconds")
    print("NxGraphs Time:\t\t" +        str(generateGraphsTime - getDataTime) +     "\tseconds")
    print("Dim Reduce Time:\t" +        str(reduceTime - buildModelTime) +          "\tseconds")
    print("Model Build Time:\t" +       str(buildModelTime - generateGraphsTime) +  "\tseconds")
    print("Total Time:\t\t" +           str(reduceTime - startTime) +                    "\tseconds")

    clusters = mf.get_embedding_clusters()
    motifs_by_cluster = mf.get_motifs_by_cluster(clusters)

    fg.generate_motif_visualisations_by_cluster()


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
