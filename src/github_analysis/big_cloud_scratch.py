import pandas_gbq
import networkx as nx
import matplotlib.pyplot as plt
import graph2vec as g2v
import time
import numpy.distutils.system_info as sysinfo
import github_analysis.reduce_embedding_dim as red
import feather
import github_analysis.data_layer as dl
import pandas as pd
import multiprocessing

def git_graph(commitData):
    """
    Function to generate the commit graph in networkx from the query results.
    :param commitData: Data pulled from the commit_query string.
    :return nxGraph: Networkx graph
    """
    source_target_commits = commitData[["parent_id", "commit_id"]].dropna().astype("int64")
    #print(source_target_commits.head())
    source_target_commits.columns = ["source", "target"]

    return nx.from_pandas_edgelist(source_target_commits, create_using=nx.OrderedDiGraph())

def multicore_git_graph(d, projectId, projectGraph):
    d[projectId] = git_graph(projectGraph)

n_dimensions = 128

if __name__ == '__main__':
    startTime = time.time()

    # query_p1 = commit_query(22003900)
    # query_p2 = commit_query(33470153)

    # data_p1 = query_ght(query_p1)
    # data_p2 = query_ght(query_p2)

    projectData = dl.getRandomProjects(50000, 12)
    #projectData = dl.getProjectsDf()

    getDataTime = time.time()

    print("Query Time:\t\t" +           str(getDataTime - startTime) +              "\tseconds")

    #manager = multiprocessing.Manager()
    #project_graphs = manager.dict()

    project_graphs = {}
    for projectId in dl.getUniqueProjectIdsFromDf(projectData):
        projectCommits = dl.getCommitsByProjectId(projectId)
        project_graphs[projectId] = git_graph(projectCommits)
        # TODO: Fix this cause it's slower than just doing it on one core.. .hmmmmmmm
        # You're doing this wrong self..
        #multicore_git_graph(project_graphs, projectId, projectCommits)

    generateGraphsTime = time.time()

    print("NxGraphs Time:\t\t" +        str(generateGraphsTime - getDataTime) +     "\tseconds")

    g2vModel = g2v.Graph2Vec(size=n_dimensions)
    g2vEmbeddings = g2vModel.fit_transform(list(project_graphs.values()))

    print(g2vEmbeddings)

    buildModelTime = time.time()

    print("Model Build Time:\t" +       str(buildModelTime - generateGraphsTime) +  "\tseconds")

    red.reduce_dim()

    reduceTime = time.time()

    print()
    print("Query Time:\t\t" +           str(getDataTime - startTime) +              "\tseconds")
    print("NxGraphs Time:\t\t" +        str(generateGraphsTime - getDataTime) +     "\tseconds")
    print("Dim Reduce Time:\t" +        str(reduceTime - buildModelTime) +          "\tseconds\n")
    print("Model Build Time:\t" +       str(buildModelTime - generateGraphsTime) +  "\tseconds")
    print("Total Time:\t\t" +           str(reduceTime - startTime) +                    "\tseconds")

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

    # def plot_commits(graph):
    #     """
    #     Function to plot the commit graph from the networkx graph.
    #     :param graph: The graph to plot.
    #     :return None:
    #     """
    #     nx.draw_kamada_kawai(graph, alpha=0.5, node_color='blue', node_size = 2)
    #     figure = plt.gcf() # get current figure
    #     figure.set_size_inches(12, 8)
