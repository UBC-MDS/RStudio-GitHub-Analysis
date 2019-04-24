import pandas_gbq
import networkx as nx
import matplotlib.pyplot as plt
import graph2vec as g2v
import time
import numpy.distutils.system_info as sysinfo

from joblib import Parallel, delayed
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm

def query_ght(queryString):
    """
    Function to query with the provided query string.
    :param queryString: String with which to perform the query.
    :return query_result_df: Dataframe that holds the query results.
    """
    query_result_df = pandas_gbq.read_gbq(queryString)

    return query_result_df

def commit_query(projectId):
    """
    Function to generate the query that will pull all commits for a given projectId.
    :param projectId: Project ID that you'd like to get commits for.
    :return queryString: Query string for the given projectId.
    """
    return """
            select
              c.id as c_id,
              p.id as p_id,
              cp.commit_id as cp_commit_id,
              cp.parent_id as cp_parent_id
            from `ghtorrent-bq.ght.commits` c
            left join `ghtorrent-bq.ght.projects` p on (p.id = c.project_id)
            left join `ghtorrent-bq.ght.commit_parents` cp on (cp.commit_id = c.id)
            where (p.id = """ + str(projectId) + """)
            limit 10000
        """

def git_graph(commitData):
    """
    Function to generate the commit graph in networkx from the query results.
    :param commitData: Data pulled from the commit_query string.
    :return nxGraph: Networkx graph
    """
    source_target_commits = commitData[["cp_parent_id", "c_id"]].dropna().astype("int64")
    source_target_commits.columns = ["source", "target"]

    return nx.from_pandas_edgelist(source_target_commits)

def plot_commits(graph):
    """
    Function to plot the commit graph from the networkx graph.
    :param graph: The graph to plot.
    :return None:
    """
    nx.draw_kamada_kawai(graph, alpha=0.5, node_color='blue', node_size = 2)

n_workers    = 4
n_iterations = 1
n_dimensions = 128

if __name__ == '__main__':
    startTime = time.time()

    query_p1 = commit_query(22003900)
    query_p2 = commit_query(33470153)

    data_p1 = query_ght(query_p1)
    data_p2 = query_ght(query_p2)

    getDataTime = time.time()

    graphs = [git_graph(data_p1), git_graph(data_p2)]

    generateGraphsTime = time.time()

    document_collections = Parallel(n_jobs = n_workers)(delayed(g2v.feature_extractor)(graphs[g], n_iterations, str(g)) for g in tqdm(range(len(graphs))))

    featExtractTime = time.time()

    model = Doc2Vec(document_collections,
                    size = n_dimensions,
                    window = 0,
                    min_count = 5,
                    dm = 0,
                    sample = 0.0001,
                    workers = n_workers,
                    iter = n_iterations,
                    alpha = 0.025)

    buildModelTime = time.time()

    g2v.save_embedding("./results/embeddings.csv", model, len(graphs), n_dimensions)

    saveEmbeddingsTime = time.time()

    print("Query Time:\t\t" + str(getDataTime - startTime) + "\tseconds")
    print("NxGraphs Time:\t\t" + str(generateGraphsTime - getDataTime) + "\tseconds")
    print("FeatExtract Time:\t" + str(featExtractTime - generateGraphsTime) + "\tseconds")
    print("Model Build Time:\t" + str(buildModelTime - featExtractTime) + "\tseconds")
    print("Save Embeddings Time:\t" + str(saveEmbeddingsTime - buildModelTime) + "\tseconds")

    # for graph in graphs:
    #     plot_commits(graph)
    #     plt.savefig("./imgs/branch_test")
    #     plt.show()
