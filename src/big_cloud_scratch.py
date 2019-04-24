import pandas_gbq
import networkx as nx
import matplotlib.pyplot as plt
import graph2vec as g2v
import time
from joblib import Parallel, delayed
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy.distutils.system_info as sysinfo
from tqdm import tqdm

def query_ght(queryString):
    # https://bigquery.cloud.google.com/dataset/ghtorrent-bq:ght
    query_result_df = pandas_gbq.read_gbq(queryString)

    return query_result_df

def commit_query(projectId):
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


def git_graph(commits):
    source_target_commits = commits[["cp_parent_id", "c_id"]].dropna().astype("int64")
    source_target_commits.columns = ["source", "target"]

    return nx.from_pandas_edgelist(source_target_commits)

def plot_commits(commits):
    nx.draw_kamada_kawai(git_graph(commits), alpha=0.5, node_color='blue', node_size = 2)

if __name__ == '__main__':
    n_workers = 4
    n_iterations = 1

    query_p1 = commit_query(22003900)
    query_p2 = commit_query(5524547)
    data_p1 = query_ght(query_p1)
    #print(data_p1)
    data_p2 = query_ght(query_p2)
    graphs = [git_graph(data_p1), git_graph(data_p2)]

    document_collections = Parallel(n_jobs = n_workers)(delayed(g2v.feature_extractor)(g, n_iterations) for g in tqdm(graphs))

    model = Doc2Vec(document_collections,
                    size = 128,
                    window = 0,
                    min_count = 5,
                    dm = 0,
                    sample = 0.0001,
                    workers = n_workers,
                    iter = n_iterations,
                    alpha = 0.025)

    print(model)
    #g2v.save_embedding("./out/ok.json", model, graphs, 128)

    #print(document_collections)
    # start = time.time()
    # commits = query_ght(commitQuery)
    # getData = time.time()
    # print("Query Time:\t" + str(getData - start))
    # branchPlot = plot_commits(commits)
    # plotTime = time.time()
    # print("Plot Time:\t" + str(plotTime - getData))
    # plt.savefig("./imgs/branch_test")
    # plt.show()
